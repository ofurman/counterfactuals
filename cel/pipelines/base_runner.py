import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import time
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from cel.datasets.method_dataset import MethodDataset
from cel.dequantization.dequantizer import GroupDequantizer
from cel.dequantization.utils import DequantizationWrapper
from cel.metrics.metrics import evaluate_cf
from cel.pipelines.config_schema import REQUIRED_CFG_KEYS
from cel.pipelines.nodes.disc_model_nodes import create_disc_model
from cel.pipelines.nodes.gen_model_nodes import create_gen_model
from cel.pipelines.nodes.helper_nodes import set_model_paths
from cel.preprocessing import (
    MinMaxScalingStep,
    PreprocessingPipeline,
    TorchDataTypeStep,
)


def get_log_prob_threshold(
    gen_model: torch.nn.Module,
    dataset: MethodDataset,
    batch_size: int,
    log_prob_quantile: float,
    logger: logging.Logger,
) -> float:
    """Calculate a log-probability threshold over the training set."""
    logger.info("Calculating log_prob_threshold")
    train_dataloader = dataset.train_dataloader(batch_size=batch_size, shuffle=False)
    log_prob_threshold = torch.quantile(
        gen_model.predict_log_prob(train_dataloader),
        log_prob_quantile,
    )
    logger.info(f"log_prob_threshold: {log_prob_threshold:.4f}")
    return log_prob_threshold


@dataclass
class SearchResult:
    """Result of a counterfactual search.

    Attributes:
        X_cf: Generated counterfactual examples.
        X_test: Original test examples used for generation.
        y_orig: Original labels.
        y_target: Target labels for counterfactuals.
        model_returned: Boolean mask indicating successful CF generation per sample.
        cf_search_time: Wall-clock time of the CF search in seconds.
        extras: Method-specific additional outputs (e.g. group IDs, S/D matrices).
    """

    X_cf: np.ndarray
    X_test: np.ndarray
    y_orig: np.ndarray
    y_target: np.ndarray
    model_returned: np.ndarray
    cf_search_time: float
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class CfMethodOutput:
    """Raw output from a CF method invocation.

    Attributes:
        x_cfs: Generated counterfactual examples.
        x_origs: Original input samples.
        y_origs: Original labels.
        y_targets: Target labels for counterfactuals.
        model_returned: Boolean mask for successful CF generation. None means all succeeded.
        extras: Method-specific additional outputs.
    """

    x_cfs: np.ndarray
    x_origs: np.ndarray
    y_origs: np.ndarray
    y_targets: np.ndarray
    model_returned: np.ndarray | None = None
    extras: dict[str, Any] = field(default_factory=dict)


class PipelineRunner(ABC):
    """Template-method base class for counterfactual pipeline runners.

    Subclasses must implement :meth:`search_counterfactuals` and may override any of
    the hook methods to customise dataset loading, model creation, or metric calculation.

    Args:
        cfg: Hydra configuration for the pipeline run.
        logger: Logger instance for structured output.
    """

    cf_method_name: ClassVar[str]

    def __init__(
        self, cfg: DictConfig, logger: logging.Logger, preprocessing_pipeline=None
    ) -> None:
        self._validate_cfg(cfg)
        self.cfg = cfg
        self.logger = logger
        self.preprocessing_pipeline = preprocessing_pipeline

    @staticmethod
    def _validate_cfg(cfg: DictConfig) -> None:
        """Validate that all required config keys are present.

        Checks every key in :data:`~cel.pipelines.config_schema.REQUIRED_CFG_KEYS`
        using :func:`omegaconf.OmegaConf.select`.  Raises :exc:`ValueError` early
        so misconfigured runs fail before any model loading or training.

        Args:
            cfg: Hydra ``DictConfig`` passed to the runner constructor.

        Raises:
            ValueError: If one or more required keys are absent from ``cfg``.
        """
        _SENTINEL = object()
        missing = [
            key
            for key in REQUIRED_CFG_KEYS
            if OmegaConf.select(cfg, key, default=_SENTINEL) is _SENTINEL
        ]
        if missing:
            raise ValueError(
                f"Pipeline config is missing required keys: {missing}\n"
                "Check that your Hydra config includes all mandatory fields "
                "(see cel.pipelines.config_schema)."
            )

    @classmethod
    def default_preprocessing(cls) -> PreprocessingPipeline:
        """Standard preprocessing pipeline: MinMax scaling followed by Torch dtype conversion.

        Returns:
            A :class:`PreprocessingPipeline` with ``("minmax", MinMaxScalingStep())``
            and ``("torch_dtype", TorchDataTypeStep())``.
        """
        return PreprocessingPipeline(
            [
                ("minmax", MinMaxScalingStep()),
                ("torch_dtype", TorchDataTypeStep()),
            ]
        )

    def run(self) -> None:
        """Orchestrate the full pipeline across all CV folds."""
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        self.logger.info("Loading dataset")
        dataset = self.load_dataset()
        dequantizer = GroupDequantizer(dataset.categorical_features_lists)

        for fold_n, _ in enumerate(dataset.get_cv_splits(5)):
            try:
                disc_model_path, gen_model_path, save_folder = set_model_paths(
                    self.cfg, fold=fold_n
                )

                try:
                    disc_model = self.create_disc_model(dataset, disc_model_path, save_folder)
                except (FileNotFoundError, RuntimeError, OSError) as e:
                    self.logger.warning(
                        f"Fold {fold_n}: Failed to create discriminative model at '{disc_model_path}': {e}. "
                        "Skipping to next fold."
                    )
                    continue

                if self.cfg.experiment.relabel_with_disc_model:
                    self.relabel_with_disc_model(dataset, disc_model)

                dequantizer.fit(dataset.X_train)

                try:
                    gen_model = self.create_gen_model(dataset, gen_model_path, dequantizer)
                except (FileNotFoundError, RuntimeError, OSError) as e:
                    self.logger.warning(
                        f"Fold {fold_n}: Failed to create generative model at '{gen_model_path}': {e}. "
                        "Skipping to next fold."
                    )
                    continue

                log_prob_threshold = self.compute_log_prob_threshold(
                    gen_model, dataset, dequantizer
                )

                try:
                    result = self.search_counterfactuals(
                        dataset, gen_model, disc_model, save_folder, log_prob_threshold
                    )
                except (RuntimeError, ValueError, torch.cuda.OutOfMemoryError) as e:
                    self.logger.warning(
                        f"Fold {fold_n}: Counterfactual search failed: {e}. Skipping to next fold."
                    )
                    continue

                wrapped_gen_model = DequantizationWrapper(gen_model, dequantizer)

                try:
                    metrics = self.calculate_metrics(
                        wrapped_gen_model, disc_model, dataset, result, log_prob_threshold
                    )
                    self.save_results(metrics, result.cf_search_time, save_folder)
                except Exception as e:
                    self.logger.warning(
                        f"Fold {fold_n}: Failed to calculate/save metrics: {e}. "
                        "Continuing to next fold."
                    )

            except Exception as e:
                self.logger.error(f"Fold {fold_n}: Unexpected error during pipeline run: {e}")
                continue

    def load_dataset(self) -> MethodDataset:
        """Load and preprocess the dataset.

        Override to apply custom dataset handling (e.g. one-hot encoding for AReS).

        Returns:
            Preprocessed :class:`MethodDataset` ready for model training.

        Raises:
            OSError: If dataset file cannot be found or read.
            ValueError: If dataset configuration is invalid.
        """
        try:
            file_dataset = instantiate(self.cfg.dataset)
            return MethodDataset(file_dataset, self.preprocessing_pipeline)
        except OSError as e:
            dataset_target = self.cfg.dataset.get("_target_", "unknown")
            self.logger.error(f"Failed to load dataset '{dataset_target}': {e}")
            raise
        except ValueError as e:
            self.logger.error(f"Invalid dataset configuration: {e}")
            raise

    def create_disc_model(
        self, dataset: MethodDataset, path: str, save_folder: str
    ) -> torch.nn.Module:
        """Create (or load) the discriminative model.

        Args:
            dataset: The current fold's dataset.
            path: Path used to save/load the model checkpoint.
            save_folder: Directory for auxiliary outputs.

        Returns:
            Trained discriminative model.
        """
        return create_disc_model(self.cfg, dataset, path, save_folder)

    def relabel_with_disc_model(self, dataset: MethodDataset, disc_model: torch.nn.Module) -> None:
        """Relabel training and test sets using discriminative model predictions.

        Args:
            dataset: Dataset whose labels will be overwritten in-place.
            disc_model: Trained discriminative model.
        """
        dataset.y_train = disc_model.predict(dataset.X_train)
        dataset.y_test = disc_model.predict(dataset.X_test)

    def create_gen_model(
        self, dataset: MethodDataset, path: str, dequantizer: GroupDequantizer
    ) -> torch.nn.Module:
        """Create (or load) the generative model.

        Override for methods that require a custom generative setup (e.g. CeFlow).

        Args:
            dataset: The current fold's dataset.
            path: Path used to save/load the model checkpoint.
            dequantizer: Fitted dequantizer passed to the gen model factory.

        Returns:
            Trained generative model.
        """
        return create_gen_model(self.cfg, dataset, path, dequantizer)

    def compute_log_prob_threshold(
        self,
        gen_model: torch.nn.Module,
        dataset: MethodDataset,
        dequantizer: GroupDequantizer,
    ) -> float:
        """Compute the log-probability threshold used for plausibility filtering.

        Temporarily dequantizes ``dataset.X_train``, computes the quantile over
        training log-probabilities, then restores the original training data.

        Args:
            gen_model: Trained generative model.
            dataset: Dataset providing the training dataloader.
            dequantizer: Dequantizer used to transform/inverse-transform X_train.

        Returns:
            Scalar log-probability threshold at the configured quantile.
        """
        self.logger.info("Calculating log_prob_threshold")
        dataset.X_train = dequantizer.transform(dataset.X_train)
        train_dataloader = dataset.train_dataloader(
            batch_size=self.cfg.counterfactuals_params.batch_size, shuffle=False
        )
        log_prob_threshold = torch.quantile(
            gen_model.predict_log_prob(train_dataloader),
            self.cfg.counterfactuals_params.log_prob_quantile,
        )
        dataset.X_train = dequantizer.inverse_transform(dataset.X_train)
        self.logger.info(f"log_prob_threshold: {log_prob_threshold:.4f}")
        return log_prob_threshold

    @abstractmethod
    def search_counterfactuals(
        self,
        dataset: MethodDataset,
        gen_model: torch.nn.Module,
        disc_model: torch.nn.Module,
        save_folder: str,
        log_prob_threshold: float,
    ) -> SearchResult:
        """Generate counterfactuals for the current fold.

        Args:
            dataset: The current fold's dataset.
            gen_model: Trained generative model.
            disc_model: Trained discriminative model.
            save_folder: Directory for saving generated counterfactuals.
            log_prob_threshold: Plausibility threshold from :meth:`compute_log_prob_threshold`.

        Returns:
            :class:`SearchResult` with counterfactuals and timing information.
        """
        ...

    # --- Template-method hooks (override in subclasses) ---

    def create_cf_method(
        self,
        dataset: MethodDataset,
        gen_model: torch.nn.Module,
        disc_model: torch.nn.Module,
    ) -> object:
        """Instantiate the CF method object.

        Required hook for :meth:`_default_search_counterfactuals`.

        Args:
            dataset: The current fold's dataset.
            gen_model: Trained generative model.
            disc_model: Trained discriminative model.

        Returns:
            Instantiated CF method.

        Raises:
            NotImplementedError: Subclasses must implement this hook.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement create_cf_method()")

    def pre_cf_generation(self, cf_method: object, dataset: MethodDataset) -> None:
        """Optional hook called after CF method creation, before CF generation.

        Default implementation is a no-op. Override for methods that require
        a fitting step (e.g. density estimator fitting in Artelt).

        Args:
            cf_method: The CF method instance.
            dataset: The current fold's dataset.
        """

    def run_cf_method(
        self,
        cf_method: object,
        cf_dataloader: torch.utils.data.DataLoader,
        dataset: MethodDataset,
        log_prob_threshold: float,
    ) -> CfMethodOutput:
        """Run the CF method and return raw outputs.

        Required hook for :meth:`_default_search_counterfactuals`.

        Args:
            cf_method: The CF method instance from :meth:`create_cf_method`.
            cf_dataloader: DataLoader for the filtered test set.
            dataset: The current fold's dataset.
            log_prob_threshold: Plausibility threshold.

        Returns:
            :class:`CfMethodOutput` with raw CF generation results.

        Raises:
            NotImplementedError: Subclasses must implement this hook.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement run_cf_method()")

    def postprocess_cf_output(
        self, output: CfMethodOutput, dataset: MethodDataset
    ) -> CfMethodOutput:
        """Optional hook to post-process raw CF output before saving.

        Default implementation returns the output unchanged. Override for methods
        that require categorical discretization or other post-processing.

        Args:
            output: Raw output from :meth:`run_cf_method`.
            dataset: The current fold's dataset.

        Returns:
            Processed :class:`CfMethodOutput`.
        """
        return output

    def _default_search_counterfactuals(
        self,
        dataset: MethodDataset,
        gen_model: torch.nn.Module,
        disc_model: torch.nn.Module,
        save_folder: str,
        log_prob_threshold: float,
    ) -> SearchResult:
        """Template-method implementation of counterfactual search.

        Orchestrates the standard CF generation pipeline using the hook methods
        :meth:`create_cf_method`, :meth:`pre_cf_generation`, :meth:`run_cf_method`,
        and :meth:`postprocess_cf_output`. Simple runners opt in by calling this
        from :meth:`search_counterfactuals`.

        Args:
            dataset: The current fold's dataset.
            gen_model: Trained generative model.
            disc_model: Trained discriminative model.
            save_folder: Directory for saving generated counterfactuals.
            log_prob_threshold: Plausibility threshold.

        Returns:
            :class:`SearchResult` with counterfactuals and timing information.
        """
        disc_model_name = self._get_disc_model_name()
        target_class = self._get_target_class()

        X_test, y_test = self._filter_test_data(dataset, target_class)
        cf_method = self.create_cf_method(dataset, gen_model, disc_model)
        cf_dataloader = self._create_cf_dataloader(
            X_test, y_test, self.cfg.counterfactuals_params.batch_size
        )
        self.pre_cf_generation(cf_method, dataset)

        with self._timed_search() as timer:
            output = self.run_cf_method(cf_method, cf_dataloader, dataset, log_prob_threshold)
        cf_search_time = timer["elapsed"]

        output = self.postprocess_cf_output(output, dataset)

        if output.model_returned is None:
            model_returned = np.ones(output.x_cfs.shape[0], dtype=bool)
        else:
            model_returned = output.model_returned

        self._save_counterfactuals(output.x_cfs, save_folder, self.cf_method_name, disc_model_name)

        return SearchResult(
            X_cf=output.x_cfs,
            X_test=output.x_origs,
            y_orig=output.y_origs,
            y_target=output.y_targets,
            model_returned=model_returned,
            cf_search_time=cf_search_time,
        )

    def calculate_metrics(
        self,
        gen_model: torch.nn.Module,
        disc_model: torch.nn.Module,
        dataset: MethodDataset,
        result: SearchResult,
        log_prob_threshold: float,
    ) -> dict[str, Any]:
        """Compute evaluation metrics for the generated counterfactuals.

        Override for group methods (GLANCE, PUMAL) that require specialised metric functions.

        Args:
            gen_model: Wrapped generative model (with dequantization).
            disc_model: Trained discriminative model.
            dataset: The current fold's dataset.
            result: Output of :meth:`search_counterfactuals`.
            log_prob_threshold: Plausibility threshold from :meth:`compute_log_prob_threshold`.

        Returns:
            Dictionary of metric name → value.
        """
        self.logger.info("Calculating metrics")
        metrics = evaluate_cf(
            disc_model=disc_model,
            gen_model=gen_model,
            X_cf=result.X_cf,
            model_returned=result.model_returned,
            categorical_features=dataset.categorical_features_indices,
            continuous_features=dataset.numerical_features_indices,
            X_train=dataset.X_train,
            y_train=dataset.y_train.reshape(-1),
            X_test=result.X_test,
            y_test=result.y_orig,
            y_target=result.y_target,
            median_log_prob=log_prob_threshold,
        )
        self.logger.info(f"Metrics:\n{metrics}")
        return metrics

    def save_results(
        self, metrics: dict[str, Any], cf_search_time: float, save_folder: str
    ) -> None:
        """Persist metrics to a CSV file in the save folder.

        Args:
            metrics: Dictionary of metric name → value.
            cf_search_time: Duration of the CF search in seconds.
            save_folder: Directory where the CSV will be written.
        """
        df = pd.DataFrame(metrics, index=[0])
        df["cf_search_time"] = cf_search_time
        disc_model_name = self._get_disc_model_name()
        df.to_csv(os.path.join(save_folder, f"cf_metrics_{disc_model_name}.csv"), index=False)

    @staticmethod
    def _filter_test_data(dataset: MethodDataset, target_class: int) -> tuple[Any, Any]:
        """Filter out target class samples from test set.

        Args:
            dataset: Dataset containing X_test and y_test.
            target_class: Class label to exclude from counterfactual generation.

        Returns:
            tuple of (X_test_filtered, y_test_filtered).
        """
        X_test_origin = dataset.X_test[dataset.y_test != target_class]
        y_test_origin = dataset.y_test[dataset.y_test != target_class]
        return X_test_origin, y_test_origin

    @staticmethod
    def _create_cf_dataloader(
        X_test: Any, y_test: Any, batch_size: int
    ) -> torch.utils.data.DataLoader:
        """Create DataLoader for counterfactual generation.

        Args:
            X_test: Test features.
            y_test: Test labels.
            batch_size: Batch size for DataLoader.

        Returns:
            DataLoader with TensorDataset containing X_test and y_test.
        """
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.tensor(X_test).float(),
                torch.tensor(y_test).float(),
            ),
            batch_size=batch_size,
            shuffle=False,
        )

    @staticmethod
    def _save_counterfactuals(
        X_cf: Any, save_folder: str, cf_method_name: str, disc_model_name: str
    ) -> str:
        """Save counterfactuals to CSV file.

        Args:
            X_cf: Generated counterfactual examples.
            save_folder: Directory to save the CSV file.
            cf_method_name: Name of the CF method (for filename).
            disc_model_name: Name of the discriminative model (for filename).

        Returns:
            Path to the saved CSV file.
        """
        counterfactuals_path = os.path.join(
            save_folder, f"counterfactuals_{cf_method_name}_{disc_model_name}.csv"
        )
        pd.DataFrame(X_cf).to_csv(counterfactuals_path, index=False)
        return counterfactuals_path

    def _get_method_names(self) -> tuple[str, str]:
        """Extract CF method and disc model names from config.

        Returns:
            tuple of (cf_method_name, disc_model_name).
        """
        cf_method_name = self.cfg.counterfactuals_params.cf_method._target_.split(".")[-1]
        disc_model_name = self._get_disc_model_name()
        return cf_method_name, disc_model_name

    def _get_disc_model_name(self) -> str:
        """Extract the discriminative model class name from config.

        Returns:
            Class name of the discriminative model.
        """
        return self.cfg.disc_model.model._target_.split(".")[-1]

    def _get_target_class(self) -> int:
        """Extract the target class from config.

        Returns:
            Target class index for counterfactual generation.
        """
        return self.cfg.counterfactuals_params.target_class

    @contextmanager
    def _timed_search(self) -> Generator[dict[str, float], None, None]:
        """Context manager that measures wall-clock time of the CF search.

        Yields a mutable dict; ``result["elapsed"]`` is set on block exit.
        """
        result: dict[str, float] = {"elapsed": 0.0}
        start = time()
        yield result
        result["elapsed"] = time() - start
        self.logger.info("Counterfactual search completed in %.4f seconds", result["elapsed"])

    @staticmethod
    def _run_cf_generation_safely(
        cf_method: object,
        X: np.ndarray,
        y_origin: np.ndarray,
        y_target: np.ndarray,
        logger: logging.Logger,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run CF generation with per-sample error handling.

        This helper method wraps counterfactual generation to gracefully handle
        failures on individual samples or batches. Failed samples are marked with
        NaN in the counterfactuals array and False in the model_returned mask.

        Args:
            cf_method: Counterfactual generation method instance.
            X: Original input samples.
            y_origin: Original labels.
            y_target: Target labels for counterfactuals.
            logger: Logger instance for warning messages.
            **kwargs: Additional arguments passed to the CF method.

        Returns:
            tuple of (counterfactuals, model_returned mask).
            Failed samples have NaN counterfactuals and False in model_returned.
        """
        n_samples = len(X)
        n_features = X.shape[1]

        # Initialize output arrays with NaN and False
        X_cf = np.full((n_samples, n_features), np.nan, dtype=np.float32)
        model_returned = np.zeros(n_samples, dtype=bool)

        for i in range(n_samples):
            try:
                X_i = X[i : i + 1]
                y_origin_i = y_origin[i : i + 1]
                y_target_i = y_target[i : i + 1]

                result = cf_method.generate(
                    X=X_i, y_origin=y_origin_i, y_target=y_target_i, **kwargs
                )

                X_cf[i] = result
                model_returned[i] = True

            except Exception as e:
                logger.warning(f"CF generation failed for sample {i}: {e}")
                X_cf[i] = np.nan
                model_returned[i] = False

        failed_count = n_samples - model_returned.sum()
        if failed_count > 0:
            logger.warning(
                f"CF generation completed with {failed_count}/{n_samples} failed samples "
                f"({100 * failed_count / n_samples:.1f}%)"
            )

        return X_cf, model_returned
