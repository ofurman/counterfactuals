import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from counterfactuals.datasets.method_dataset import MethodDataset
from counterfactuals.dequantization.dequantizer import GroupDequantizer
from counterfactuals.dequantization.utils import DequantizationWrapper
from counterfactuals.metrics.metrics import evaluate_cf
from counterfactuals.pipelines.nodes.disc_model_nodes import create_disc_model
from counterfactuals.pipelines.nodes.gen_model_nodes import create_gen_model
from counterfactuals.pipelines.nodes.helper_nodes import set_model_paths


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

    X_cf: Any
    X_test: Any
    y_orig: Any
    y_target: Any
    model_returned: Any
    cf_search_time: float
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
        self.cfg = cfg
        self.logger = logger
        self.preprocessing_pipeline = preprocessing_pipeline

    def run(self) -> None:
        """Orchestrate the full pipeline across all CV folds."""
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        self.logger.info("Loading dataset")
        dataset = self.load_dataset()
        dequantizer = GroupDequantizer(dataset.categorical_features_lists)

        for fold_n, _ in enumerate(dataset.get_cv_splits(5)):
            disc_model_path, gen_model_path, save_folder = set_model_paths(self.cfg, fold=fold_n)

            disc_model = self.create_disc_model(dataset, disc_model_path, save_folder)

            if self.cfg.experiment.relabel_with_disc_model:
                self.relabel_with_disc_model(dataset, disc_model)

            dequantizer.fit(dataset.X_train)
            gen_model = self.create_gen_model(dataset, gen_model_path, dequantizer)

            log_prob_threshold = self.compute_log_prob_threshold(gen_model, dataset, dequantizer)

            result = self.search_counterfactuals(
                dataset, gen_model, disc_model, save_folder, log_prob_threshold
            )

            wrapped_gen_model = DequantizationWrapper(gen_model, dequantizer)

            metrics = self.calculate_metrics(
                wrapped_gen_model, disc_model, dataset, result, log_prob_threshold
            )

            self.save_results(metrics, result.cf_search_time, save_folder)

    def load_dataset(self) -> MethodDataset:
        """Load and preprocess the dataset.

        Override to apply custom dataset handling (e.g. one-hot encoding for AReS).

        Returns:
            Preprocessed :class:`MethodDataset` ready for model training.
        """
        file_dataset = instantiate(self.cfg.dataset)
        return MethodDataset(file_dataset, self.preprocessing_pipeline)

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
        disc_model_name = self.cfg.disc_model.model._target_.split(".")[-1]
        df.to_csv(os.path.join(save_folder, f"cf_metrics_{disc_model_name}.csv"), index=False)

    @staticmethod
    def _filter_test_data(dataset: MethodDataset, target_class: int) -> tuple[Any, Any]:
        """Filter out target class samples from test set.

        Args:
            dataset: Dataset containing X_test and y_test.
            target_class: Class label to exclude from counterfactual generation.

        Returns:
            Tuple of (X_test_filtered, y_test_filtered).
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
            Tuple of (cf_method_name, disc_model_name).
        """
        cf_method_name = self.cfg.counterfactuals_params.cf_method._target_.split(".")[-1]
        disc_model_name = self.cfg.disc_model.model._target_.split(".")[-1]
        return cf_method_name, disc_model_name
