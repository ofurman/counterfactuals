"""Pipeline runner for the DiCoFlex counterfactual method.

DiCoFlex conditions a normalising flow on ``[factual, target class, mask, p]`` and
samples counterfactuals directly, so it departs from the plain generative-model
contract in three ways that the hooks below implement:

* the flow is conditional and trains on mined factual/neighbour pairs rather than
  on the raw training matrix, so :meth:`DiCoFlexPipelineRunner.create_gen_model`
  builds its own dataloaders and training loop;
* the plausibility threshold is therefore computed over those same context-carrying
  batches;
* metrics need the context that produced each row, supplied by
  :class:`DiCoFlexGeneratorMetricsAdapter`.

The dataloaders built during training are reused at inference (they carry the
class index map and the mask catalogue), so they are kept on the instance between
hook calls.
"""

import logging
from typing import Any

import numpy as np
import torch
import torch.utils.data
from hydra.utils import instantiate
from omegaconf import DictConfig

from cel.cf_methods.local_methods.dicoflex import DiCoFlex, DiCoFlexParams
from cel.cf_methods.local_methods.dicoflex.context_utils import (
    DiCoFlexGeneratorMetricsAdapter,
    build_context_matrix,
    get_numpy_pointer,
)
from cel.cf_methods.local_methods.dicoflex.data import (
    build_actionability_mask,
    build_monotonic_direction_vector,
    create_dicoflex_dataloaders,
)
from cel.datasets.method_dataset import MethodDataset
from cel.dequantization.dequantizer import GroupDequantizer
from cel.metrics.metrics import evaluate_cf
from cel.pipelines.base_runner import CfMethodOutput, PipelineRunner, SearchResult
from cel.pipelines.nodes.factual_selection import resolve_target_labels, select_factual_indices
from cel.pipelines.utils import apply_categorical_discretization

logger = logging.getLogger(__name__)

# Generation-space values live in ~[0, 1]; anything past this bound is a dead
# flow sample whose inverse transform would overflow float32.
_OVERFLOW_BOUND = 1e6


def build_masks(dataset: MethodDataset, cf_params: DictConfig) -> list[np.ndarray]:
    """Assemble the mask catalogue used during DiCoFlex training.

    Args:
        dataset: Dataset the masks are built against.
        cf_params: The ``counterfactuals_params`` config node.

    Returns:
        List of mask vectors, each of length ``n_features``. Falls back to a
        single all-ones mask when no mask is configured.

    Raises:
        ValueError: If a custom mask length does not match the feature dimension.
    """
    masks: list[np.ndarray] = []
    monotonic_overrides = dict(cf_params.get("monotonic_overrides") or {})
    if cf_params.use_actionability_mask:
        masks.append(build_actionability_mask(dataset, extra_actionable=monotonic_overrides.keys()))
    for custom_mask in cf_params.get("custom_masks", []):
        mask_vec = np.asarray(custom_mask, dtype=np.float32).reshape(-1)
        if mask_vec.shape[0] != dataset.X_train.shape[1]:
            raise ValueError(
                "Custom mask length does not match the feature dimension after preprocessing."
            )
        masks.append(mask_vec)
    if not masks:
        masks.append(np.ones(dataset.X_train.shape[1], dtype=np.float32))
    return masks


def train_dicoflex_generator(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    cfg: DictConfig,
    model_path: str,
    device: torch.device,
) -> None:
    """Train the conditional flow with early stopping on the validation loss.

    The best checkpoint is written to ``model_path`` as it improves and reloaded
    once training ends, so the caller always holds the best-scoring weights.

    Args:
        model: Conditional flow to train.
        train_loader: Loader over mined factual/neighbour pairs.
        val_loader: Held-out slice of the same pairs.
        cfg: Full pipeline config; reads ``gen_model.{lr,epochs,patience,eps}``.
        model_path: Where the best checkpoint is written.
        device: Device to train on.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.gen_model.lr)
    best_val = float("inf")
    patience_counter = 0
    eps = cfg.gen_model.get("eps", 1e-5)

    for epoch in range(cfg.gen_model.epochs):
        model.train()
        train_loss = 0.0
        for batch_cf, batch_context in train_loader:
            batch_cf = batch_cf.reshape(-1, batch_cf.shape[-1]).to(device)
            batch_context = batch_context.reshape(-1, batch_context.shape[-1]).to(device)
            optimizer.zero_grad()
            loss = -model(batch_cf, context=batch_context).mean()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_cf, batch_context in val_loader:
                batch_cf = batch_cf.reshape(-1, batch_cf.shape[-1]).to(device)
                batch_context = batch_context.reshape(-1, batch_context.shape[-1]).to(device)
                val_loss += (-model(batch_cf, context=batch_context).mean()).item()
        val_loss /= max(1, len(val_loader))
        logger.info("Epoch %s | train loss %.4f | val loss %.4f", epoch, train_loss, val_loss)

        if val_loss < best_val - eps:
            best_val = val_loss
            patience_counter = 0
            model.save(model_path)
        else:
            patience_counter += 1
            if patience_counter > cfg.gen_model.patience:
                logger.info("Early stopping after %s epochs", epoch + 1)
                break

    model.load(model_path)


def flow_log_prob_threshold(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    quantile: float,
    device: torch.device,
) -> float:
    """Estimate a plausibility threshold from training log probabilities.

    Args:
        model: Trained conditional flow.
        dataloader: Loader over the full set of training pairs.
        quantile: Quantile of the log-probability distribution to cut at.
        device: Device the flow lives on.

    Returns:
        The log-probability value at ``quantile``.
    """
    log_probs = []
    model.eval()
    with torch.no_grad():
        for batch_cf, batch_context in dataloader:
            batch_cf = batch_cf.reshape(-1, batch_cf.shape[-1]).to(device)
            batch_context = batch_context.reshape(-1, batch_context.shape[-1]).to(device)
            log_probs.append(model(batch_cf, context=batch_context).cpu())
    return torch.quantile(torch.cat(log_probs), quantile).item()


def full_training_loader(
    subset_loader: torch.utils.data.DataLoader, batch_size: int
) -> torch.utils.data.DataLoader:
    """Create a loader over the complete DiCoFlex dataset behind a train split.

    Args:
        subset_loader: The training loader, whose dataset may be a ``Subset``.
        batch_size: Batch size for the new loader.

    Returns:
        A loader iterating every mined pair, train and validation alike.
    """
    base_dataset = (
        subset_loader.dataset.dataset
        if hasattr(subset_loader.dataset, "dataset")
        else subset_loader.dataset
    )
    return torch.utils.data.DataLoader(base_dataset, batch_size=batch_size, shuffle=False)


class DiCoFlexPipelineRunner(PipelineRunner):
    """Pipeline runner for DiCoFlex counterfactual generation."""

    cf_method_name = "DiCoFlex"

    def __init__(
        self, cfg: DictConfig, logger: logging.Logger, preprocessing_pipeline=None
    ) -> None:
        """Initialise the runner and resolve the flow-training device.

        The GPU, when requested via ``experiment.use_gpu``, accelerates flow
        training only. The trained flow is moved back to CPU before anything
        else touches it, so generation, the classifier, and metrics all run on
        CPU and checkpoints are CPU-loadable by default.

        Args:
            cfg: Hydra configuration for the pipeline run.
            logger: Logger instance for structured output.
            preprocessing_pipeline: Preprocessing applied to the dataset.
        """
        super().__init__(cfg, logger, preprocessing_pipeline)
        use_gpu = cfg.experiment.get("use_gpu", False)
        self._train_device = torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        )
        self._flow: torch.nn.Module | None = None
        self._disc_model: torch.nn.Module | None = None
        self._train_loader: torch.utils.data.DataLoader | None = None
        self._class_to_index: dict[int, int] = {}
        self._mask_vectors: list[np.ndarray] = []

    def create_disc_model(
        self, dataset: MethodDataset, path: str, save_folder: str
    ) -> torch.nn.Module:
        """Create the classifier and keep a handle for the neighbour filter.

        Args:
            dataset: The current fold's dataset.
            path: Path used to save/load the model checkpoint.
            save_folder: Directory for auxiliary outputs.

        Returns:
            Trained discriminative model.
        """
        self._disc_model = super().create_disc_model(dataset, path, save_folder)
        return self._disc_model

    def _neighbor_target_eligibility(self, dataset: MethodDataset) -> np.ndarray | None:
        """Mask of training points confident enough to serve as neighbour targets.

        Mirrors the reference implementation's ``prob_threshold``: a training
        point may be mined as a neighbour target only when the classifier
        assigns its (relabeled) class at least ``neighbor_prob_threshold``
        probability, so the flow learns to land in confidently-classified
        regions. A threshold of 0 disables the filter.

        Args:
            dataset: The current fold's dataset, already relabeled.

        Returns:
            Boolean mask over the training rows, or None when disabled.
        """
        threshold = float(
            self.cfg.counterfactuals_params.get("neighbor_prob_threshold", 0.0) or 0.0
        )
        if threshold <= 0.0 or self._disc_model is None:
            return None
        train_probs = self._disc_model.predict_proba(dataset.X_train)
        own_class_conf = np.take_along_axis(
            train_probs, dataset.y_train.reshape(-1, 1).astype(int), axis=1
        ).reshape(-1)
        eligibility = own_class_conf >= threshold
        self.logger.info(
            "Neighbor target filter: %d of %d training points pass prob >= %.2f",
            int(eligibility.sum()),
            len(eligibility),
            threshold,
        )
        return eligibility

    def create_gen_model(
        self, dataset: MethodDataset, path: str, dequantizer: GroupDequantizer
    ) -> torch.nn.Module:
        """Mine factual/neighbour pairs and train the conditional flow on them.

        The dequantizer is unused: DiCoFlex trains on mined pairs in the model
        space rather than on dequantized rows.

        Args:
            dataset: The current fold's dataset.
            path: Path used to save/load the flow checkpoint.
            dequantizer: Unused, kept for the base-class signature.

        Returns:
            The trained conditional flow.
        """
        cf_params = self.cfg.counterfactuals_params
        masks = build_masks(dataset, cf_params)
        (
            self._train_loader,
            val_loader,
            self._class_to_index,
            self._mask_vectors,
            context_dim,
        ) = create_dicoflex_dataloaders(
            dataset.X_train,
            dataset.y_train,
            masks=masks,
            p_values=list(cf_params.p_values),
            n_neighbors=cf_params.n_neighbors,
            noise_level=cf_params.noise_level,
            categorical_noise_level=cf_params.get("categorical_noise_level", 0.08),
            factual_batch_size=cf_params.train_batch_factuals,
            val_ratio=cf_params.val_ratio,
            seed=self.cfg.experiment.get("seed", 42),
            numerical_indices=dataset.numerical_features_indices,
            categorical_indices=dataset.categorical_features_indices,
            factual_chunk_size=cf_params.get("neighbor_factual_chunk_size"),
            target_chunk_size=cf_params.get("neighbor_target_chunk_size"),
            target_eligibility=self._neighbor_target_eligibility(dataset),
        )

        gen_model = instantiate(
            self.cfg.gen_model.model,
            features=dataset.X_train.shape[1],
            context_features=context_dim,
        )

        if self.cfg.gen_model.train_model:
            gen_model.to(self._train_device)
            train_dicoflex_generator(
                gen_model, self._train_loader, val_loader, self.cfg, path, self._train_device
            )
            # Inference is CPU-only: bring the flow back and rewrite the best
            # checkpoint with CPU tensors so it loads anywhere without
            # map_location gymnastics.
            gen_model.to(torch.device("cpu"))
            gen_model.save(path)
        else:
            gen_model.load(path)
        self._flow = gen_model
        return gen_model

    def compute_log_prob_threshold(
        self,
        gen_model: torch.nn.Module,
        dataset: MethodDataset,
        dequantizer: GroupDequantizer,
    ) -> float:
        """Compute the plausibility threshold over the mined training pairs.

        Args:
            gen_model: The trained conditional flow.
            dataset: The current fold's dataset (unused).
            dequantizer: Unused, kept for the base-class signature.

        Returns:
            Scalar log-probability threshold at the configured quantile.
        """
        cf_params = self.cfg.counterfactuals_params
        loader = full_training_loader(self._train_loader, cf_params.train_batch_factuals)
        threshold = flow_log_prob_threshold(
            gen_model, loader, cf_params.log_prob_quantile, next(gen_model.parameters()).device
        )
        self.logger.info(f"log_prob_threshold: {threshold:.4f}")
        return threshold

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
            gen_model: Trained conditional flow.
            disc_model: Trained discriminative model.
            save_folder: Directory for saving generated counterfactuals.
            log_prob_threshold: Plausibility threshold.

        Returns:
            :class:`SearchResult` with counterfactuals and timing information.
        """
        return self._default_search_counterfactuals(
            dataset, gen_model, disc_model, save_folder, log_prob_threshold
        )

    def create_cf_method(
        self,
        dataset: MethodDataset,
        gen_model: torch.nn.Module,
        disc_model: torch.nn.Module,
    ) -> DiCoFlex:
        """Instantiate DiCoFlex over the trained flow and classifier.

        Args:
            dataset: The current fold's dataset.
            gen_model: Trained conditional flow.
            disc_model: Trained discriminative model.

        Returns:
            The configured :class:`DiCoFlex` instance.
        """
        cf_params = self.cfg.counterfactuals_params
        params = DiCoFlexParams(
            mask_index=cf_params.inference_mask_index,
            p_value=cf_params.inference_p_value,
            num_counterfactuals=cf_params.num_counterfactuals,
            target_class=cf_params.target_class,
            sampling_batch_size=cf_params.sampling_batch_size,
            cf_samples_per_factual=cf_params.cf_samples_per_factual,
            temperature=cf_params.get("temperature", 1.0),
        )
        monotonic_overrides = dict(cf_params.get("monotonic_overrides") or {})
        monotonic_direction = build_monotonic_direction_vector(
            dataset, overrides=monotonic_overrides
        )
        if np.any(monotonic_direction != 0):
            self.logger.info(
                "Monotonic constraints active on %d feature(s). Overrides: %s",
                int(np.sum(monotonic_direction != 0)),
                monotonic_overrides,
            )
        return DiCoFlex(
            gen_model=gen_model,
            disc_model=disc_model,
            class_to_index=self._class_to_index,
            mask_vectors=self._mask_vectors,
            params=params,
            device="cpu",
            monotonic_direction=monotonic_direction,
        )

    def run_cf_method(
        self,
        cf_method: DiCoFlex,
        cf_dataloader: torch.utils.data.DataLoader,
        dataset: MethodDataset,
        log_prob_threshold: float,
    ) -> CfMethodOutput:
        """Sample counterfactuals for every filtered test row.

        Args:
            cf_method: The DiCoFlex instance.
            cf_dataloader: Loader over the filtered test set.
            dataset: The current fold's dataset.
            log_prob_threshold: Plausibility threshold (unused by DiCoFlex).

        Returns:
            :class:`CfMethodOutput` with one block of counterfactuals per factual.
        """
        X_test = np.concatenate([batch[0].numpy() for batch in cf_dataloader])
        y_test = np.concatenate([batch[1].numpy() for batch in cf_dataloader])
        # target_class None flips each factual's own label, so one run covers
        # both directions; an integer pushes every factual to that class.
        y_target = resolve_target_labels(y_test, self._get_target_class())

        result = cf_method.explain(X=X_test, y_origin=y_test, y_target=y_target)
        model_returned = np.asarray(result.logs.get("model_returned_mask", []), dtype=bool)
        if model_returned.size == 0:
            model_returned = np.ones(result.x_cfs.shape[0], dtype=bool)

        return CfMethodOutput(
            x_cfs=np.ascontiguousarray(result.x_cfs, dtype=np.float32),
            x_origs=np.ascontiguousarray(result.x_origs, dtype=np.float32),
            y_origs=result.y_origs,
            y_targets=result.y_cf_targets,
            model_returned=model_returned,
        )

    def _filter_test_data(
        self, dataset: MethodDataset, target_class: int | None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Select the query set: seeded, optionally capped, both-direction aware.

        Overrides the base filter so ``target_class: null`` explains every test
        row towards its own flip, and ``n_test_samples`` caps the query set with
        a dedicated generator, keeping it identical across reruns of a seed.

        Args:
            dataset: The current fold's dataset, already relabelled.
            target_class: Fixed target class, or None for per-instance flips.

        Returns:
            Tuple of (X_test, y_test) restricted to the selected rows.
        """
        indices = select_factual_indices(
            dataset.y_test,
            target_class=target_class,
            n_test_samples=self.cfg.counterfactuals_params.get("n_test_samples"),
            seed=self.cfg.experiment.get("seed", 42),
        )
        return dataset.X_test[indices], dataset.y_test[indices]

    def postprocess_cf_output(
        self, output: CfMethodOutput, dataset: MethodDataset
    ) -> CfMethodOutput:
        """Repair dead samples, snap categoricals, and bound the numeric tail.

        A flow sampled at temperature < 1 is bounded in practice but not in
        principle, so the numeric columns are clipped to the training box the
        MinMax model space defines. Without it a handful of far samples dominate
        the mean-based proximity metrics.

        Args:
            output: Raw output from :meth:`run_cf_method`.
            dataset: The current fold's dataset.

        Returns:
            The processed :class:`CfMethodOutput`.
        """
        x_cfs = output.x_cfs.copy()
        dead = ~np.isfinite(x_cfs).all(axis=1) | (np.abs(x_cfs) >= _OVERFLOW_BOUND).any(axis=1)
        if np.any(dead):
            self.logger.info(
                "Replacing %d failed counterfactual row(s) with the original factual",
                int(np.sum(dead)),
            )
            x_cfs[dead] = output.x_origs[dead]

        x_cfs = apply_categorical_discretization(dataset.categorical_features_lists, x_cfs)

        if self.cfg.experiment.get("clamp_numeric_to_box", True):
            num_idx = dataset.numerical_features_indices
            if len(num_idx) > 0:
                x_cfs[:, num_idx] = np.clip(x_cfs[:, num_idx], 0.0, 1.0)

        output.x_cfs = x_cfs
        return output

    def calculate_metrics(
        self,
        gen_model: torch.nn.Module,
        disc_model: torch.nn.Module,
        dataset: MethodDataset,
        result: SearchResult,
        log_prob_threshold: float,
    ) -> dict[str, Any]:
        """Score the closest counterfactual of each factual.

        DiCoFlex emits ``cf_samples_per_factual`` counterfactuals per query,
        ordered closest-valid-first, so metrics read the leading row of each
        block while the CSV keeps the full set.

        Args:
            gen_model: Dequantization-wrapped model from the base runner. It is
                deliberately unused: that wrapper calls the flow without a
                context, which a conditional flow cannot accept, so scoring goes
                through the raw flow behind a context-aware adapter instead.
            disc_model: Trained discriminative model.
            dataset: The current fold's dataset.
            result: Output of :meth:`search_counterfactuals`.
            log_prob_threshold: Plausibility threshold.

        Returns:
            Dictionary of metric name to value.
        """
        cf_params = self.cfg.counterfactuals_params
        stride = cf_params.cf_samples_per_factual
        x_cf = np.ascontiguousarray(result.X_cf[::stride], dtype=np.float32)
        x_orig = np.ascontiguousarray(result.X_test[::stride], dtype=np.float32)
        y_orig = result.y_orig[::stride]
        y_target = result.y_target[::stride]
        model_returned = result.model_returned[::stride]

        mask_vector = self._mask_vectors[cf_params.inference_mask_index]
        context_lookup = {
            get_numpy_pointer(x_cf): build_context_matrix(
                factual_points=x_orig,
                labels=y_target,
                mask_vector=mask_vector,
                p_value=cf_params.inference_p_value,
                class_to_index=self._class_to_index,
            ),
            get_numpy_pointer(x_orig): build_context_matrix(
                factual_points=x_orig,
                labels=y_orig,
                mask_vector=mask_vector,
                p_value=cf_params.inference_p_value,
                class_to_index=self._class_to_index,
            ),
        }

        self.logger.info("Calculating metrics")
        metrics = evaluate_cf(
            disc_model=disc_model,
            gen_model=DiCoFlexGeneratorMetricsAdapter(
                base_model=self._flow, context_lookup=context_lookup
            ),
            X_cf=x_cf,
            model_returned=model_returned,
            categorical_features=dataset.categorical_features_indices,
            continuous_features=dataset.numerical_features_indices,
            X_train=dataset.X_train,
            y_train=dataset.y_train.reshape(-1),
            X_test=x_orig,
            y_test=y_orig,
            y_target=y_target,
            median_log_prob=log_prob_threshold,
        )
        self.logger.info(f"Metrics:\n{metrics}")
        return metrics
