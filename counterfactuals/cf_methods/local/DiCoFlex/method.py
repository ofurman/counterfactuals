"""Implementation of the DiCoFlex counterfactual method."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from counterfactuals.cf_methods.counterfactual_base import (
    BaseCounterfactualMethod,
    ExplanationResult,
)
from counterfactuals.cf_methods.local_counterfactual_mixin import (
    LocalCounterfactualMixin,
)
from counterfactuals.models.generative_mixin import GenerativePytorchMixin
from counterfactuals.models.pytorch_base import PytorchBase


@dataclass
class DiCoFlexParams:
    """Container for inference-time DiCoFlex settings."""

    mask_index: int
    p_value: float
    num_counterfactuals: int
    target_class: int
    sampling_batch_size: int


class DiCoFlex(BaseCounterfactualMethod, LocalCounterfactualMixin):
    """Sample-based counterfactual generator backed by a conditional flow."""

    def __init__(
        self,
        gen_model: GenerativePytorchMixin,
        disc_model: PytorchBase,
        class_to_index: Dict[int, int],
        mask_vectors: list[np.ndarray],
        params: DiCoFlexParams,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(
            gen_model=gen_model,
            disc_model=disc_model,
            device=device,
        )
        if not mask_vectors:
            raise ValueError("At least one mask vector must be supplied for DiCoFlex.")
        if params.mask_index >= len(mask_vectors):
            raise ValueError("mask_index exceeds available mask vectors.")
        if params.target_class not in class_to_index:
            raise ValueError(
                f"Target class {params.target_class} not observed in the training data."
            )
        self.class_to_index = class_to_index
        self.mask_vectors = mask_vectors
        self.params = params
        self.device = device or "cpu"
        self.gen_model.to(self.device)
        self.disc_model.to(self.device)

    def explain(
        self,
        X: np.ndarray,
        y_origin: np.ndarray,
        y_target: np.ndarray,
        X_train: Optional[np.ndarray] = None,
        y_train: Optional[np.ndarray] = None,
        **kwargs,
    ) -> ExplanationResult:
        """Generate counterfactuals for the provided samples."""
        x_np = np.asarray(X, dtype=np.float32)
        y_origin_vec = np.asarray(y_origin).reshape(-1, 1)
        y_target_vec = np.asarray(y_target).reshape(-1, 1)
        cf_batch, target_probs, valid_mask, log_probs = self._sample_counterfactuals(
            x_np, y_target_vec
        )
        logs = {
            "sampling/mean_target_probability": float(target_probs.mean()),
            "sampling/valid_ratio": float(valid_mask.mean()),
            "sampling/log_prob_mean": float(log_probs.mean()),
            "model_returned_mask": valid_mask.tolist(),
        }
        return ExplanationResult(
            x_cfs=cf_batch,
            y_cf_targets=y_target_vec,
            x_origs=x_np,
            y_origs=y_origin_vec,
            logs=logs,
        )

    def explain_dataloader(
        self,
        dataloader,
        epochs: int,
        lr: float,
        patience_eps: float = 1e-5,
        **kwargs,
    ) -> ExplanationResult:
        """Adapter around explain() for DataLoader inputs."""
        xs, ys = [], []
        for batch_x, batch_y in dataloader:
            xs.append(batch_x.numpy())
            ys.append(batch_y.numpy())
        X = np.vstack(xs)
        y_origin = np.concatenate(ys)
        y_target = np.full_like(y_origin, fill_value=self.params.target_class)
        return self.explain(
            X=X,
            y_origin=y_origin,
            y_target=y_target,
        )

    def _sample_counterfactuals(
        self, X: np.ndarray, y_target: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample candidate counterfactuals and pick the best per instance."""
        batch_size = self.params.sampling_batch_size
        selected_cf = []
        selected_probs = []
        selected_valid = []
        selected_log_probs = []

        for start in range(0, len(X), batch_size):
            end = start + batch_size
            X_batch = X[start:end]
            y_target_batch = y_target[start:end]
            context = self._build_context(X_batch, y_target_batch)
            samples, log_probs = self.gen_model.sample_and_log_proba(
                n_samples=self.params.num_counterfactuals, context=context
            )
            # Flow returns samples with shape (batch, n_samples, features).
            cf_candidates = samples
            candidate_log_probs = log_probs
            (
                batch_cf,
                batch_probs,
                batch_valid,
                best_indices,
            ) = self._select_best_candidates(X_batch, y_target_batch, cf_candidates)
            selected_cf.append(batch_cf)
            selected_probs.append(batch_probs)
            selected_valid.append(batch_valid)
            selected_log_probs.append(
                candidate_log_probs[np.arange(len(X_batch)), best_indices]
            )

        return (
            np.vstack(selected_cf),
            np.concatenate(selected_probs),
            np.concatenate(selected_valid),
            np.concatenate(selected_log_probs),
        )

    def _build_context(self, X: np.ndarray, y_target: np.ndarray) -> np.ndarray:
        mask = self.mask_vectors[self.params.mask_index]
        mask_matrix = np.repeat(mask[None, :], repeats=len(X), axis=0)
        p_value_column = np.full(
            (len(X), 1), fill_value=self.params.p_value, dtype=np.float32
        )
        class_one_hot = np.zeros((len(X), len(self.class_to_index)), dtype=np.float32)
        target_indices = np.vectorize(self.class_to_index.get, otypes=[int])(
            y_target.reshape(-1).astype(int)
        )
        class_one_hot[np.arange(len(X)), target_indices] = 1.0
        return np.concatenate(
            [X, class_one_hot, mask_matrix, p_value_column], axis=1
        ).astype(np.float32)

    def _select_best_candidates(
        self,
        factual_batch: np.ndarray,
        y_target_batch: np.ndarray,
        candidates: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        batch_size, num_samples, _ = candidates.shape
        flat_candidates = candidates.reshape(batch_size * num_samples, -1)
        probs = self.disc_model.predict_proba(flat_candidates).reshape(
            batch_size, num_samples, -1
        )
        num_classes = probs.shape[-1]
        target_indices = np.vectorize(self.class_to_index.get, otypes=[int])(
            y_target_batch.reshape(-1).astype(int)
        )
        if np.any(target_indices >= num_classes):
            raise ValueError("Target class index exceeds discriminator outputs.")
        expanded_targets = np.repeat(target_indices[:, None], num_samples, axis=1)
        target_probs = np.take_along_axis(
            probs, expanded_targets[..., None], axis=2
        ).squeeze(2)
        predicted_class = np.argmax(probs, axis=2)
        valid_mask_matrix = predicted_class == target_indices[:, None]

        best_indices = np.zeros(batch_size, dtype=int)
        valid_rows = np.any(valid_mask_matrix, axis=1)
        if np.any(valid_rows):
            filtered = np.where(
                valid_mask_matrix[valid_rows], target_probs[valid_rows], -np.inf
            )
            best_indices[valid_rows] = np.argmax(filtered, axis=1)
        if np.any(~valid_rows):
            best_indices[~valid_rows] = np.argmax(target_probs[~valid_rows], axis=1)

        row_selector = np.arange(batch_size)
        selected_cf = candidates[row_selector, best_indices]
        selected_probs = target_probs[row_selector, best_indices]
        selected_valid = valid_mask_matrix[row_selector, best_indices]
        return selected_cf, selected_probs, selected_valid, best_indices
