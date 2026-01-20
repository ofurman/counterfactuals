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
    cf_samples_per_factual: int = 1


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
        (
            cf_batch,
            y_target_flat,
            x_orig_flat,
            y_origin_flat,
            target_probs,
            valid_mask,
            log_probs,
            group_ids,
        ) = self._sample_counterfactuals(x_np, y_origin_vec, y_target_vec)
        logs = {
            "sampling/mean_target_probability": float(target_probs.mean()),
            "sampling/valid_ratio": float(valid_mask.mean()),
            "sampling/log_prob_mean": float(log_probs.mean()),
            "model_returned_mask": valid_mask.tolist(),
            "cf_group_ids": group_ids.tolist(),
        }
        return ExplanationResult(
            x_cfs=cf_batch,
            y_cf_targets=y_target_flat,
            x_origs=x_orig_flat,
            y_origs=y_origin_flat,
            logs=logs,
            cf_group_ids=group_ids,
        )

    def explain_dataloader(
        self,
        dataloader,
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
        self, X: np.ndarray, y_origin: np.ndarray, y_target: np.ndarray
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Sample candidate counterfactuals and keep top-k per factual."""
        batch_size = self.params.sampling_batch_size
        k = max(1, self.params.cf_samples_per_factual)

        flat_cf: list[np.ndarray] = []
        flat_y_target: list[np.ndarray] = []
        flat_x_orig: list[np.ndarray] = []
        flat_y_origin: list[np.ndarray] = []
        flat_probs: list[np.ndarray] = []
        flat_valid: list[np.ndarray] = []
        flat_log_probs: list[np.ndarray] = []
        group_ids: list[int] = []
        global_idx = 0

        for start in range(0, len(X), batch_size):
            end = start + batch_size
            X_batch = X[start:end]
            y_target_batch = y_target[start:end]
            y_origin_batch = y_origin[start:end]
            if X_batch.size == 0:
                continue
            context = self._build_context(X_batch, y_target_batch)
            samples, log_probs = self.gen_model.sample_and_log_proba(
                n_samples=self.params.num_counterfactuals, context=context
            )

            (
                batch_selected_cf,
                batch_selected_probs,
                batch_selected_valid,
                batch_selected_logs,
            ) = self._select_topk_candidates(
                y_target_batch=y_target_batch,
                candidates=samples,
                candidate_log_probs=log_probs,
                top_k=k,
            )

            batch_size_current = X_batch.shape[0]
            flat_cf.append(batch_selected_cf.reshape(-1, samples.shape[-1]))
            flat_probs.append(batch_selected_probs.reshape(-1))
            flat_valid.append(batch_selected_valid.reshape(-1))
            flat_log_probs.append(batch_selected_logs.reshape(-1))
            flat_x_orig.append(np.repeat(X_batch, k, axis=0))
            flat_y_target.append(np.repeat(y_target_batch, k, axis=0))
            flat_y_origin.append(np.repeat(y_origin_batch, k, axis=0))
            group_ids.extend(
                np.repeat(np.arange(global_idx, global_idx + batch_size_current), k)
            )
            global_idx += batch_size_current

        return (
            np.vstack(flat_cf),
            np.vstack(flat_y_target),
            np.vstack(flat_x_orig),
            np.vstack(flat_y_origin),
            np.concatenate(flat_probs),
            np.concatenate(flat_valid),
            np.concatenate(flat_log_probs),
            np.array(group_ids, dtype=int),
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

    def _select_topk_candidates(
        self,
        y_target_batch: np.ndarray,
        candidates: np.ndarray,
        candidate_log_probs: np.ndarray,
        top_k: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

        sorted_indices = np.argsort(-target_probs, axis=1)
        top_indices = np.zeros((batch_size, top_k), dtype=int)
        for i in range(batch_size):
            chosen: list[int] = []
            for idx in sorted_indices[i]:
                if valid_mask_matrix[i, idx]:
                    chosen.append(idx)
                if len(chosen) == top_k:
                    break
            if len(chosen) < top_k:
                for idx in sorted_indices[i]:
                    if idx not in chosen:
                        chosen.append(idx)
                    if len(chosen) == top_k:
                        break
            if not chosen:
                fallback = sorted_indices[i][0] if sorted_indices[i].size else 0
                chosen = [int(fallback)]
            if len(chosen) < top_k:
                chosen.extend([chosen[-1]] * (top_k - len(chosen)))
            top_indices[i] = np.array(chosen[:top_k])

        row_selector = np.arange(batch_size)[:, None]
        selected_cf = candidates[row_selector, top_indices]
        selected_probs = target_probs[row_selector, top_indices]
        selected_valid = valid_mask_matrix[row_selector, top_indices]
        selected_logs = candidate_log_probs[row_selector, top_indices]
        return selected_cf, selected_probs, selected_valid, selected_logs
