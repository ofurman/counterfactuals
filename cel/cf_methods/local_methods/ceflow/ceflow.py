"""CeFlow counterfactual method implementation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Literal

import numpy as np
import torch

from cel.cf_methods.counterfactual_base import (
    BaseCounterfactualMethod,
    ExplanationResult,
)
from cel.cf_methods.local_counterfactual_mixin import (
    LocalCounterfactualMixin,
)
from cel.models.pytorch_base import PytorchBase

logger = logging.getLogger(__name__)


@dataclass
class CeFlowParams:
    """Configuration for CeFlow mean-shift search."""

    batch_size: int = 4096
    alpha_min: float = 0.1
    alpha_max: float = 0.9
    alpha_steps: int = 9
    distance_metric: Literal["latent", "original"] = "original"
    binary_logits: bool | None = None
    clamp_min: float | None = None
    clamp_max: float | None = None
    use_predicted_labels: bool = True
    alpha_grid: list[float] = field(default_factory=list)


class CeFlow(BaseCounterfactualMethod, LocalCounterfactualMixin):
    """Normalizing-flow-based counterfactual generator using latent optimization."""

    def __init__(
        self,
        flow_model: torch.nn.Module,
        disc_model: PytorchBase,
        params: CeFlowParams | None = None,
        encode_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        decode_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        device: str | None = None,
        **kwargs,
    ) -> None:
        """Initialize CeFlow.

        Args:
            flow_model: Normalizing flow model used to map data to/from latent space.
            disc_model: Predictive model to explain.
            params: Optimization parameters for latent search.
            encode_fn: Optional encoder mapping inputs to latent space.
            decode_fn: Optional decoder mapping latent vectors to input space.
            device: Torch device string.
            **kwargs: Ignored extra parameters for interface compatibility.
        """
        super().__init__(gen_model=None, disc_model=disc_model, device=device)
        self.flow_model = flow_model
        self.params = params or CeFlowParams()
        self.device = device or "cpu"
        self._encode_fn = encode_fn
        self._decode_fn = decode_fn
        self._binary_logits = self.params.binary_logits
        self._class_means: Dict[int, torch.Tensor] = {}
        self._alpha_values = self._resolve_alpha_grid(self.params)

        if self._decode_fn is None and not hasattr(self.flow_model, "inverse"):
            raise ValueError(
                "CeFlow requires decode_fn or a flow model with .inverse()."
            )

        if hasattr(self.flow_model, "to"):
            self.flow_model.to(self.device)
        if hasattr(self.flow_model, "eval"):
            self.flow_model.eval()
        self.disc_model.to(self.device)
        self.disc_model.eval()

    def fit(
        self, X_train: np.ndarray, y_train: np.ndarray | None = None, **kwargs
    ) -> None:
        """Compute class means in the latent space.

        Args:
            X_train: Training features to encode.
            y_train: Optional labels. If not provided or if configured, predicted labels
                from the discriminator are used.
            **kwargs: Ignored extra parameters.
        """
        if hasattr(self.flow_model, "get_class_means"):
            logger.info("CeFlow using learned GMM class means from flow model.")
            self._class_means = {
                label: mean.to(self.device)
                for label, mean in self.flow_model.get_class_means().items()
            }
            return

        x_np = np.asarray(X_train, dtype=np.float32)
        if y_train is None or self.params.use_predicted_labels:
            logger.info("CeFlow computing class means using discriminator predictions.")
            y_labels = self.disc_model.predict(x_np)
        else:
            y_labels = np.asarray(y_train)
        y_labels = y_labels.reshape(-1)
        self._class_means = self._compute_class_means(x_np, y_labels)

    def explain(
        self,
        X: np.ndarray,
        y_origin: np.ndarray,
        y_target: np.ndarray,
        X_train: np.ndarray | None = None,
        y_train: np.ndarray | None = None,
        **kwargs,
    ) -> ExplanationResult:
        """Generate counterfactuals for the provided samples."""
        x_np = np.asarray(X, dtype=np.float32)
        y_origin_vec = np.asarray(y_origin).reshape(-1)
        y_target_vec = np.asarray(y_target).reshape(-1)
        if x_np.shape[0] != y_target_vec.shape[0]:
            raise ValueError("y_target must have the same number of rows as X.")

        if not self._class_means:
            if X_train is None:
                raise ValueError("CeFlow requires X_train to compute class means.")
            self.fit(X_train=X_train, y_train=y_train)

        cfs: list[np.ndarray] = []
        logs: list[dict[str, float]] = []
        batch_size = max(1, int(self.params.batch_size))
        for start in range(0, len(x_np), batch_size):
            end = start + batch_size
            batch_x = x_np[start:end]
            batch_targets = y_target_vec[start:end]
            batch_origins = y_origin_vec[start:end]
            try:
                batch_cfs, batch_logs = self._search_batch(
                    batch_x, batch_origins, batch_targets
                )
                cfs.extend(batch_cfs)
                logs.extend(batch_logs)
            except Exception as exc:
                logger.warning(
                    "CeFlow batch search failed for rows %s-%s: %s",
                    start,
                    end - 1,
                    exc,
                )
                for idx, target in enumerate(batch_targets):
                    cf, log = self._search_single(
                        batch_x[idx], batch_origins[idx], target
                    )
                    cfs.append(cf)
                    logs.append(log)

        return ExplanationResult(
            x_cfs=np.stack(cfs, axis=0),
            y_cf_targets=y_target_vec,
            x_origs=x_np,
            y_origs=y_origin_vec,
            logs={"per_instance": logs},
        )

    def explain_dataloader(
        self,
        dataloader,
        target_class: int | None = None,
        **kwargs,
    ) -> ExplanationResult:
        """Generate counterfactuals for a full dataloader."""
        xs, ys = [], []
        for batch_x, batch_y in dataloader:
            xs.append(batch_x.numpy())
            ys.append(batch_y.numpy())
        X = np.vstack(xs)
        y_origin = np.concatenate(ys)
        if target_class is None:
            raise ValueError("target_class must be provided for explain_dataloader.")
        y_target = np.full_like(y_origin, fill_value=target_class)
        return self.explain(X=X, y_origin=y_origin, y_target=y_target)

    def _search_single(
        self, x: np.ndarray, y_origin: float | int, y_target: float | int
    ) -> tuple[np.ndarray, dict]:
        x_tensor = torch.from_numpy(x).float().to(self.device).unsqueeze(0)
        with torch.no_grad():
            z_factual = self._encode(x_tensor)
            delta = self._get_delta_vector(y_origin, y_target)
            best = self._search_over_alphas(
                z_factual, x_tensor, y_target, delta, self._alpha_values
            )
        return best

    def _search_batch(
        self, x: np.ndarray, y_origin: np.ndarray, y_target: np.ndarray
    ) -> tuple[list[np.ndarray], list[dict[str, float]]]:
        x_tensor = torch.from_numpy(x).float().to(self.device)
        y_origin_tensor = torch.from_numpy(y_origin).to(self.device)
        y_target_tensor = torch.from_numpy(y_target).to(self.device)

        with torch.no_grad():
            z_factual = self._encode(x_tensor)
            delta = self._get_delta_matrix(y_origin_tensor, y_target_tensor)
            best_cf = None
            best_prob = torch.full(
                (x_tensor.shape[0],), -float("inf"), device=self.device
            )
            best_distance = torch.full(
                (x_tensor.shape[0],), float("inf"), device=self.device
            )
            best_alpha = torch.full(
                (x_tensor.shape[0],), float("nan"), device=self.device
            )

            for alpha in self._alpha_values:
                z_hat = z_factual + alpha * delta
                x_hat = self._decode(z_hat)
                x_hat = self._clamp(x_hat)
                preds = self.disc_model(x_hat)
                target_prob = self._target_probability_batch(preds, y_target_tensor)
                distance = self._distance_values(z_factual, z_hat, x_tensor, x_hat)
                reached = self._is_target_hit_batch(preds, y_target_tensor)
                improved = reached & (distance < best_distance)
                fallback = (~reached) & (target_prob > best_prob)
                update = improved | fallback
                if best_cf is None:
                    best_cf = x_hat.detach().clone()
                best_cf = torch.where(update[:, None], x_hat.detach(), best_cf)
                best_prob = torch.where(update, target_prob, best_prob)
                best_distance = torch.where(update, distance, best_distance)
                best_alpha = torch.where(
                    update, torch.full_like(best_alpha, alpha), best_alpha
                )

        cfs = best_cf.cpu().numpy()
        logs = []
        for idx in range(cfs.shape[0]):
            logs.append(
                {
                    "target_probability": float(best_prob[idx].item()),
                    "distance": float(best_distance[idx].item()),
                    "alpha": float(best_alpha[idx].item()),
                }
            )
        return [cfs[idx] for idx in range(cfs.shape[0])], logs

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        if self._encode_fn is not None:
            return self._encode_fn(x)
        return self.flow_model(x)

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        if self._decode_fn is not None:
            return self._decode_fn(z)
        return self.flow_model.inverse(z)

    def _search_over_alphas(
        self,
        z_factual: torch.Tensor,
        x_factual: torch.Tensor,
        y_target: float | int,
        delta: torch.Tensor,
        alphas: Iterable[float],
    ) -> tuple[np.ndarray, dict]:
        best_cf = None
        best_distance = float("inf")
        best_prob = -float("inf")
        best_alpha = float("nan")
        for alpha in alphas:
            z_hat = z_factual + alpha * delta
            x_hat = self._clamp(self._decode(z_hat))
            preds = self.disc_model(x_hat)
            target_prob = float(self._target_probability(preds, y_target).item())
            distance = float(self._distance_value(z_factual, z_hat, x_factual, x_hat))
            reached = self._is_target_hit(preds, y_target)
            if reached and distance < best_distance:
                best_distance = distance
                best_prob = target_prob
                best_alpha = float(alpha)
                best_cf = x_hat.detach().clone()
            elif best_cf is None and target_prob > best_prob:
                best_prob = target_prob
                best_alpha = float(alpha)
                best_distance = distance
                best_cf = x_hat.detach().clone()
        if best_cf is None:
            best_cf = x_hat.detach()
            best_distance = distance
            best_prob = target_prob
            best_alpha = float(alpha)
        return (
            best_cf.squeeze(0).cpu().numpy(),
            {
                "target_probability": best_prob,
                "distance": best_distance,
                "alpha": best_alpha,
            },
        )

    def _target_probability(
        self, preds: torch.Tensor, target: float | int
    ) -> torch.Tensor:
        if self.disc_model.num_targets == 1:
            use_logits = self._binary_logits
            if use_logits is None:
                use_logits = not torch.all((preds >= 0.0) & (preds <= 1.0))
                self._binary_logits = bool(use_logits)
            prob = torch.sigmoid(preds) if use_logits else preds
            prob = prob.view(-1)
            return prob if int(target) == 1 else 1 - prob
        probs = torch.softmax(preds, dim=1)
        target_index = int(target)
        return probs[:, target_index]

    def _target_probability_batch(
        self, preds: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        if self.disc_model.num_targets == 1:
            use_logits = self._binary_logits
            if use_logits is None:
                use_logits = not torch.all((preds >= 0.0) & (preds <= 1.0))
                self._binary_logits = bool(use_logits)
            prob = torch.sigmoid(preds) if use_logits else preds
            prob = prob.view(-1)
            targets = targets.view(-1)
            return torch.where(targets == 1, prob, 1 - prob)
        probs = torch.softmax(preds, dim=1)
        target_indices = targets.long().view(-1)
        return probs[torch.arange(probs.shape[0]), target_indices]

    def _is_target_hit(self, preds: torch.Tensor, target: float | int) -> bool:
        if self.disc_model.num_targets == 1:
            prob = self._target_probability(preds, target)
            return bool((prob >= 0.5).item())
        return bool(torch.argmax(preds, dim=1).item() == int(target))

    def _is_target_hit_batch(
        self, preds: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        if self.disc_model.num_targets == 1:
            prob = self._target_probability_batch(preds, targets)
            return prob >= 0.5
        predicted = torch.argmax(preds, dim=1)
        return predicted == targets.long().view(-1)

    def _clamp(self, x_hat: torch.Tensor) -> torch.Tensor:
        if self.params.clamp_min is None and self.params.clamp_max is None:
            return x_hat
        return torch.clamp(
            x_hat,
            min=self.params.clamp_min
            if self.params.clamp_min is not None
            else -float("inf"),
            max=self.params.clamp_max
            if self.params.clamp_max is not None
            else float("inf"),
        )

    def _distance_value(
        self,
        z_factual: torch.Tensor,
        z_hat: torch.Tensor,
        x_factual: torch.Tensor,
        x_hat: torch.Tensor,
    ) -> torch.Tensor:
        if self.params.distance_metric == "latent":
            return torch.mean((z_hat - z_factual).pow(2))
        return torch.mean((x_hat - x_factual).pow(2))

    def _distance_values(
        self,
        z_factual: torch.Tensor,
        z_hat: torch.Tensor,
        x_factual: torch.Tensor,
        x_hat: torch.Tensor,
    ) -> torch.Tensor:
        if self.params.distance_metric == "latent":
            diff = (z_hat - z_factual).pow(2)
        else:
            diff = (x_hat - x_factual).pow(2)
        return diff.view(diff.shape[0], -1).mean(dim=1)

    def _resolve_alpha_grid(self, params: CeFlowParams) -> list[float]:
        if params.alpha_grid:
            return [float(alpha) for alpha in params.alpha_grid]
        if params.alpha_steps <= 1:
            return [float(params.alpha_max)]
        return np.linspace(
            params.alpha_min, params.alpha_max, params.alpha_steps
        ).tolist()

    def _compute_class_means(
        self, X: np.ndarray, y_labels: np.ndarray
    ) -> Dict[int, torch.Tensor]:
        class_means: Dict[int, torch.Tensor] = {}
        unique_labels = np.unique(y_labels).astype(int)
        batch_size = max(1, int(self.params.batch_size))
        latent_by_class: Dict[int, list[torch.Tensor]] = {
            int(k): [] for k in unique_labels
        }
        for start in range(0, len(X), batch_size):
            end = start + batch_size
            x_tensor = torch.from_numpy(X[start:end]).float().to(self.device)
            with torch.no_grad():
                z_batch = self._encode(x_tensor)
            for label in unique_labels:
                mask = y_labels[start:end] == label
                if np.any(mask):
                    latent_by_class[int(label)].append(z_batch[mask])
        for label, chunks in latent_by_class.items():
            if not chunks:
                continue
            class_means[label] = torch.mean(torch.cat(chunks, dim=0), dim=0)
        if not class_means:
            raise ValueError("CeFlow could not compute class means from training data.")
        return class_means

    def _get_delta_vector(
        self, y_origin: float | int, y_target: float | int
    ) -> torch.Tensor:
        origin = int(y_origin)
        target = int(y_target)
        if origin not in self._class_means or target not in self._class_means:
            raise ValueError("CeFlow missing class means for requested labels.")
        return self._class_means[target] - self._class_means[origin]

    def _get_delta_matrix(
        self, y_origin: torch.Tensor, y_target: torch.Tensor
    ) -> torch.Tensor:
        origin_labels = y_origin.long().view(-1).tolist()
        target_labels = y_target.long().view(-1).tolist()
        deltas = []
        for origin, target in zip(origin_labels, target_labels):
            deltas.append(self._get_delta_vector(origin, target))
        return torch.stack(deltas, dim=0)
