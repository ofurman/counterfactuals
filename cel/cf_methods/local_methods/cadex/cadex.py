from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from cel.cf_methods.counterfactual_base import (
    BaseCounterfactualMethod,
    ExplanationResult,
)
from cel.cf_methods.local_counterfactual_mixin import (
    LocalCounterfactualMixin,
)

ScaleFn = Callable[[np.ndarray], np.ndarray]


class InputAddLayer(torch.nn.Module):
    """Layer that adds trainable offsets to inputs."""

    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.delta = torch.nn.Parameter(torch.zeros(1, n_features))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.delta

    def transform(self, inputs: np.ndarray) -> np.ndarray:
        return inputs + self.delta.detach().cpu().numpy()


class CadexEngine:
    """CADEX engine that optimizes an additive input modifier."""

    def __init__(
        self,
        model: torch.nn.Module,
        categorical_attributes: Optional[list[list[int]]] = None,
        ordinal_attributes: Optional[list[int]] = None,
        scale: Optional[ScaleFn] = None,
        unscale: Optional[ScaleFn] = None,
        device: Optional[str] = None,
    ) -> None:
        if ordinal_attributes is not None and (scale is None or unscale is None):
            raise ValueError(
                "scale and unscale must be provided for ordinal attributes."
            )

        self.original_model = model
        self.original_model.eval()

        self._categorical_attributes = categorical_attributes
        self._ordinal_attributes = ordinal_attributes
        self._scale = scale
        self._unscale = unscale
        self.device = device or "cpu"

        self._input_layer: Optional[InputAddLayer] = None
        self._mask: Optional[torch.Tensor] = None
        self._optimizer: Optional[torch.optim.Optimizer] = None

    def reset(self, n_features: int) -> None:
        self._input_layer = InputAddLayer(n_features).to(self.device)

    def train(
        self,
        inputs: np.ndarray,
        target: int,
        num_classes: int,
        num_changed_attributes: Optional[int] = None,
        max_epochs: int = 1000,
        skip_attributes: int = 0,
        categorical_threshold: float = 0.2,
        direction_constraints: Optional[np.ndarray] = None,
    ) -> tuple[Optional[np.ndarray], int]:
        """Train the input modifier to produce a CADEX explanation."""
        self.reset(inputs.shape[1])
        self._begin_train(
            inputs,
            target,
            num_changed_attributes=num_changed_attributes,
            skip_attributes=skip_attributes,
            direction_constraints=direction_constraints,
        )
        return self._train_step(
            inputs,
            target,
            num_classes,
            max_epochs=max_epochs,
            categorical_threshold=categorical_threshold,
        )

    def _begin_train(
        self,
        inputs: np.ndarray,
        target: int,
        num_changed_attributes: Optional[int] = None,
        skip_attributes: int = 0,
        direction_constraints: Optional[np.ndarray] = None,
    ) -> None:
        mask = np.ones(inputs.shape, dtype=np.float32)
        if num_changed_attributes is not None or direction_constraints is not None:
            first_grad = self.get_gradient(inputs, target)

            if self._categorical_attributes is not None:
                for attr_set in self._categorical_attributes:
                    for attr in attr_set:
                        if first_grad[0, attr] > 0 or inputs[0, attr] > 0:
                            first_grad[0, attr] = 0

            if skip_attributes > 0:
                ind = np.argsort(np.abs(first_grad))
                first_grad[0, ind[0, -skip_attributes:]] = 0

            if direction_constraints is not None:
                mask = mask * np.sign(
                    np.sign(first_grad * (-direction_constraints)) + 1
                )

            if num_changed_attributes is not None:
                ind = np.argsort(np.abs(first_grad * mask))
                change_mask = np.zeros(inputs.shape, dtype=np.float32)
                change_mask[0, ind[0, -num_changed_attributes:]] = 1
                mask = mask * change_mask

        self._mask = torch.tensor(mask, dtype=torch.float32, device=self.device)
        self._optimizer = torch.optim.Adam([self._input_layer.delta])

    def _train_step(
        self,
        inputs: np.ndarray,
        target: int,
        num_classes: int,
        max_epochs: int = 1000,
        categorical_threshold: float = 0.0,
    ) -> tuple[Optional[np.ndarray], int]:
        result = None
        pred_threshold = 1.0 / num_classes

        categorical_attrs = self._categorical_attributes
        ordinal_attrs = self._ordinal_attributes

        for epoch in range(max_epochs):
            self._train_on_instance(inputs, target, num_classes)

            if categorical_attrs is not None:
                self._adjust_categorical(
                    inputs, categorical_attrs, threshold=categorical_threshold
                )

            preds = self._predict_proba(inputs)
            if preds[0, target] > pred_threshold:
                if categorical_attrs is not None or ordinal_attrs is not None:
                    constrained_input, constrained_weights = self._apply_constraints(
                        inputs, categorical_attrs, ordinal_attrs
                    )
                    constrained_preds = self._predict_proba(
                        constrained_input, use_input_layer=False
                    )
                    if constrained_preds[0, target] > pred_threshold:
                        self._input_layer.delta.data = torch.tensor(
                            constrained_weights, device=self.device
                        )
                        result = constrained_input
                        break
                else:
                    result = self._input_layer.transform(inputs)
                    break

        return result, epoch

    def _train_on_instance(
        self, inputs: np.ndarray, target: int, num_classes: int
    ) -> None:
        if self._optimizer is None or self._mask is None or self._input_layer is None:
            raise ValueError("CadexEngine must be initialized before training.")

        inputs_tensor = torch.tensor(inputs, dtype=torch.float32, device=self.device)

        self._optimizer.zero_grad()
        logits = self.original_model(self._input_layer(inputs_tensor))
        loss = self._classification_loss(logits, target, num_classes)
        loss.backward()

        self._input_layer.delta.grad *= self._mask
        self._optimizer.step()

    def _adjust_categorical(
        self,
        inputs: np.ndarray,
        categorical_attributes: list[list[int]],
        threshold: float = 0.0,
    ) -> np.ndarray:
        input_mod = self._input_layer.transform(inputs)
        input_target = input_mod.copy()

        for attr_set in categorical_attributes:
            max_vals = np.argsort(input_mod[0, attr_set])[::-1]
            if input_mod[0, attr_set[max_vals[1]]] > threshold:
                input_target[0, attr_set] = 0
                input_target[0, attr_set[max_vals[1]]] = 1

        if np.any(input_target != input_mod):
            updated_weights = input_target - inputs
            self._input_layer.delta.data = torch.tensor(
                updated_weights, dtype=torch.float32, device=self.device
            )
        return input_target

    def _apply_constraints(
        self,
        inputs: np.ndarray,
        categorical_attributes: Optional[list[list[int]]],
        ordinal_attributes: Optional[list[int]],
    ) -> tuple[np.ndarray, np.ndarray]:
        input_mod = self._input_layer.transform(inputs)
        input_target = input_mod.copy()

        if categorical_attributes is not None:
            for attr_set in categorical_attributes:
                max_vals = np.argsort(input_mod[0, attr_set])[::-1]
                max_cat = attr_set[max_vals[0]]
                input_target[0, attr_set] = 0
                input_target[0, max_cat] = 1

        if ordinal_attributes is not None and self._scale and self._unscale:
            unscaled = self._unscale(input_target)
            for attr in ordinal_attributes:
                unscaled[0, attr] = np.round(unscaled[0, attr])
            input_target = self._scale(unscaled)

        updated_weights = input_target - inputs
        return input_target, updated_weights

    def get_gradient(self, inputs: np.ndarray, target: int) -> np.ndarray:
        """Calculate gradient in input space."""
        inputs_tensor = torch.tensor(inputs, dtype=torch.float32, device=self.device)
        inputs_tensor.requires_grad = True
        logits = self.original_model(inputs_tensor)
        num_classes = logits.shape[1] if logits.ndim > 1 else 2
        loss = self._classification_loss(logits, target, num_classes)
        grads = torch.autograd.grad(loss, inputs_tensor)[0]
        return grads.detach().cpu().numpy()

    def _predict_proba(
        self, inputs: np.ndarray, use_input_layer: bool = True
    ) -> np.ndarray:
        inputs_tensor = torch.tensor(inputs, dtype=torch.float32, device=self.device)
        if use_input_layer:
            inputs_tensor = self._input_layer(inputs_tensor)
        with torch.no_grad():
            logits = self.original_model(inputs_tensor)
            if logits.ndim == 1 or logits.shape[1] == 1:
                probs_pos = torch.sigmoid(logits).view(-1, 1)
                probs = torch.cat([1 - probs_pos, probs_pos], dim=1)
            else:
                probs = torch.softmax(logits, dim=1)
        return probs.detach().cpu().numpy()

    @staticmethod
    def _classification_loss(
        logits: torch.Tensor, target: int, num_classes: int
    ) -> torch.Tensor:
        if logits.ndim == 1 or logits.shape[1] == 1:
            target_value = torch.tensor([float(target == 1)], device=logits.device)
            logits = logits.view(-1)
            return torch.nn.functional.binary_cross_entropy_with_logits(
                logits, target_value
            )
        target_tensor = torch.tensor([target], dtype=torch.long, device=logits.device)
        return torch.nn.functional.cross_entropy(logits, target_tensor)


class CADEX(BaseCounterfactualMethod, LocalCounterfactualMixin):
    """Constrained adversarial counterfactual explanations (CADEX)."""

    def __init__(
        self,
        disc_model: torch.nn.Module,
        categorical_attributes: Optional[list[list[int]]] = None,
        ordinal_attributes: Optional[list[int]] = None,
        scale: Optional[ScaleFn] = None,
        unscale: Optional[ScaleFn] = None,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(disc_model=disc_model, device=device)
        self._engine = CadexEngine(
            model=disc_model,
            categorical_attributes=categorical_attributes,
            ordinal_attributes=ordinal_attributes,
            scale=scale,
            unscale=unscale,
            device=self.device,
        )

    def explain(
        self,
        X: np.ndarray,
        y_origin: np.ndarray,
        y_target: np.ndarray,
        X_train: Optional[np.ndarray] = None,
        y_train: Optional[np.ndarray] = None,
        **kwargs,
    ) -> ExplanationResult:
        """Generate counterfactual explanations for provided instances."""
        _ = X_train, y_train
        params = {
            "num_changed_attributes": kwargs.get("num_changed_attributes"),
            "max_epochs": kwargs.get("max_epochs", 1000),
            "skip_attributes": kwargs.get("skip_attributes", 0),
            "categorical_threshold": kwargs.get("categorical_threshold", 0.0),
            "direction_constraints": kwargs.get("direction_constraints"),
        }

        num_classes = self._infer_num_classes(X)
        x_cfs = []
        epochs = []

        for i in range(X.shape[0]):
            input_row = X[i : i + 1]
            target_label = int(y_target[i])
            cf, epoch = self._engine.train(
                input_row,
                target_label,
                num_classes,
                **params,
            )
            if cf is None:
                cf = input_row.copy()
            x_cfs.append(cf)
            epochs.append(epoch)

        return ExplanationResult(
            x_cfs=np.vstack(x_cfs),
            y_cf_targets=y_target,
            x_origs=X,
            y_origs=y_origin,
            logs={"epochs": np.array(epochs)},
        )

    def explain_dataloader(
        self,
        dataloader: DataLoader,
        epochs: int,
        lr: float,
        patience_eps: float = 1e-5,
        **search_step_kwargs,
    ) -> ExplanationResult:
        """CADEX does not support dataloader-based explanation in this wrapper."""
        _ = dataloader, epochs, lr, patience_eps, search_step_kwargs
        raise NotImplementedError("CADEX does not implement explain_dataloader.")

    def _infer_num_classes(self, X: np.ndarray) -> int:
        if hasattr(self.disc_model, "num_targets"):
            return int(self.disc_model.num_targets)
        inputs_tensor = torch.tensor(X[:1], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.disc_model(inputs_tensor)
        if logits.ndim == 1 or logits.shape[1] == 1:
            return 2
        return logits.shape[1]
