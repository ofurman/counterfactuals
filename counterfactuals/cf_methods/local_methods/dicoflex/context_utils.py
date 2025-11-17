"""Context helpers for the DiCoFlex conditional flow."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch


def _ensure_float32_contiguous(array: np.ndarray) -> np.ndarray:
    """Return a float32 contiguous view of the provided array."""
    return np.ascontiguousarray(array.astype(np.float32, copy=False))


def _label_to_index(label: float | int, class_to_index: Dict[int, int]) -> int:
    """Map a raw label value to its contiguous class index."""
    if label in class_to_index:
        return class_to_index[label]
    as_int = int(label)
    if as_int in class_to_index:
        return class_to_index[as_int]
    raise KeyError(f"Label {label!r} not found in class mapping {class_to_index}")


def build_context_matrix(
    factual_points: np.ndarray,
    labels: np.ndarray,
    mask_vector: np.ndarray,
    p_value: float,
    class_to_index: Dict[int, int],
) -> torch.Tensor:
    """Construct the DiCoFlex conditioning matrix for a batch of samples."""
    factual = _ensure_float32_contiguous(factual_points)
    n_samples, n_features = factual.shape
    mask = _ensure_float32_contiguous(mask_vector).reshape(1, -1)
    if mask.shape[1] != n_features:
        raise ValueError(
            "Mask dimensionality does not match number of features "
            f"({mask.shape[1]} != {n_features})."
        )

    mask_block = np.repeat(mask, repeats=n_samples, axis=0)
    label_indices = np.array(
        [_label_to_index(label, class_to_index) for label in labels.reshape(-1)],
        dtype=np.int64,
    )
    num_classes = len(class_to_index)
    class_one_hot = np.zeros((n_samples, num_classes), dtype=np.float32)
    class_one_hot[np.arange(n_samples), label_indices] = 1.0
    p_column = np.full((n_samples, 1), float(p_value), dtype=np.float32)

    context_np = np.concatenate(
        [factual, class_one_hot, mask_block, p_column],
        axis=1,
    )
    return torch.from_numpy(context_np)


def get_numpy_pointer(array: np.ndarray) -> int:
    """Return the base memory pointer for a numpy array."""
    return int(array.__array_interface__["data"][0])


class DiCoFlexGeneratorMetricsAdapter(torch.nn.Module):
    """
    Adapter that supplies the correct conditioning vectors when the generative
    model is queried by the metrics framework.
    """

    def __init__(
        self,
        base_model: torch.nn.Module,
        context_lookup: Dict[int, torch.Tensor],
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.context_lookup = context_lookup

    def eval(self) -> "DiCoFlexGeneratorMetricsAdapter":
        """Set the underlying model to eval mode and return self."""
        self.base_model.eval()
        return self

    def forward(self, X: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Proxy forward pass that injects the stored context."""
        context = self._resolve_context(X)
        if context.device != X.device:
            context = context.to(X.device)
        return self.base_model(X, context=context)

    def _resolve_context(self, X: torch.Tensor) -> torch.Tensor:
        pointer = int(X.data_ptr())
        context = self.context_lookup.get(pointer)
        if context is None:
            raise ValueError(
                "Unable to resolve DiCoFlex context for tensor with pointer "
                f"{pointer}. Ensure the numpy array passed to metrics is the "
                "same object used during registration."
            )
        return context

    def __getattr__(self, name: str):
        """Delegate attribute access to the wrapped model when needed."""
        if name in {"base_model", "context_lookup"}:
            return super().__getattr__(name)
        return getattr(self.base_model, name)
