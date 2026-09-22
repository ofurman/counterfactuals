"""Present a classifier trained in one model space to code running in another.

The DICTUM-aligned setup trains one classifier per (dataset, seed) in the
`standard` model space and reloads it for every method, but DiCoFlex generates
in `minmax_qt`. Both spaces are invertible per-feature maps of the same raw
data, so inputs can be converted exactly: caller space -> original units ->
classifier space. Wrapping the classifier in this adapter keeps "the same model
under explanation" true across methods without retraining anything.
"""

from __future__ import annotations

import numpy as np
import torch

from counterfactuals.datasets.method_dataset import MethodDataset


class SpaceAdapterClassifier:
    """Wrap a classifier so callers may feed it rows from a different model space.

    Every prediction input is converted from `caller_dataset`'s space through
    original units into `model_dataset`'s space before the wrapped classifier
    sees it. All other attribute access is delegated to the wrapped model.

    Args:
        base_model: Classifier trained in `model_dataset`'s space.
        caller_dataset: Dataset fitted with the space the calling code works in.
        model_dataset: Dataset fitted with the space the classifier was trained in.
    """

    def __init__(
        self,
        base_model,
        caller_dataset: MethodDataset,
        model_dataset: MethodDataset,
    ):
        self.base_model = base_model
        self.caller_dataset = caller_dataset
        self.model_dataset = model_dataset

    def _convert(self, X) -> np.ndarray:
        arr = X.detach().cpu().numpy() if torch.is_tensor(X) else np.asarray(X)
        arr = self.model_dataset.transform(self.caller_dataset.inverse_transform(arr.copy()))
        return np.asarray(arr, dtype=np.float32)

    def predict(self, X) -> np.ndarray:
        """Predict class labels for rows given in the caller's space."""
        return self.base_model.predict(self._convert(X))

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities for rows given in the caller's space."""
        return self.base_model.predict_proba(self._convert(X))

    def forward(self, X) -> torch.Tensor:
        """Forward pass on rows given in the caller's space."""
        converted = torch.from_numpy(self._convert(X))
        device = next(self.base_model.parameters()).device
        return self.base_model(converted.to(device))

    __call__ = forward

    def to(self, device) -> SpaceAdapterClassifier:
        """Move the wrapped model to `device`; the adapter itself is stateless."""
        self.base_model.to(device)
        return self

    def __getattr__(self, name):
        # Only called for attributes not found on the adapter itself; delegate
        # everything else (eval, parameters, load, ...) to the wrapped model.
        return getattr(self.base_model, name)
