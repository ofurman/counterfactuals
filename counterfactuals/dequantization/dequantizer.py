from typing import List, Union

import numpy as np
import torch
from sklearn.base import BaseEstimator, TransformerMixin

from counterfactuals.dequantization.noise import NOISE_REGISTRY
from counterfactuals.processing.transform import GroupTransformer

ALPHA = 1e-6


class GroupDequantizer(GroupTransformer):
    """Group dequantizer for categorical features."""

    def __init__(
        self, groups: List[List[int]], transformer_factory=lambda: Dequantizer()
    ):
        super().__init__(groups, transformer_factory)


class Dequantizer(BaseEstimator, TransformerMixin):
    def __init__(self, noise_type: str = "gaussian"):
        self.noise_type = NOISE_REGISTRY[noise_type]

    def _dequantize(self, x, rng):
        """
        Adds noise to pixels to dequantize them.
        Ensures the output stays in the valid range [0, 1].
        """
        for i in range(x.shape[1]):
            data_with_noise = x[:, i] + self.noise_type(rng, x[:, i].shape)
            divider = self.dividers[i]
            x[:, i] = data_with_noise / divider
        return x

    def _logit_transform(self, x):
        """
        Transforms pixel values with logit to be unconstrained.
        """
        x = ALPHA + (1 - 2 * ALPHA) * x
        return np.log(x / (1.0 - x))

    @staticmethod
    def inverse(x: Union[np.ndarray, torch.Tensor], dividers: list) -> np.ndarray:
        """
        Inverse transform the logit transformation.
        Handles both numpy arrays and torch tensors.
        """
        x = x.copy()
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = (torch.sigmoid(x) - 1e-6) / (1 - 2e-6)
        for i in range(x.shape[1]):
            bins = np.linspace(0, 1, dividers[i])
            x[:, i] = np.digitize(x[:, i], bins) - 1
        return x.numpy()

    def fit(self, X, y=None):
        self.dividers = [int(X[:, i].max()) + 1 for i in range(X.shape[1])]
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed = self._dequantize(X_transformed, np.random.RandomState(42))
        X_transformed = self._logit_transform(X_transformed)
        return X_transformed

    def inverse_transform(self, X):
        """
        Inverse transform the logit transformation.
        Handles both numpy arrays and torch tensors.
        """
        x = X.copy()
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        x = (torch.sigmoid(torch.from_numpy(x)).numpy() - 1e-6) / (1 - 2e-6)
        for i in range(x.shape[1]):
            bins = np.linspace(0, 1, self.dividers[i] + 1)
            x[:, i] = np.digitize(x[:, i], bins) - 1
        return x

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
