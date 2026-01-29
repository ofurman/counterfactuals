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
        """
        Initializes the Dequantizer with a specified noise type.

        Args:
            noise_type (str): The type of noise to add for dequantization.
                              Must be a key in NOISE_REGISTRY.
        """
        self.noise_type = NOISE_REGISTRY[noise_type]

    def _dequantize(self, x, rng):
        """
        Adds noise to pixels to dequantize them.
        Ensures the output stays in the valid range [0, 1].

        Args:
            x (np.ndarray): Input array of shape (n_samples, n_features) representing quantized data.
            rng (np.random.RandomState): Random number generator for noise generation.

        Returns:
            np.ndarray: Dequantized data with noise added and scaled by dividers.
        """
        for i in range(x.shape[1]):
            data_with_noise = x[:, i] + self.noise_type(rng, x[:, i].shape)
            divider = self.dividers[i]
            x[:, i] = data_with_noise / divider
        return x

    def _logit_transform(self, x):
        """
        Transforms pixel values with logit to be unconstrained.

        Applies a smoothing term ALPHA to avoid numerical issues with logit at 0 and 1.

        Args:
            x (np.ndarray): Input array with values in [0, 1].

        Returns:
            np.ndarray: Logit-transformed array with unconstrained real values.
        """
        x = np.clip(x, 0, 1)
        x = ALPHA + (1 - 2 * ALPHA) * x
        return np.log(x / (1.0 - x))

    @staticmethod
    def inverse(x: Union[np.ndarray, torch.Tensor], dividers: list) -> np.ndarray:
        """
        Inverse transform the logit transformation.
        Handles both numpy arrays and torch tensors.

        Args:
            x (Union[np.ndarray, torch.Tensor]): Logit-transformed data to invert.
            dividers (list): List of integer dividers used for scaling each feature.

        Returns:
            np.ndarray: Discretized data in original quantized scale.

        Raises:
            ValueError: If input x is not a numpy array or torch tensor.
        """
        x = x.copy()
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        elif not isinstance(x, torch.Tensor):
            raise ValueError("Input must be a numpy array or torch tensor.")
        x = (torch.sigmoid(x) - 1e-6) / (1 - 2e-6)
        for i in range(x.shape[1]):
            bins = np.linspace(0, 1, dividers[i])
            x[:, i] = np.digitize(x[:, i], bins) - 1
        return x.numpy()

    def fit(self, X, y=None):
        """
        Fits the Dequantizer by determining the dividers for each feature.

        Args:
            X (np.ndarray): Input data array of shape (n_samples, n_features).
            y: Ignored.

        Returns:
            Dequantizer: Returns self.
        """
        self.dividers = [max(int(X[:, i].max()) + 1, 2) for i in range(X.shape[1])]
        return self

    def transform(self, X):
        """
        Transforms the input data by dequantizing and applying logit transform.

        Args:
            X (np.ndarray): Input quantized data of shape (n_samples, n_features).

        Returns:
            np.ndarray: Transformed data with noise added and logit applied.
        """
        X_transformed = X.copy()
        X_transformed = self._dequantize(X_transformed, np.random.RandomState(42))
        X_transformed = self._logit_transform(X_transformed)
        return X_transformed

    def inverse_transform(self, X):
        """
        Inverse transform the logit transformation.
        Handles both numpy arrays and torch tensors.

        Args:
            X (Union[np.ndarray, torch.Tensor]): Transformed data to invert.

        Returns:
            np.ndarray: Data transformed back to original quantized scale.
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
        """
        Computes the sigmoid function.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Sigmoid of input array.
        """
        return 1 / (1 + np.exp(-x))
