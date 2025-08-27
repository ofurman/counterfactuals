from typing import Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer

ALPHA = 1e-6


class DequantizingFlow(nn.Module):
    def __init__(self, gen_model, dequantizer, dataset):
        super().__init__()
        self.gen_model = gen_model
        self.dequantizer = dequantizer
        self.dequantize = dequantize
        self.dataset = dataset

    def forward(self, X, y):
        if isinstance(X, torch.Tensor):
            X = X.numpy()
        _, X = self.dequantize(self.dataset, X, self.dequantizer)
        X = torch.from_numpy(X)
        log_probs = self.gen_model(X, y)
        return log_probs


def dequantize(dataset, data=None, transformer=None):
    """
    Apply dequantization, only affecting categorical features

    Parameters:
    -----------
    dataset : Dataset object
        Dataset containing categorical_features_lists
    data : np.ndarray, optional
        Optional external data to transform instead of dataset.X_train/X_test
    transformer : ColumnTransformer, optional
        Pre-fitted transformer to use for transformation. If None, create and fit a new one.

    Returns:
    --------
    tuple or np.ndarray
        If data is None: returns (transformer, None)
        If data is provided: returns (transformer, transformed_data)
    """
    # If no transformer is provided, create a new one
    if transformer is None:
        transformers = [
            (f"cat_group_{i}", CustomCategoricalTransformer(), group)
            for i, group in enumerate(dataset.categorical_features_lists)
        ]

        transformer = ColumnTransformer(
            transformers=transformers,
            remainder="drop",  # Drop continuous features
        )
        transformer.dropped_numerical = dataset.numerical_columns

        if data is None:
            X_train_original = dataset.X_train.copy()
            X_test_original = dataset.X_test.copy()

            cat_transformed_train = transformer.fit_transform(dataset.X_train)
            cat_transformed_test = transformer.transform(dataset.X_test)

            dataset.X_train = X_train_original.copy()
            dataset.X_test = X_test_original.copy()

            cat_idx = 0
            for group in dataset.categorical_features_lists:
                for i, feature_idx in enumerate(group):
                    dataset.X_train[:, feature_idx] = cat_transformed_train[:, cat_idx]
                    dataset.X_test[:, feature_idx] = cat_transformed_test[:, cat_idx]
                    cat_idx += 1

            return transformer, None
        else:
            transformer.fit(dataset.X_train)

    if data is not None:
        data_copy = data.copy()

        for i, group in enumerate(dataset.categorical_features_lists):
            transformer_name = f"cat_group_{i}"

            group_data = data_copy[:, group]

            transformed_data = transformer.named_transformers_[
                transformer_name
            ].transform(group_data)

            for j, feature_idx in enumerate(group):
                data_copy[:, feature_idx] = transformed_data[:, j]

        return transformer, data_copy

    return transformer, None


def inverse_dequantize(dataset, dequantizer, data=None):
    """
    Inverse the dequantization process, only affecting categorical features

    Parameters:
    -----------
    dataset : Dataset object
        Dataset containing categorical_features_lists
    dequantizer : ColumnTransformer
        The fitted dequantizer returned by dequantize()
    data : np.ndarray, optional
        Optional external data to transform instead of dataset.X_train/X_test

    Returns:
    --------
    np.ndarray or None
        Returns transformed data if data parameter provided, otherwise None
    """
    if data is not None:
        data_copy = data.copy()

        for i, group in enumerate(dataset.categorical_features_lists):
            transformer_name = f"cat_group_{i}"
            group_data = data_copy[:, group]

            transformed_data = dequantizer.named_transformers_[
                transformer_name
            ].inverse_transform(group_data)

            for j, feature_idx in enumerate(group):
                data_copy[:, feature_idx] = transformed_data[:, j]

        return data_copy
    else:
        X_train_copy = dataset.X_train.copy()
        X_test_copy = dataset.X_test.copy()

        for i, group in enumerate(dataset.categorical_features_lists):
            transformer_name = f"cat_group_{i}"

            group_train = X_train_copy[:, group]
            transformed_train = dequantizer.named_transformers_[
                transformer_name
            ].inverse_transform(group_train)
            for j, feature_idx in enumerate(group):
                X_train_copy[:, feature_idx] = transformed_train[:, j]

            group_test = X_test_copy[:, group]
            transformed_test = dequantizer.named_transformers_[
                transformer_name
            ].inverse_transform(group_test)
            for j, feature_idx in enumerate(group):
                X_test_copy[:, feature_idx] = transformed_test[:, j]

        dataset.X_train = X_train_copy
        dataset.X_test = X_test_copy
        return None


class CustomCategoricalTransformer(BaseEstimator, TransformerMixin):
    def _dequantize(self, x, rng):
        """
        Adds noise to pixels to dequantize them.
        Ensures the output stays in the valid range [0, 1].
        """
        for i in range(x.shape[1]):
            data_with_noise = x[:, i] + sigmoid(rng.randn(*x[:, i].shape))
            # data_with_noise = x[:, i] + rng.rand(*x[:, i].shape)
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


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
