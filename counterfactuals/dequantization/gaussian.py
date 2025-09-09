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
        self.dataset = dataset

    def forward(self, X, y):
        if isinstance(X, torch.Tensor):
            X = X.numpy()

        X = self.dequantizer.dequantize(self.dataset, X)

        X = torch.from_numpy(X)
        log_probs = self.gen_model(X, y)
        return log_probs


class GaussianDequantizer:
    """
    Dequantizes categorical (discrete) features into continuous space using
    per-group Gaussian noise models, with an sklearn-like fit/transform API.

    This class wraps a set of ``CustomCategoricalTransformer`` instances inside
    a scikit-learn ``ColumnTransformer``. It provides a way to map categorical
    features to continuous representations (suitable for flows, VAEs, etc.)
    and invert the mapping back to the original categories. Supports both
    in-place dataset transformation and transformation of external arrays.

    Attributes:
        transformer (ColumnTransformer or None): Fitted column transformer
            composed of per-group ``CustomCategoricalTransformer``s.
            Initialized after calling ``fit()``.
        is_fitted (bool): Indicates whether the dequantizer has been fitted.
        dataset (object or None): Reference to the dataset passed during
            ``fit()``. Used for default transformations if no data is passed.
        dropped_numerical (list[int] or list[str]): Copy of
            ``dataset.numerical_columns``. Numerical features are not altered.

    Expected Dataset Interface:
        The ``dataset`` object must provide:
          - ``X_train (np.ndarray)``: Training matrix of shape (n_samples, n_features).
          - ``X_test (np.ndarray)``: Test matrix with the same width as ``X_train``.
          - ``categorical_features_lists (list[list[int]])``: Groups of column indices
            representing categorical features to be dequantized.
          - ``numerical_columns (list[int] or list[str])``: Indices or names of
            numerical columns (left unchanged).

    Notes:
        - This class focuses on *categorical* features, turning them into continuous
          variables by applying a stochastic, invertible mapping per categorical group.
        - Inversion is possible using ``quantize()``, which applies the
          corresponding ``inverse_transform`` on each categorical group.
        - The method follows the practice of *dequantization* for discrete data
          used in continuous likelihood models.

        For theoretical background and variants (e.g., variational/learned dequantization),
        see Ho et al., *“Dequantization for discrete data in flows”*, arXiv:2001.11235.
        https://arxiv.org/pdf/2001.11235

    Warnings:
        - ``fit()`` must be called before any transform/quantize operations.
          Otherwise, a ``ValueError`` is raised.
        - Methods without a ``data`` argument (e.g., ``transform()``,
          ``dequantize()``, ``quantize()``) will modify the dataset in place.
        - Assumes indices in ``categorical_features_lists`` are valid for the
          dataset matrices and consistent across train/test/external data.

    Examples:
        Fit on dataset and transform in place:

        >>> deq = GaussianDequantizer().fit(dataset)
        >>> deq.transform()  # modifies dataset.X_train and dataset.X_test in place

        Transform an external array (no mutation of dataset):

        >>> X_new_deq = deq.transform(data=X_new)

        Invert the dequantization (quantize) back to categories in place:

        >>> deq.quantize(dataset)

        Invert on external array:

        >>> X_orig = deq.quantize(dataset, data=X_new_deq)
    """

    def __init__(self):
        self.transformer = None
        self.is_fitted = False
        self.dataset = None

    def fit(self, dataset):
        """Fit the dequantizer on the dataset's training data.

        Args:
            dataset (Dataset): Dataset object containing
                `categorical_features_lists` and `X_train`.

        Returns:
            GaussianDequantizer: The fitted dequantizer instance for method chaining.
        """
        transformers = [
            (f"cat_group_{i}", CustomCategoricalTransformer(), group)
            for i, group in enumerate(dataset.categorical_features_lists)
        ]

        self.transformer = ColumnTransformer(
            transformers=transformers,
            remainder="drop",  # Drop continuous features
        )
        self.dropped_numerical = dataset.numerical_columns

        # Fit only on categorical features
        self.transformer.fit(dataset.X_train)
        self.is_fitted = True
        self.dataset = dataset

        return self

    def transform(self, data=None):
        return self.dequantize(self.dataset, data)

    def dequantize(self, dataset, data=None):
        """Apply dequantization to categorical features.

        Args:
            dataset (Dataset): Dataset object containing
                `categorical_features_lists`.
            data (np.ndarray, optional): External data to transform. If None,
                the method transforms `dataset.X_train` and `dataset.X_test`
                in place.

        Returns:
            np.ndarray or None: If `data` is provided, returns the transformed
            array. If `data` is None, modifies the dataset in place and
            returns None.
        """
        if not self.is_fitted:
            raise ValueError("Dequantizer must be fitted before use. Call fit() first.")

        if data is not None:
            return self._transform_data(dataset, data)
        else:
            # Transform dataset in-place
            self._transform_dataset_inplace(dataset)
            return None

    def quantize(self, dataset, data=None):
        """Apply inverse dequantization (quantization) to categorical features.

        Args:
            dataset (Dataset): Dataset object containing
                `categorical_features_lists`.
            data (np.ndarray, optional): External data to transform. If None,
                the method transforms `dataset.X_train` and `dataset.X_test`
                in place.

        Returns:
            np.ndarray | None: If `data` is provided, returns the transformed
            array. If `data` is None, modifies the dataset in place and
            returns None.
        """
        if not self.is_fitted:
            raise ValueError("Dequantizer must be fitted before use. Call fit() first.")

        if data is not None:
            return self._inverse_transform_data(dataset, data)
        else:
            # Transform dataset in-place
            self._inverse_transform_dataset_inplace(dataset)
            return None

    def _transform_data(self, dataset, data):
        """Transform external data."""
        data_copy = data.copy()

        for i, group in enumerate(dataset.categorical_features_lists):
            transformer_name = f"cat_group_{i}"
            group_data = data_copy[:, group]

            transformed_data = self.transformer.named_transformers_[
                transformer_name
            ].transform(group_data)

            for j, feature_idx in enumerate(group):
                data_copy[:, feature_idx] = transformed_data[:, j]

        return data_copy

    def _inverse_transform_data(self, dataset, data):
        """Inverse transform external data."""
        data_copy = data.copy()

        for i, group in enumerate(dataset.categorical_features_lists):
            transformer_name = f"cat_group_{i}"
            group_data = data_copy[:, group]

            transformed_data = self.transformer.named_transformers_[
                transformer_name
            ].inverse_transform(group_data)

            for j, feature_idx in enumerate(group):
                data_copy[:, feature_idx] = transformed_data[:, j]

        return data_copy

    def _transform_dataset_inplace(self, dataset):
        """Transform dataset in-place."""
        X_train_original = dataset.X_train.copy()
        X_test_original = dataset.X_test.copy()

        # Transform categorical features
        cat_transformed_train = self.transformer.transform(dataset.X_train)
        cat_transformed_test = self.transformer.transform(dataset.X_test)

        # Restore original data
        dataset.X_train = X_train_original.copy()
        dataset.X_test = X_test_original.copy()

        # Apply transformations to categorical features only
        cat_idx = 0
        for group in dataset.categorical_features_lists:
            for i, feature_idx in enumerate(group):
                dataset.X_train[:, feature_idx] = cat_transformed_train[:, cat_idx]
                dataset.X_test[:, feature_idx] = cat_transformed_test[:, cat_idx]
                cat_idx += 1

    def _inverse_transform_dataset_inplace(self, dataset):
        """Inverse transform dataset in-place."""
        X_train_copy = dataset.X_train.copy()
        X_test_copy = dataset.X_test.copy()

        for i, group in enumerate(dataset.categorical_features_lists):
            transformer_name = f"cat_group_{i}"

            # Transform training data
            group_train = X_train_copy[:, group]
            transformed_train = self.transformer.named_transformers_[
                transformer_name
            ].inverse_transform(group_train)
            for j, feature_idx in enumerate(group):
                X_train_copy[:, feature_idx] = transformed_train[:, j]

            # Transform test data
            group_test = X_test_copy[:, group]
            transformed_test = self.transformer.named_transformers_[
                transformer_name
            ].inverse_transform(group_test)
            for j, feature_idx in enumerate(group):
                X_test_copy[:, feature_idx] = transformed_test[:, j]

        dataset.X_train = X_train_copy
        dataset.X_test = X_test_copy


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
