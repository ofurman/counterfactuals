from typing import Optional

import numpy as np
from sklearn.preprocessing import MinMaxScaler as SklearnMinMaxScaler
from sklearn.preprocessing import QuantileTransformer as SklearnQuantileTransformer
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler

from counterfactuals.preprocessing.base import PreprocessingContext, PreprocessingStep


class QuantileTransformCategoricalStep(PreprocessingStep):
    """Quantile-transform the (one-hot) categorical columns only.

    Mirrors the original DiCoFlex generation space
    (``ofurman/DiCoFlex``, ``datasets/DCENF/*.py``): small Gaussian tie-breaking
    noise is added to the one-hot columns so the ``QuantileTransformer`` sees a
    non-degenerate distribution, then a ``QuantileTransformer`` is fit on those
    columns. Numerical columns are left untouched (a MinMax step run alongside
    this one handles them). Inverse-transform clips samples back into the fitted
    quantile range, which bounds the categorical block before the downstream
    argmax one-hot discretisation.

    The noise is applied only while FITTING (to fit a smooth quantile map); at
    transform time clean 0/1 blocks map near the extremes of the output range,
    which is the intended behaviour for one-hot inputs.
    """

    def __init__(self, noise_std: float = 0.05):
        self.noise_std = noise_std
        self.transformer: Optional[SklearnQuantileTransformer] = None
        self._categorical_indices: Optional[list[int]] = None

    def fit(self, context: PreprocessingContext) -> "QuantileTransformCategoricalStep":
        self._categorical_indices = context.categorical_indices
        if len(self._categorical_indices) > 0:
            X_cat = context.X_train[:, self._categorical_indices].astype(np.float64, copy=True)
            # Gaussian tie-breaking noise so the quantile map is not degenerate on
            # binary 0/1 columns (matches the reference implementation).
            X_cat = X_cat + np.random.normal(0.0, self.noise_std, size=X_cat.shape)
            self.transformer = SklearnQuantileTransformer()
            self.transformer.fit(X_cat)
        return self

    def transform(self, context: PreprocessingContext) -> PreprocessingContext:
        if len(self._categorical_indices) == 0 or self.transformer is None:
            return context
        return PreprocessingContext(
            X_train=self._transform_array(context.X_train),
            X_test=None if context.X_test is None else self._transform_array(context.X_test),
            y_train=context.y_train,
            y_test=context.y_test,
            categorical_indices=context.categorical_indices,
            continuous_indices=context.continuous_indices,
        )

    def inverse_transform(self, context: PreprocessingContext) -> PreprocessingContext:
        if len(self._categorical_indices) == 0 or self.transformer is None:
            return context
        return PreprocessingContext(
            X_train=self._inverse_transform_array(context.X_train),
            X_test=None
            if context.X_test is None
            else self._inverse_transform_array(context.X_test),
            y_train=context.y_train,
            y_test=context.y_test,
            categorical_indices=context.categorical_indices,
            continuous_indices=context.continuous_indices,
        )

    def _transform_array(self, X: np.ndarray) -> np.ndarray:
        out = X.copy()
        cat = X[:, self._categorical_indices].astype(np.float64)
        out[:, self._categorical_indices] = self.transformer.transform(cat)
        return out

    def _inverse_transform_array(self, X: np.ndarray) -> np.ndarray:
        out = X.copy()
        cat = X[:, self._categorical_indices].astype(np.float64)
        out[:, self._categorical_indices] = self.transformer.inverse_transform(cat)
        return out


class MinMaxScalingStep(PreprocessingStep):
    """Min-max normalization for continuous features only.

    This step uses sklearn's MinMaxScaler internally and only transforms
    continuous features specified in the context.

    Attributes:
        scaler: Internal sklearn MinMaxScaler instance.
        feature_range: Target range for scaled features.
    """

    def __init__(self, feature_range: tuple[float, float] = (0.0, 1.0)):
        """Initialize MinMaxScaler.

        Args:
            feature_range: Desired range of transformed data (min, max).
        """
        self.feature_range = feature_range
        self.scaler: Optional[SklearnMinMaxScaler] = None
        self._continuous_indices: Optional[list[int]] = None
        self._categorical_indices: Optional[list[int]] = None

    def fit(self, context: PreprocessingContext) -> "MinMaxScalingStep":
        """Fit the scaler on continuous features from training data.

        Args:
            context: Preprocessing context with training data and feature indices.

        Returns:
            Self for method chaining.
        """
        self._continuous_indices = context.continuous_indices
        self._categorical_indices = context.categorical_indices

        if len(self._continuous_indices) > 0:
            # Extract and fit on continuous features only
            X_cont = context.X_train[:, self._continuous_indices]
            self.scaler = SklearnMinMaxScaler(feature_range=self.feature_range)
            self.scaler.fit(X_cont)

        return self

    def transform(self, context: PreprocessingContext) -> PreprocessingContext:
        """Scale continuous features to the specified range.

        Args:
            context: Preprocessing context with data to transform.

        Returns:
            New context with transformed data.
        """
        if len(self._continuous_indices) == 0 or self.scaler is None:
            # No continuous features, return unchanged
            return context

        # Transform train data
        X_train_transformed = self._transform_array(context.X_train)

        # Transform test data if present
        X_test_transformed = None
        if context.X_test is not None:
            X_test_transformed = self._transform_array(context.X_test)

        return PreprocessingContext(
            X_train=X_train_transformed,
            X_test=X_test_transformed,
            y_train=context.y_train,
            y_test=context.y_test,
            categorical_indices=context.categorical_indices,
            continuous_indices=context.continuous_indices,
        )

    def inverse_transform(self, context: PreprocessingContext) -> PreprocessingContext:
        """Reverse the scaling transformation.

        Args:
            context: Preprocessing context with transformed data.

        Returns:
            New context with inverse transformed data.
        """
        if len(self._continuous_indices) == 0 or self.scaler is None:
            # No continuous features, return unchanged
            return context

        # Inverse transform train data
        X_train_inv = self._inverse_transform_array(context.X_train)

        # Inverse transform test data if present
        X_test_inv = None
        if context.X_test is not None:
            X_test_inv = self._inverse_transform_array(context.X_test)

        return PreprocessingContext(
            X_train=X_train_inv,
            X_test=X_test_inv,
            y_train=context.y_train,
            y_test=context.y_test,
            categorical_indices=context.categorical_indices,
            continuous_indices=context.continuous_indices,
        )

    def _transform_array(self, X: np.ndarray) -> np.ndarray:
        """Transform a single array by scaling continuous features.

        Args:
            X: Input array with shape (n_samples, n_features).

        Returns:
            Transformed array with continuous features scaled.
        """
        X_transformed = X.copy()
        X_cont = X[:, self._continuous_indices]
        X_cont_scaled = self.scaler.transform(X_cont)
        X_transformed[:, self._continuous_indices] = X_cont_scaled
        return X_transformed

    def _inverse_transform_array(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform a single array by unscaling continuous features.

        Args:
            X: Transformed array with scaled continuous features.

        Returns:
            Original scale array.
        """
        X_inv = X.copy()
        X_cont = X[:, self._continuous_indices]
        X_cont_unscaled = self.scaler.inverse_transform(X_cont)
        X_inv[:, self._continuous_indices] = X_cont_unscaled
        return X_inv


class StandardScalingStep(PreprocessingStep):
    """Standardization for continuous features only.

    This step uses sklearn's StandardScaler internally and only transforms
    continuous features specified in the context.

    Attributes:
        scaler: Internal sklearn StandardScaler instance.
    """

    def __init__(self):
        """Initialize StandardScaler."""
        self.scaler: Optional[SklearnStandardScaler] = None
        self._continuous_indices: Optional[list[int]] = None
        self._categorical_indices: Optional[list[int]] = None

    def fit(self, context: PreprocessingContext) -> "StandardScalingStep":
        """Fit the scaler on continuous features from training data.

        Args:
            context: Preprocessing context with training data and feature indices.

        Returns:
            Self for method chaining.
        """
        self._continuous_indices = context.continuous_indices
        self._categorical_indices = context.categorical_indices

        if len(self._continuous_indices) > 0:
            # Extract and fit on continuous features only
            X_cont = context.X_train[:, self._continuous_indices]
            self.scaler = SklearnStandardScaler()
            self.scaler.fit(X_cont)

        return self

    def transform(self, context: PreprocessingContext) -> PreprocessingContext:
        """Standardize continuous features to zero mean and unit variance.

        Args:
            context: Preprocessing context with data to transform.

        Returns:
            New context with transformed data.
        """
        if len(self._continuous_indices) == 0 or self.scaler is None:
            # No continuous features, return unchanged
            return context

        # Transform train data
        X_train_transformed = self._transform_array(context.X_train)

        # Transform test data if present
        X_test_transformed = None
        if context.X_test is not None:
            X_test_transformed = self._transform_array(context.X_test)

        return PreprocessingContext(
            X_train=X_train_transformed,
            X_test=X_test_transformed,
            y_train=context.y_train,
            y_test=context.y_test,
            categorical_indices=context.categorical_indices,
            continuous_indices=context.continuous_indices,
        )

    def inverse_transform(self, context: PreprocessingContext) -> PreprocessingContext:
        """Reverse the standardization transformation.

        Args:
            context: Preprocessing context with transformed data.

        Returns:
            New context with inverse transformed data.
        """
        if len(self._continuous_indices) == 0 or self.scaler is None:
            # No continuous features, return unchanged
            return context

        # Inverse transform train data
        X_train_inv = self._inverse_transform_array(context.X_train)

        # Inverse transform test data if present
        X_test_inv = None
        if context.X_test is not None:
            X_test_inv = self._inverse_transform_array(context.X_test)

        return PreprocessingContext(
            X_train=X_train_inv,
            X_test=X_test_inv,
            y_train=context.y_train,
            y_test=context.y_test,
            categorical_indices=context.categorical_indices,
            continuous_indices=context.continuous_indices,
        )

    def _transform_array(self, X: np.ndarray) -> np.ndarray:
        """Transform a single array by standardizing continuous features.

        Args:
            X: Input array with shape (n_samples, n_features).

        Returns:
            Transformed array with continuous features standardized.
        """
        X_transformed = X.copy()
        X_cont = X[:, self._continuous_indices]
        X_cont_scaled = self.scaler.transform(X_cont)
        X_transformed[:, self._continuous_indices] = X_cont_scaled
        return X_transformed

    def _inverse_transform_array(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform a single array by unstandardizing continuous features.

        Args:
            X: Transformed array with standardized continuous features.

        Returns:
            Original scale array.
        """
        X_inv = X.copy()
        X_cont = X[:, self._continuous_indices]
        X_cont_unscaled = self.scaler.inverse_transform(X_cont)
        X_inv[:, self._continuous_indices] = X_cont_unscaled
        return X_inv
