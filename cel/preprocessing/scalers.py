import numpy as np
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from cel.preprocessing.base import PreprocessingContext, PreprocessingStep


class ScalingStep(PreprocessingStep):
    """Base class for scaling continuous features using an sklearn-compatible scaler.

    Subclasses set ``self.scaler`` to an (unfitted) sklearn transformer instance.
    Only continuous feature columns identified in the PreprocessingContext are scaled;
    categorical columns are left unchanged.
    """

    scaler: TransformerMixin

    def fit(self, context: PreprocessingContext) -> "ScalingStep":
        """Fit the scaler on continuous features from training data.

        Args:
            context: Preprocessing context with training data and feature indices.

        Returns:
            Self for method chaining.
        """
        self._continuous_indices: list[int] = context.continuous_indices
        self._categorical_indices: list[int] = context.categorical_indices

        if len(self._continuous_indices) > 0:
            X_cont = context.X_train[:, self._continuous_indices]
            self.scaler.fit(X_cont)

        return self

    def transform(self, context: PreprocessingContext) -> PreprocessingContext:
        """Scale continuous features.

        Args:
            context: Preprocessing context with data to transform.

        Returns:
            New context with transformed data.
        """
        if len(self._continuous_indices) == 0:
            return context

        return PreprocessingContext(
            X_train=self._transform_array(context.X_train),
            X_test=self._transform_array(context.X_test) if context.X_test is not None else None,
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
        if len(self._continuous_indices) == 0:
            return context

        return PreprocessingContext(
            X_train=self._inverse_transform_array(context.X_train),
            X_test=(
                self._inverse_transform_array(context.X_test)
                if context.X_test is not None
                else None
            ),
            y_train=context.y_train,
            y_test=context.y_test,
            categorical_indices=context.categorical_indices,
            continuous_indices=context.continuous_indices,
        )

    def _transform_array(self, X: np.ndarray) -> np.ndarray:
        X_transformed = X.copy()
        X_transformed[:, self._continuous_indices] = self.scaler.transform(
            X[:, self._continuous_indices]
        )
        return X_transformed

    def _inverse_transform_array(self, X: np.ndarray) -> np.ndarray:
        X_inv = X.copy()
        X_inv[:, self._continuous_indices] = self.scaler.inverse_transform(
            X[:, self._continuous_indices]
        )
        return X_inv


class MinMaxScalingStep(ScalingStep):
    """Min-max normalization for continuous features.

    Wraps sklearn's MinMaxScaler and applies it only to continuous feature columns.

    Args:
        feature_range: Desired range of transformed data (min, max). Defaults to (0.0, 1.0).

    Examples:
        MinMaxScalingStep()                   # scales to [0, 1]
        MinMaxScalingStep(feature_range=(-1, 1))  # scales to [-1, 1]
    """

    def __init__(self, feature_range: tuple[float, float] = (0.0, 1.0)):
        self.scaler: MinMaxScaler = MinMaxScaler(feature_range=feature_range)


class StandardScalingStep(ScalingStep):
    """Standardization for continuous features (zero mean, unit variance).

    Wraps sklearn's StandardScaler and applies it only to continuous feature columns.

    Examples:
        StandardScalingStep()
    """

    def __init__(self):
        self.scaler: StandardScaler = StandardScaler()
