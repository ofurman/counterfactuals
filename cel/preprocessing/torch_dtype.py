import numpy as np

from cel.preprocessing.base import PreprocessingContext, PreprocessingStep


class TorchDataTypeStep(PreprocessingStep):
    """Abstract base class for preprocessing steps.

    Each step operates on PreprocessingContext and transforms only relevant features.
    """

    def fit(self, context: PreprocessingContext) -> "TorchDataTypeStep":
        """Fit the preprocessing step on training data.

        Args:
            context: Preprocessing context with data and feature indices.

        Returns:
            Self for method chaining.
        """
        return self

    def transform(self, context: PreprocessingContext) -> PreprocessingContext:
        """Transform data in the context.

        Args:
            context: Preprocessing context with data to transform.

        Returns:
            New context with transformed data.
        """
        context.X_train = context.X_train.astype(np.float32)
        context.X_test = context.X_test.astype(np.float32)
        context.y_train = context.y_train.astype(np.int64)
        context.y_test = context.y_test.astype(np.int64)
        return context

    def inverse_transform(self, context: PreprocessingContext) -> PreprocessingContext:
        """Inverse transform data in the context.

        Args:
            context: Preprocessing context with transformed data.

        Returns:
            New context with inverse transformed data.
        """
        return context
