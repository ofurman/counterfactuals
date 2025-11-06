from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PreprocessingContext:
    """Context holding data and feature indices for preprocessing.

    Attributes:
        X_train: Training features.
        X_test: Test features (optional).
        y_train: Training labels (optional).
        y_test: Test labels (optional).
        categorical_indices: Indices of categorical features.
        continuous_indices: Indices of continuous features.
    """

    X_train: np.ndarray
    X_test: Optional[np.ndarray] = None
    y_train: Optional[np.ndarray] = None
    y_test: Optional[np.ndarray] = None
    categorical_indices: Optional[list[int]] = None
    continuous_indices: Optional[list[int]] = None

    def __post_init__(self):
        """Infer feature indices if not provided."""
        n_features = self.X_train.shape[1]

        if self.categorical_indices is None:
            self.categorical_indices = []

        if self.continuous_indices is None:
            all_indices = set(range(n_features))
            self.continuous_indices = sorted(
                list(all_indices - set(self.categorical_indices))
            )


class PreprocessingStep(ABC):
    """Abstract base class for preprocessing steps.

    Each step operates on PreprocessingContext and transforms only relevant features.
    """

    @abstractmethod
    def fit(self, context: PreprocessingContext) -> "PreprocessingStep":
        """Fit the preprocessing step on training data.

        Args:
            context: Preprocessing context with data and feature indices.

        Returns:
            Self for method chaining.
        """
        pass

    @abstractmethod
    def transform(self, context: PreprocessingContext) -> PreprocessingContext:
        """Transform data in the context.

        Args:
            context: Preprocessing context with data to transform.

        Returns:
            New context with transformed data.
        """
        pass

    @abstractmethod
    def inverse_transform(self, context: PreprocessingContext) -> PreprocessingContext:
        """Inverse transform data in the context.

        Args:
            context: Preprocessing context with transformed data.

        Returns:
            New context with inverse transformed data.
        """
        pass
