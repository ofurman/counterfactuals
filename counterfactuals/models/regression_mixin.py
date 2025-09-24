from abc import ABC, abstractmethod

import numpy as np


class RegressionPytorchMixin(ABC):
    """
    Mixin class providing regression interface for PyTorch models.

    This mixin defines the standard interface for regression models,
    providing abstract methods for prediction and probability estimation.
    """

    @abstractmethod
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Make predictions on test data.

        Args:
            X_test: Input data as numpy array of shape (n_samples, n_features)

        Returns:
            np.ndarray: Predicted values of shape (n_samples,) or (n_samples, n_outputs)
        """
        pass

    @abstractmethod
    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        Return probabilities for each class on test data.

        Args:
            X_test: Input data as numpy array of shape (n_samples, n_features)

        Returns:
            np.ndarray: Class probabilities of shape (n_samples, n_classes)
                       Each row sums to 1.0 and represents probabilities for each class
        """
        pass
