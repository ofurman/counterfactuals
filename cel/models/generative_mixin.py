from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np


class GenerativePytorchMixin(ABC):
    """
    Mixin class providing generative model interface for PyTorch models.

    This mixin defines the standard interface for generative models,
    providing abstract methods for log probability prediction and sampling.
    """

    @abstractmethod
    def predict_log_proba(
        self, X_test: np.ndarray, context: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Predict log probabilities for input data.

        Args:
            X_test: Input data as numpy array of shape (n_samples, n_features)
            context: Context data as numpy array of shape (n_samples, n_features)
        Returns:
            np.ndarray: Log probabilities of shape (n_samples,) or (n_samples, n_classes)
        """
        pass

    @abstractmethod
    def sample_and_log_proba(
        self, n_samples: int, context: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample from the generative model and return log probabilities.

        Args:
            n_samples: Number of samples to generate
            context: Context data as numpy array of shape (n_samples, n_features)

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - Generated samples of shape (n_samples, n_features)
                - Log probabilities of shape (n_samples,)
        """
        pass
