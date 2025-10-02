from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader


class ClassifierPytorchMixin(ABC):
    """
    Abstract mixin class defining the classifier interface.

    This mixin provides the standard classifier methods that should be
    implemented by all PyTorch-based classifiers.
    """

    @abstractmethod
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict class labels for input data.

        Args:
            X_test: Input data as numpy array

        Returns:
            Predicted class labels as numpy array
        """
        pass

    @abstractmethod
    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for input data.

        Args:
            X_test: Input data as numpy array

        Returns:
            Class probabilities for each class as numpy array
        """
        pass


class PytorchBase(torch.nn.Module, ClassifierPytorchMixin, ABC):
    """
    Base class for PyTorch-based discriminative models.

    This class provides a common interface for all PyTorch discriminative models
    used in the counterfactuals library. It combines PyTorch's nn.Module with
    the classifier interface defined in ClassifierPytorchMixin.
    """

    def __init__(self, num_inputs: int, num_targets: int):
        """
        Initialize the PyTorch base model.

        Args:
            num_inputs: Number of input features
            num_targets: Number of target classes/outputs
        """
        super(PytorchBase, self).__init__()
        self.num_inputs = num_inputs
        self.num_targets = num_targets

    def save(self, path: str) -> None:
        """Save model state to file."""
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        """Load model state from file."""
        self.load_state_dict(torch.load(path))

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        pass

    @abstractmethod
    def fit(
        self,
        train_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        epochs: int = 200,
        lr: float = 0.003,
        **kwargs,
    ) -> None:
        """
        Train the model.

        Args:
            train_loader: Training data loader
            test_loader: Optional test data loader for validation
            epochs: Number of training epochs
            lr: Learning rate
            **kwargs: Additional training parameters
        """
        pass

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Make predictions on input data.

        Args:
            x: Input data as numpy array

        Returns:
            Predictions as numpy array
        """
        pass

    @abstractmethod
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for input data.

        Args:
            x: Input data as numpy array

        Returns:
            Class probabilities as numpy array
        """
        pass
