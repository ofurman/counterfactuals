from abc import ABC, abstractmethod

import numpy as np
import torch
from torch.utils.data import DataLoader


class PytorchBase(torch.nn.Module, ABC):
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

    def _to_tensor(self, X: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Convert input to a float32 tensor on the model's device.

        Args:
            X: Input data as numpy array or tensor.

        Returns:
            Float32 tensor on the same device as the model parameters, so
            prediction helpers keep working after the model is moved to GPU.
        """
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        return X.to(next(self.parameters()).device)

    def save(self, path: str) -> None:
        """Save model state to file."""
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        """Load model state from file onto the device the model currently lives on.

        The explicit ``map_location`` lets a checkpoint trained on GPU be loaded
        into a CPU-resident model (and vice versa), which the train-on-GPU /
        generate-on-CPU split relies on.
        """
        self.load_state_dict(torch.load(path, map_location=next(self.parameters()).device))

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
        test_loader: DataLoader | None = None,
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
