from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import numpy as np
from torch.utils.data import DataLoader

from counterfactuals.models.pytorch_base import PytorchBase


@dataclass
class ExplanationResult:
    """
    Data structure for storing the result of a counterfactual explanation.

    This dataclass encapsulates all the important outputs from a counterfactual
    explanation process, including the generated counterfactuals, their targets,
    the original instances, and any additional logging information.

    Attributes:
        x_cfs (np.ndarray): Generated counterfactual examples.
        y_cf_targets (np.ndarray): Target labels/values for the counterfactuals.
        x_origs (np.ndarray): Original input instances.
        y_origs (np.ndarray): Original labels/values for the input instances.
        logs (Optional[Dict[str, Any]]): Additional logging information such as
            loss curves, convergence metrics, or method-specific data.
    """

    x_cfs: np.ndarray
    y_cf_targets: np.ndarray
    x_origs: np.ndarray
    y_origs: np.ndarray
    logs: Optional[Dict[str, Any]] = None
    cf_group_ids: Optional[np.ndarray] = None


class BaseCounterfactualMethod(ABC):
    """
    Abstract base class for all counterfactual explanation methods.

    This class defines the interface that all counterfactual methods must implement.
    It provides a consistent API for fitting, explaining, and generating counterfactuals
    across different methodological approaches.

    The class supports both individual explanations and batch processing through
    DataLoader objects, making it suitable for various use cases from single
    instance explanations to large-scale evaluations.

    Attributes:
        gen_model: Generative model used for counterfactual generation (if applicable).
        disc_model (PytorchBase): Discriminative/classification model to be explained.
        disc_model_criterion: Loss function for the discriminative model.
        device (str): Computing device ('cpu' or 'cuda') for PyTorch operations.
    """

    def __init__(
        self,
        gen_model: Optional[Any] = None,
        disc_model: Optional[PytorchBase] = None,
        disc_model_criterion: Optional[Any] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the counterfactual method.

        Args:
            gen_model (Optional[Any]): Generative model for CF generation. Can be None
                for methods that don't use generative models.
            disc_model (Optional[PytorchBase]): The model to be explained. Should be
                a PyTorch-based model wrapped in our PytorchBase interface.
            disc_model_criterion (Optional[Any]): Loss function for the discriminative
                model. Required by optimization-based methods.
            device (Optional[str]): Device for computation. Defaults to 'cpu'.
            **kwargs: Additional method-specific parameters.
        """
        self.gen_model = gen_model
        self.disc_model = disc_model
        self.disc_model_criterion = disc_model_criterion
        self.device = device or "cpu"

        # Move models to device if they exist and have a .to() method
        if self.gen_model is not None and hasattr(self.gen_model, "to"):
            self.gen_model.to(self.device)
        if self.disc_model is not None and hasattr(self.disc_model, "to"):
            self.disc_model.to(self.device)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> None:
        """
        Fit the counterfactual method on training data.

        This method allows the counterfactual explanation method to learn
        from training data. This might involve training auxiliary models,
        learning data distributions, or other preparatory steps.

        Args:
            X_train (np.ndarray): Training features with shape (n_samples, n_features).
            y_train (np.ndarray): Training labels with shape (n_samples,).
            **kwargs: Additional method-specific parameters.
        """
        pass

    @abstractmethod
    def explain(
        self,
        X: np.ndarray,
        y_origin: np.ndarray,
        y_target: np.ndarray,
        X_train: Optional[np.ndarray] = None,
        y_train: Optional[np.ndarray] = None,
        **kwargs,
    ) -> ExplanationResult:
        """
        Generate counterfactual explanations for given instances.

        This is the core method that generates counterfactual explanations
        for input instances. It should return counterfactuals that, when
        passed through the model, produce the desired target outcomes.

        Args:
            X (np.ndarray): Input instances to explain with shape (n_instances, n_features).
            y_origin (np.ndarray): Original predictions/labels for X with shape (n_instances,).
            y_target (np.ndarray): Desired target predictions/labels with shape (n_instances,).
            X_train (Optional[np.ndarray]): Training data, if needed by the method.
            y_train (Optional[np.ndarray]): Training labels, if needed by the method.
            **kwargs: Additional method-specific parameters.

        Returns:
            ExplanationResult: Object containing counterfactuals, targets, originals,
                and any additional logging information.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """
        raise NotImplementedError("Subclasses must implement the explain method")

    @abstractmethod
    def explain_dataloader(
        self,
        dataloader: DataLoader,
        epochs: int,
        lr: float,
        patience_eps: Union[float, int] = 1e-5,
        **search_step_kwargs,
    ) -> ExplanationResult:
        """
        Generate counterfactual explanations for data provided via DataLoader.

        This method is designed for batch processing of counterfactual generation,
        particularly useful for optimization-based methods that require iterative
        search procedures. It processes data in batches and typically involves
        gradient-based optimization.

        Args:
            dataloader (DataLoader): PyTorch DataLoader containing (X, y) pairs
                where X are instances to explain and y are their labels.
            epochs (int): Maximum number of optimization epochs per instance.
            lr (float): Learning rate for optimization procedures.
            patience_eps (Union[float, int]): Convergence threshold. When loss
                drops below this value, optimization can terminate early.
            **search_step_kwargs: Additional parameters passed to the search
                step function, such as regularization weights, constraints, etc.

        Returns:
            ExplanationResult: Object containing all generated counterfactuals,
                their targets, original instances, and detailed logging information
                including loss curves and convergence metrics.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """
        raise NotImplementedError(
            "Subclasses must implement the explain_dataloader method"
        )
