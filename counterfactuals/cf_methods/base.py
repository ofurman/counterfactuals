from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from counterfactuals.discriminative_models.base import BaseDiscModel
from counterfactuals.generative_models.base import BaseGenModel


@dataclass
class ExplanationResult:
    """
    Dataclass for storing the result of a counterfactual explanation.
    """

    x_cfs: np.ndarray
    y_cf_targets: np.ndarray
    x_origs: np.ndarray
    y_origs: np.ndarray


class BaseCounterfactual(ABC):
    @abstractmethod
    def __init__(
        self,
        gen_model: BaseGenModel,
        disc_model: BaseDiscModel,
        disc_model_criterion: torch.nn.modules.loss._Loss = None,
        device: str = None,
    ):
        """Initialize the class with arg1 and arg2."""
        pass

    @abstractmethod
    def explain(
        self,
        X: np.ndarray,
        y_origin: np.ndarray,
        y_target: np.ndarray,
        X_train: np.ndarray,
        y_train: np.ndarray,
        **kwargs,
    ) -> ExplanationResult:
        """
        Return single explanation for a data point.
        """
        pass

    @abstractmethod
    def explain_dataloader(
        self,
        dataloader: DataLoader,
        epochs: int,
        lr: float,
        patience_eps,
        **search_step_kwargs,
    ) -> ExplanationResult:
        """
        Search counterfactual explanations for the given dataloader.
        """
        pass
