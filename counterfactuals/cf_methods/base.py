from abc import ABC, abstractmethod

import numpy as np
from torch.utils.data import DataLoader


class BaseCounterfactual(ABC):
    @abstractmethod
    def explain(
        self,
        X: np.ndarray,
        y_origin: np.ndarray,
        y_target: np.ndarray,
        X_train: np.ndarray,
        y_train: np.ndarray,
        **kwargs,
    ):
        """
        Performs a single training step on a batch of data.
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
    ):
        """
        Search counterfactual explanations for the given dataloader.
        """
        pass
