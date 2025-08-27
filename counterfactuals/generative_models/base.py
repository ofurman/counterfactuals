from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseGenModel(nn.Module, ABC):
    def __init__(self):
        super(BaseGenModel, self).__init__()

    @abstractmethod
    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        num_epochs: int = 100,
        learning_rate: float = 1e-3,
        patience: int = 20,
        eps: float = 1e-3,
        checkpoint_path: str = "best_model.pth",
    ):
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor, context: torch.Tensor = None):
        pass

    @abstractmethod
    def predict_log_prob(self, dataloader: torch.utils.data.DataLoader):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass
