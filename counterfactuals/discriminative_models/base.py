from abc import ABC, abstractmethod

import torch
from torch.utils.data import DataLoader


class BaseDiscModel(torch.nn.Module, ABC):
    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def fit(self, train_loader: DataLoader, epochs: int = 200, lr: float = 0.003):
        pass

    @abstractmethod
    def predict(self, X_test: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def predict_proba(self, X_test: torch.Tensor) -> torch.Tensor:
        pass
