import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod


def ordinal_to_one_hot(x, categorical_feature_indexes, number_of_categories):
    one_hot_encoded_features = []
    for i, feature_index in enumerate(categorical_feature_indexes):
        categorical_feature = x[:, feature_index].long()
        one_hot = F.one_hot(categorical_feature, num_classes=number_of_categories[i])
        one_hot_encoded_features.append(one_hot)

    non_categorical_features = [
        x[:, i : i + 1]
        for i in range(x.size(1))
        if i not in categorical_feature_indexes
    ]
    return torch.cat(non_categorical_features + one_hot_encoded_features, dim=1)


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
