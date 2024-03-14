from abc import ABC, abstractmethod

import torch
from torch.utils.data import DataLoader, TensorDataset


class AbstractDataset(ABC):
    def __init__(self, data=None):
        self.data = data

    @abstractmethod
    def load(self, file_path):
        """
        Load data from a file or source and store it in the 'data' attribute.
        """
        pass

    @abstractmethod
    def preprocess(self):
        """
        Preprocess the loaded data, if necessary.
        """
        pass

    @abstractmethod
    def save(self, file_path):
        """
        Save the processed data to a file or destination.
        """
        pass

    @abstractmethod
    def get_split_data(self):
        """
        Return X_train, X_test, y_train, y_test.
        """

    def train_dataloader(
        self, batch_size: int, shuffle: bool, noise_lvl=0, **kwargs_dataloader
    ):
        def collate_fn(batch):
            X, y = zip(*batch)
            X = torch.stack(X)
            y = torch.stack(y)

            # Add Gaussian noise to train features
            noise = torch.randn_like(X[:, self.numerical_features]) * noise_lvl
            X[:, self.numerical_features] = X[:, self.numerical_features] + noise
            return X, y

        return DataLoader(
            TensorDataset(
                torch.from_numpy(self.X_train), torch.from_numpy(self.y_train)
            ),
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn if noise_lvl else None,
            **kwargs_dataloader,
        )

    def test_dataloader(self, batch_size: int, shuffle: bool, **kwargs_dataloader):
        return DataLoader(
            TensorDataset(torch.from_numpy(self.X_test), torch.from_numpy(self.y_test)),
            batch_size=batch_size,
            shuffle=shuffle,
            **kwargs_dataloader,
        )

    @property
    def categorical_features_lists(self) -> list:
        categorical_features_lists = []
        for col in self.categorical_columns:
            n_cat = self.data[self.feature_columns[col]].nunique()
            # n_cat = 1 if n_cat == 2 else n_cat
            if len(categorical_features_lists) == 0:
                categorical_features_lists.append(
                    list(
                        range(
                            len(self.numerical_columns),
                            len(self.numerical_columns) + n_cat,
                        )
                    )
                )
            else:
                categorical_features_lists.append(
                    list(
                        range(
                            categorical_features_lists[-1][-1] + 1,
                            categorical_features_lists[-1][-1] + 1 + n_cat,
                        )
                    )
                )
        return categorical_features_lists
