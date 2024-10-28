import torch
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from counterfactuals.datasets.base import AbstractDataset


class Scm20dDataset(AbstractDataset):
    def __init__(self, file_path: str = "data/regression/scm20d.csv"):
        self.raw_data = self.load(file_path=file_path, index_col=False)
        self.X, self.y = self.preprocess(raw_data=self.raw_data)
        self.X_train, self.X_test, self.y_train, self.y_test = self.get_split_data(
            self.X, self.y
        )
        self.X_train, self.X_test, self.y_train, self.y_test = self.transform(
            self.X_train, self.X_test, self.y_train, self.y_test
        )

    def get_split_data(self, X: np.ndarray, y: np.ndarray):
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            random_state=4,
            test_size=0.1,
            shuffle=True,
        )
        return X_train, X_test, y_train, y_test

    def preprocess(self, raw_data: pd.DataFrame):
        """
        Preprocess the loaded data to X and y numpy arrays.
        """
        X = raw_data[raw_data.columns[:-16]].to_numpy()
        y = raw_data[raw_data.columns[-16:]].to_numpy()

        self.feature_columns = list(raw_data.columns[:-16])
        self.numerical_features = list(range(0, len(self.feature_columns)))
        self.categorical_features = []
        self.categorical_columns = []

        return X, y

    def transform(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
    ):
        """
        Transform the loaded data by applying Min-Max scaling to the features.
        """
        self.feature_transformer = MinMaxScaler()
        X_train = self.feature_transformer.fit_transform(X_train)
        X_test = self.feature_transformer.transform(X_test)

        self.label_transformer = MinMaxScaler()
        y_train = self.label_transformer.fit_transform(y_train)
        y_test = self.label_transformer.transform(y_test)

        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
        y_train = y_train.astype(np.float32)
        y_test = y_test.astype(np.float32)

        self.categorical_columns = []

        return X_train, X_test, y_train, y_test

    def train_dataloader(
        self,
        batch_size: int,
        shuffle: bool,
        noise_lvl=0,
        label_noise_lvl=0,
        **kwargs_dataloader,
    ):
        def collate_fn(batch):
            X, y = zip(*batch)
            X = torch.stack(X)
            y = torch.stack(y)

            # Add Gaussian noise to train features
            noise = torch.randn_like(X[:, self.numerical_features]) * noise_lvl
            X[:, self.numerical_features] = X[:, self.numerical_features] + noise

            # Add Gaussian noise to train labels
            noise = torch.randn_like(y) * noise_lvl
            y = y + noise
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
