from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from torch.utils.data import DataLoader, TensorDataset


class AbstractDataset(ABC):
    def __init__(self, data=None):
        self.data = data

    @abstractmethod
    def preprocess(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
    ) -> Tuple[np.ndarray]:
        """
        Preprocess the loaded data, if necessary.
        """
        pass

    def load(self, file_path, **kwargs) -> pd.DataFrame:
        """
        Load data from a CSV file and store it in the 'data' attribute.
        """
        try:
            data = pd.read_csv(file_path, **kwargs)
            return data
        except Exception as e:
            raise FileNotFoundError(f"Error loading data from {file_path}: {e}")

    def save(self, file_path: str, data: pd.DataFrame, **kwargs):
        """
        Save the processed data (including scaled features) to a CSV file.
        """
        if data is not None:
            try:
                data.to_csv(file_path, **kwargs)
                print(f"Data saved to {file_path}")
            except Exception as e:
                print(f"Error saving data to {file_path}: {e}")
        else:
            print("No data to save.")

    def get_cv_splits(
        self, n_splits: int = 5, shuffle: bool = True, transform: bool = True
    ):
        """
        Sets and return the train and test splits for cross-validation.
        """
        cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=4)
        for train_idx, test_idx in cv.split(self.X, self.y):
            self.X_train, self.X_test = self.X[train_idx], self.X[test_idx]
            self.y_train, self.y_test = self.y[train_idx], self.y[test_idx]
            if transform:
                self.X_train, self.X_test, self.y_train, self.y_test = self.transform(
                    X_train=self.X_train,
                    X_test=self.X_test,
                    y_train=self.y_train,
                    y_test=self.y_test,
                )
            yield self.X_train, self.X_test, self.y_train, self.y_test

    def get_target_class_splits(self, target_class: int) -> Tuple[np.ndarray]:
        """
        Sets and return the train and test splits for cross-validation.
        """
        try:
            target_train_idx = np.where(self.y_train == target_class)[0]
            target_test_idx = np.where(self.y_test == target_class)[0]
            return (
                self.X_train[target_train_idx],
                self.X_test[target_test_idx],
                self.y_train[target_train_idx],
                self.y_test[target_test_idx],
            )
        except AttributeError:
            raise AttributeError(
                "X_train, X_test, y_train, and y_test must be set before calling this method."
            )

    def get_non_target_class_splits(self, target_class: int) -> Tuple[np.ndarray]:
        """
        Sets and return the train and test splits for cross-validation.
        """
        try:
            non_target_train_idx = np.where(self.y_train != target_class)[0]
            non_target_test_idx = np.where(self.y_test != target_class)[0]
            return (
                self.X_train[non_target_train_idx],
                self.X_test[non_target_test_idx],
                self.y_train[non_target_train_idx],
                self.y_test[non_target_test_idx],
            )
        except AttributeError:
            raise AttributeError(
                "X_train, X_test, y_train, and y_test must be set before calling this method."
            )

    def get_split_data(self, X: np.ndarray, y: np.ndarray, shuffle: bool = True):
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            random_state=4,
            test_size=0.2,
            shuffle=shuffle,
            stratify=y if shuffle else None,
        )
        return X_train, X_test, y_train, y_test

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
            n_cat = self.raw_data[self.feature_columns[col]].nunique()
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


class LawDataset(AbstractDataset):
    def __init__(self, file_path: str = "data/law.csv", transform=True, shuffle=True):
        """
        Initialize the Law School dataset.
        """
        self.categorical_features = []
        # include target column as last feature
        self.features = [
            "lsat",
            "gpa",
            "zfygpa",
            "sex",
            "race",
            "pass_bar",
        ]
        self.raw_data = self.load(file_path=file_path)
        self.X, self.y = self.preprocess(raw_data=self.raw_data)
        self.X_train, self.X_test, self.y_train, self.y_test = self.get_split_data(
            self.X, self.y, shuffle=shuffle
        )
        if transform:
            self.X_train, self.X_test, self.y_train, self.y_test = self.transform(
                self.X_train, self.X_test, self.y_train, self.y_test
            )

    def preprocess(self, raw_data: pd.DataFrame):
        """
        Preprocess the loaded data to X and y numpy arrays.
        """
        self.feature_columns = ["lsat", "gpa", "zfygpa", "sex", "race"]
        # self.feature_columns = ["lsat", "gpa", "zfygpa"]
        target_column = "pass_bar"
        self.numerical_columns = list(range(0, 3))
        self.categorical_columns = list(range(3, len(self.feature_columns)))
        # self.categorical_columns = []

        # Downsample to minor class
        raw_data = raw_data.dropna(subset=self.feature_columns)
        row_per_class = sum(raw_data[target_column] == 0)
        raw_data = pd.concat(
            [
                raw_data[raw_data[target_column] == 0],
                raw_data[raw_data[target_column] == 1].sample(
                    row_per_class, random_state=42
                ),
            ]
        )

        X = raw_data[self.feature_columns].to_numpy()
        y = raw_data[target_column].to_numpy()
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

        self.feature_transformer = ColumnTransformer(
            [
                ("MinMaxScaler", MinMaxScaler(), self.numerical_columns),
                (
                    "OneHotEncoder",
                    OneHotEncoder(sparse_output=False),
                    self.categorical_columns,
                ),
            ],
        )
        # self.feature_transformer = MinMaxScaler()
        X_train = self.feature_transformer.fit_transform(X_train)
        X_test = self.feature_transformer.transform(X_test)

        # target_transformer = LabelEncoder()
        # y_train = self.target_transformer.fit_transform(y_train.reshape(-1, 1))
        # y_test = self.target_transformer.transform(y_test.reshape(-1, 1))
        y_train = y_train.reshape(-1)
        y_test = y_test.reshape(-1)

        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
        y_train = y_train.astype(np.int64)
        y_test = y_test.astype(np.int64)

        self.numerical_features = list(range(0, len(self.numerical_columns)))
        self.categorical_features = list(
            range(len(self.numerical_columns), X_train.shape[1])
        )
        self.actionable_features = list(range(0, X_train.shape[1]))

        return X_train, X_test, y_train, y_test
