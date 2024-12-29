from typing import Tuple
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

import torch
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

    def get_cv_splits(self, n_splits: int = 5):
        """
        Sets and return the train and test splits for cross-validation.
        """
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=4)
        for train_idx, test_idx in cv.split(self.X, self.y):
            self.X_train, self.X_test = self.X[train_idx], self.X[test_idx]
            self.y_train, self.y_test = self.y[train_idx], self.y[test_idx]
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

    def get_split_data(self, X: np.ndarray, y: np.ndarray):
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            random_state=4,
            test_size=0.2,
            shuffle=True,
            stratify=y,
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

    def ares_one_hot(self, data):
        """
        Improvised method for one-hot encoding the data

        Input: data (whole dataset)
        Outputs: data_oh (one-hot encoded data)
                 features (list of feature values after one-hot encoding)
        """

        label_encoder = LabelEncoder()
        data_encode = data.copy()
        self.bins = {}
        self.bins_tree = {}
        self.features_tree = {}

        # Assign encoded features to one hot columns
        data_oh, features = [], []
        for x in data.columns[:-1]:
            self.features_tree[x] = []
            categorical = x in self.categorical_features
            if categorical:
                data_encode[x] = label_encoder.fit_transform(data_encode[x])
                cols = label_encoder.classes_
            elif self.n_bins is not None:
                data_encode[x] = pd.cut(
                    data_encode[x].apply(lambda x: float(x)), bins=self.n_bins
                )
                cols = data_encode[x].cat.categories
                self.bins_tree[x] = {}
            else:
                data_oh.append(data[x])
                features.append(x)
                continue

            one_hot = pd.get_dummies(data_encode[x])
            data_oh.append(one_hot)
            for col in cols:
                feature_value = x + " = " + str(col)
                features.append(feature_value)
                self.features_tree[x].append(feature_value)
                if not categorical:
                    self.bins[feature_value] = col.mid
                    self.bins_tree[x][feature_value] = col.mid

        data_oh = pd.concat(data_oh, axis=1, ignore_index=True)
        data_oh.columns = features
        self.features = features
        return data_oh
