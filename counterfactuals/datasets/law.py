from typing import Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

import torch

from counterfactuals.datasets.base import AbstractDataset

ALPHA = 1e-6


class CustomCategoricalTransformer(BaseEstimator, TransformerMixin):
    def _dequantize(self, x, rng):
        """
        Adds noise to pixels to dequantize them.
        Ensures the output stays in the valid range [0, 1].
        """
        for i in range(x.shape[1]):
            # data_with_noise = x[:, i] + sigmoid(rng.randn(*x[:, i].shape))
            data_with_noise = x[:, i] + rng.rand(*x[:, i].shape)
            divider = self.dividers[i]
            x[:, i] = data_with_noise / divider
        return x

    def _logit_transform(self, x):
        """
        Transforms pixel values with logit to be unconstrained.
        """
        x = ALPHA + (1 - 2 * ALPHA) * x
        return np.log(x / (1.0 - x))

    @staticmethod
    def inverse(x: Union[np.ndarray, torch.Tensor], dividers: list) -> np.ndarray:
        """
        Inverse transform the logit transformation.
        Handles both numpy arrays and torch tensors.
        """
        x = x.copy()
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = (torch.sigmoid(x) - 1e-6) / (1 - 2e-6)
        for i in range(x.shape[1]):
            bins = np.linspace(0, 1, dividers[i])
            x[:, i] = np.digitize(x[:, i], bins) - 1
        return x.numpy()

    def fit(self, X, y=None):
        self.dividers = [X[:, i].max() + 1 for i in range(X.shape[1])]
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed = self._dequantize(X_transformed, np.random.RandomState(42))
        X_transformed = self._logit_transform(X_transformed)
        return X_transformed

    def inverse_transform(self, X):
        """
        Inverse transform the logit transformation.
        Handles both numpy arrays and torch tensors.
        """
        x = X.copy()
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        x = (torch.sigmoid(torch.from_numpy(x)).numpy() - 1e-6) / (1 - 2e-6)
        for i in range(x.shape[1]):
            bins = np.linspace(0, 1, self.dividers[i] + 1)
            x[:, i] = np.digitize(x[:, i], bins) - 1
        return x


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


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
                    "OneHotPipeline",
                    Pipeline(
                        [
                            ("OneHotEncoder", OneHotEncoder(sparse_output=False)),
                            ("CategoricalQuantizer", CustomCategoricalTransformer()),
                        ]
                    ),
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
        self.intervals = get_categorical_intervals(self)

        return X_train, X_test, y_train, y_test


def get_categorical_intervals(dataset):
    categorical_subset = dataset.raw_data[dataset.feature_columns].iloc[
        :, dataset.categorical_columns
    ]
    categorical_subset.columns
    num_unique_feature_values = []

    for col in categorical_subset.columns:
        unique_feature_values = len(categorical_subset[col].dropna().unique())
        num_unique_feature_values.append(unique_feature_values)
    intervals = []
    start = dataset.numerical_columns[-1] + 1
    for num in num_unique_feature_values:
        intervals.append((start, start + num))
        start += num

    return intervals
