from typing import Union
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer

from counterfactuals.datasets.base import AbstractDataset


class DigitsDataset(AbstractDataset):
    alpha = 1e-6

    def __init__(
        self, file_path: str = "data/digits.csv", transform=True, shuffle=True
    ):
        self.alpha = 1e-6
        self.categorical_features = []
        self.features = ["pixel" + str(i) for i in range(1, 65)] + ["digit"]
        self.raw_data = self.load(file_path=file_path, header=None)
        self.X, self.y = self.preprocess(raw_data=self.raw_data)
        self.X_train, self.X_test, self.y_train, self.y_test = self.get_split_data(
            self.X,
            self.y,
            shuffle=shuffle,
        )
        if transform:
            self.X_train, self.X_test, self.y_train, self.y_test = self.transform(
                self.X_train, self.X_test, self.y_train, self.y_test
            )

    @staticmethod
    def _dequantize(x, rng):
        """
        Adds noise to pixels to dequantize them.
        Ensures the output stays in the valid range [0, 1].
        """
        x = (x + rng.rand(*x.shape)) / 17.0
        return x

    @staticmethod
    def _logit_transform(x):
        """
        Transforms pixel values with logit to be unconstrained.
        """
        x = DigitsDataset.alpha + (1 - 2 * DigitsDataset.alpha) * x
        return np.log(x / (1.0 - x))

    @staticmethod
    def inverse_transform(x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Inverse transform the logit transformation.
        Handles both numpy arrays and torch tensors.
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = (torch.sigmoid(x) - 1e-6) / (1 - 2e-6)
        # bins = np.linspace(0, 1, 256)
        # x = np.digitize(x, bins) - 1
        return x.numpy()

    def preprocess(self, raw_data: pd.DataFrame):
        """
        Preprocess the loaded data to X and y numpy arrays.
        """
        self.categorical_columns = []
        # raw_data = raw_data[raw_data[raw_data.columns[-1]].isin([3, 8])]
        X = raw_data[raw_data.columns[:-1]].to_numpy()

        rng = np.random.RandomState(42)
        X = self._dequantize(X, rng)
        X = self._logit_transform(X)
        y = raw_data[raw_data.columns[-1]].to_numpy()
        # y = raw_data[raw_data.columns[-1]].replace({3: 0, 8: 1}).to_numpy()

        self.numerical_features = list(range(0, X.shape[1]))
        self.numerical_columns = list(range(0, X.shape[1]))
        self.categorical_features = []
        self.categorical_columns = []
        self.actionable_features = list(range(0, X.shape[1]))
        self.not_actionable_features = []
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
        self.feature_transformer = FunctionTransformer(lambda x: np.array(x))
        X_train = self.feature_transformer.fit_transform(X_train)
        X_test = self.feature_transformer.transform(X_test)

        # add one hot encoder for y
        self.y_transformer = OneHotEncoder(sparse=False)
        y_train = self.y_transformer.fit_transform(y_train.reshape(-1, 1))
        y_test = self.y_transformer.transform(y_test.reshape(-1, 1))

        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
        y_train = y_train.astype(np.int64)
        y_test = y_test.astype(np.int64)
        return X_train, X_test, y_train, y_test
