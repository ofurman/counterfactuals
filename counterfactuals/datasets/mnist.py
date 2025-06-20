import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from counterfactuals.datasets.base import AbstractDataset


class MNISTDataset(AbstractDataset):
    alpha = 1e-6

    def __init__(self, file_path: str = "data/mnist.csv"):
        self.alpha = 1e-6
        self.raw_data = self.load(file_path=file_path, header=None)
        self.X, self.y = self.preprocess(raw_data=self.raw_data)
        self.X_train, self.X_test, self.y_train, self.y_test = self.get_split_data(
            self.X, self.y
        )
        self.X_train, self.X_test, self.y_train, self.y_test = self.transform(
            self.X_train, self.X_test, self.y_train, self.y_test
        )

    @staticmethod
    def _dequantize(x, rng):
        """
        Adds noise to pixels to dequantize them.
        Ensures the output stays in the valid range [0, 1].
        """
        x = x + rng.rand(*x.shape) / 256.0
        return np.clip(x, 0, 1)

    @staticmethod
    def _logit_transform(x):
        """
        Transforms pixel values with logit to be unconstrained.
        """
        x = MNISTDataset.alpha + (1 - 2 * MNISTDataset.alpha) * x
        return np.log(x / (1.0 - x))

    def preprocess(self, raw_data: pd.DataFrame):
        """
        Preprocess the loaded data to X and y numpy arrays.
        """
        self.categorical_columns = []
        raw_data = raw_data[raw_data[0].isin([1, 7])].iloc[0:500]
        X = raw_data[raw_data.columns[1:]].to_numpy()

        rng = np.random.RandomState(42)
        X = self._dequantize(X, rng)
        X = self._logit_transform(X)
        y = raw_data[raw_data.columns[0]].replace({1: 0, 7: 1}).to_numpy()

        self.numerical_features = list(range(0, X.shape[1]))
        self.categorical_features = []
        self.actionable_features = list(range(0, X.shape[1]))
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

        y_train = y_train.reshape(-1)
        y_test = y_test.reshape(-1)

        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
        y_train = y_train.astype(np.int64)
        y_test = y_test.astype(np.int64)
        return X_train, X_test, y_train, y_test
