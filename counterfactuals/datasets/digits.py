import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler

from counterfactuals.datasets.base import AbstractDataset


class DigitsDataset(AbstractDataset):
    def __init__(self, file_path: str = "data/digits.csv"):
        self.raw_data = pd.DataFrame()
        self.X, self.y = self.preprocess(raw_data=self.raw_data)
        self.X_train, self.X_test, self.y_train, self.y_test = self.get_split_data(
            self.X, self.y
        )
        self.X_train, self.X_test, self.y_train, self.y_test = self.transform(
            self.X_train, self.X_test, self.y_train, self.y_test
        )

    def preprocess(self, raw_data: pd.DataFrame):
        """
        Preprocess the loaded data to X and y numpy arrays.
        """
        X, y = load_digits(n_class=10, return_X_y=True)

        X = X.reshape(X.shape[0], -1)

        self.numerical_columns = list(range(0, X.shape[1]))
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
        # self.feature_transformer = PCA(n_components=0.95)
        self.feature_transformer = MinMaxScaler()
        X_train = self.feature_transformer.fit_transform(X_train)
        X_test = self.feature_transformer.transform(X_test)

        y_train = y_train.reshape(-1)
        y_test = y_test.reshape(-1)

        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
        y_train = y_train.astype(np.int64)
        y_test = y_test.astype(np.int64)

        self.numerical_features = list(range(X_train.shape[1]))
        self.categorical_features = []
        self.actionable_features = self.numerical_features

        return X_train, X_test, y_train, y_test
