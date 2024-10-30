import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from counterfactuals.datasets.base import AbstractDataset


class BlobsDataset(AbstractDataset):
    def __init__(self, file_path: str = "data/blobs.csv", method=None, n_bins=None):
        self.raw_data = self.load(file_path=file_path, header=None)
        if method in ["ares", "globe-ce"]:
            self.raw_data.columns = ["0", "1", "2"]
            self.raw_data = self.raw_data[
                (self.raw_data["2"] == 0) | (self.raw_data["2"] == 1)
            ]
            self.feature_columns = ["0", "1"]
            self.n_bins = n_bins
            self.categorical_features = []
            self.X, self.y = self.ares_one_hot(self.raw_data), self.raw_data["2"]
            self.X, self.y = self.X.to_numpy().astype(np.float32), self.y.to_numpy()
        else:
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

        X = raw_data[raw_data.columns[:-1]].to_numpy()
        y = raw_data[raw_data.columns[-1]].to_numpy()

        self.numerical_features = [0, 1]
        self.categorical_features = []
        self.actionable_features = [0, 1]
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

        y_train = y_train.reshape(-1)
        y_test = y_test.reshape(-1)

        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
        y_train = y_train.astype(np.int64)
        y_test = y_test.astype(np.int64)

        return X_train, X_test, y_train, y_test
