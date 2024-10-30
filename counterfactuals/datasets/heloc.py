import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from counterfactuals.datasets.base import AbstractDataset


class HelocDataset(AbstractDataset):
    def __init__(self, file_path: str = "data/heloc.csv", method=None, n_bins=None):
        self.raw_data = self.load(file_path=file_path)

        if method in ["ares", "globe-ce"]:
            target_column = "RiskPerformance"
            self.feature_columns = self.raw_data.columns.drop(target_column)
            self.n_bins = n_bins
            self.categorical_features = []
            self.raw_data = self.ares_prepro(self.raw_data)
            self.X, self.y = (
                self.ares_one_hot(self.raw_data),
                self.raw_data["RiskPerformance"],
            )
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
        target_column = "RiskPerformance"
        self.feature_columns = raw_data.columns.drop(target_column)

        self.numerical_columns = list(range(0, len(self.feature_columns)))
        self.categorical_columns = []

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
        self.feature_transformer = MinMaxScaler()
        X_train = self.feature_transformer.fit_transform(X_train)
        X_test = self.feature_transformer.transform(X_test)

        self.target_transformer = LabelEncoder()
        y_train = self.target_transformer.fit_transform(y_train)
        y_test = self.target_transformer.transform(y_test)

        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
        y_train = y_train.astype(np.int64)
        y_test = y_test.astype(np.int64)

        self.categorical_features = []
        self.numerical_features = list(range(0, len(self.feature_columns)))

        return X_train, X_test, y_train, y_test

    def ares_prepro(self, data):
        data = data[(data.iloc[:, 1:] >= 0).any(axis=1)]
        data["RiskPerformance"] = data["RiskPerformance"].replace(
            ["Bad", "Good"], [0, 1]
        )
        y = data.pop("RiskPerformance")
        data["RiskPerformance"] = y
        data = data[data >= 0]
        nan_cols = data.isnull().any(axis=0)
        for col in data.columns:
            if nan_cols[col]:
                data[col] = data[col].replace(np.nan, np.nanmedian(data[col]))
        return data
