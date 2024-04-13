import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from counterfactuals.datasets.base import AbstractDataset


class AuditDataset(AbstractDataset):
    def __init__(self, file_path: str = "data/audit.csv"):
        self.raw_data = self.load(file_path=file_path, index_col=False)
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

        target_column = raw_data.columns[-1]
        self.feature_columns = list(raw_data.columns[2:-1])
        self.feature_columns.remove("Detection_Risk")
        self.numerical_columns = list(range(0, len(self.feature_columns)))
        self.categorical_columns = []

        row_per_class = sum(raw_data[target_column] == 1)
        raw_data = pd.concat(
            [
                raw_data[raw_data[target_column] == 0].sample(
                    row_per_class, random_state=42
                ),
                raw_data[raw_data[target_column] == 1],
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
