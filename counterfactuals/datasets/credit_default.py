import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from counterfactuals.datasets.base import AbstractDataset


class CreditDefaultDataset(AbstractDataset):
    def __init__(
        self, file_path: str = "data/credit_card.csv", transform=True, shuffle=True
    ):
        self.raw_data = self.load(file_path=file_path, index_col=False)
        self.features = [
            "LIMIT_BAL",
            "SEX",
            "EDUCATION",
            "MARRIAGE",
            "AGE",
            "PAY_0",
            "PAY_2",
            "PAY_3",
            "PAY_4",
            "PAY_5",
            "PAY_6",
            "BILL_AMT1",
            "BILL_AMT2",
            "BILL_AMT3",
            "BILL_AMT4",
            "BILL_AMT5",
            "BILL_AMT6",
            "PAY_AMT1",
            "PAY_AMT2",
            "PAY_AMT3",
            "PAY_AMT4",
            "PAY_AMT5",
            "PAY_AMT6",
            "default payment next month",
        ]
        self.numerical_columns = [
            "LIMIT_BAL",
            "AGE",
            "BILL_AMT1",
            "BILL_AMT2",
            "BILL_AMT3",
            "BILL_AMT4",
            "BILL_AMT5",
            "BILL_AMT6",
            "PAY_AMT1",
            "PAY_AMT2",
            "PAY_AMT3",
            "PAY_AMT4",
            "PAY_AMT5",
            "PAY_AMT6",
        ]
        self.categorical_columns = [
            "SEX",
            "EDUCATION",
            "MARRIAGE",
            "PAY_0",
            "PAY_2",
            "PAY_3",
            "PAY_4",
            "PAY_5",
            "PAY_6",
        ]

        self.X, self.y = self.preprocess(raw_data=self.raw_data)
        self.X_train, self.X_test, self.y_train, self.y_test = self.get_split_data(
            self.X, self.y
        )
        if transform:
            self.X_train, self.X_test, self.y_train, self.y_test = self.transform(
                self.X_train, self.X_test, self.y_train, self.y_test
            )

    def preprocess(self, raw_data: pd.DataFrame):
        """
        Preprocess the loaded data to X and y numpy arrays.
        """
        raw_data = raw_data.copy()
        target_column = "default payment next month"
        self.feature_columns = self.features[:-1]

        raw_data = raw_data.dropna()

        X = raw_data[self.feature_columns].to_numpy()
        y = raw_data[target_column].to_numpy()

        self.numerical_columns = list(range(0, len(self.numerical_columns)))
        self.categorical_columns = list(
            range(len(self.numerical_columns), len(self.feature_columns))
        )

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
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    self.categorical_columns,
                ),
            ],
        )

        X_train = self.feature_transformer.fit_transform(X_train)
        X_test = self.feature_transformer.transform(X_test)

        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
        y_train = y_train.astype(np.int64)
        y_test = y_test.astype(np.int64)

        self.numerical_columns = list(range(0, len(self.numerical_columns)))
        self.categorical_columns = list(
            range(len(self.numerical_columns), len(self.feature_columns))
        )

        self.actionable_features = list(range(0, X_train.shape[1]))

        return X_train, X_test, y_train, y_test
