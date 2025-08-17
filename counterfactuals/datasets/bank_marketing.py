import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from counterfactuals.datasets.base import AbstractDataset

SAMPLES_KEEP = 100000


class BankMarketingDataset(AbstractDataset):
    def __init__(
        self,
        file_path: str = "data/bank_marketing.csv",
        transform=True,
        shuffle=True,
    ):
        """
        Initialize the GiveMeSomeCreditDataset dataset.
        """
        self.features = [
            "age",
            "job",
            "marital",
            "education",
            "default",
            "balance",
            "housing",
            "loan",
            "contact",
            "day",
            "month",
            "duration",
            "campaign",
            "pdays",
            "previous",
            "poutcome",
            "y",
        ]
        self.raw_data = self.load(file_path=file_path, index_col=False)
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
        target_column = "y"

        data_df = raw_data[self.features]

        self.numerical_columns = [
            "age",
            "balance",
            "duration",
            "campaign",
            "pdays",
            "previous",
        ]
        self.categorical_columns = [
            "job",
            "marital",
            "education",
            "default",
            "housing",
            "loan",
            "contact",
            "month",
            "poutcome",
        ]

        data_df = data_df.dropna()
        data_df = data_df[:SAMPLES_KEEP]
        X = data_df[self.numerical_columns + self.categorical_columns].to_numpy()
        y = data_df[target_column].to_numpy()
        self.numerical_columns = list(range(0, len(self.numerical_columns)))
        self.categorical_columns = list(
            range(
                len(self.numerical_columns),
                len(self.numerical_columns) + len(self.categorical_columns),
            )
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
                    OneHotEncoder(sparse_output=False),
                    self.categorical_columns,
                ),
            ],
        )
        # self.feature_transformer.set_output(transform='pandas')

        X_train = self.feature_transformer.fit_transform(X_train)
        X_test = self.feature_transformer.transform(X_test)

        self.y_transformer = OneHotEncoder(sparse_output=False)
        y_train = self.y_transformer.fit_transform(y_train.reshape(-1, 1))
        y_test = self.y_transformer.transform(y_test.reshape(-1, 1))

        X_train = np.array(X_train.astype(np.float32))
        X_test = np.array(X_test.astype(np.float32))
        y_train = np.array(y_train.astype(np.int64))
        y_test = np.array(y_test.astype(np.int64))

        self.numerical_features = list(range(0, len(self.features)))
        self.categorical_features = list(
            range(len(self.numerical_columns), X_train.shape[1])
        )

        return X_train, X_test, y_train, y_test
