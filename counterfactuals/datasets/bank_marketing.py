import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

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
            "previous",
            "pdays",
            "poutcome",
            "y",
        ]
        self.numerical_columns = [
            "age",
            "balance",
            "day",
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
        raw_data = raw_data.copy()
        target_column = "y"
        self.feature_columns = self.features[:-1]

        data_df = raw_data[self.features]
        data_df = data_df.dropna()
        data_df = data_df[:SAMPLES_KEEP]
        data_df[target_column] = data_df[target_column].replace({"yes": 1, "no": 0})

        X = data_df[self.numerical_columns + self.categorical_columns].to_numpy()
        y = data_df[target_column].to_numpy()

        # Set up indices for internal use
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

        X_train = np.array(X_train.astype(np.float32))
        X_test = np.array(X_test.astype(np.float32))
        y_train = np.array(y_train.astype(np.int64))
        y_test = np.array(y_test.astype(np.int64))

        self.numerical_features = list(range(0, len(self.numerical_columns)))
        self.categorical_features = list(
            range(len(self.numerical_columns), X_train.shape[1])
        )

        return X_train, X_test, y_train, y_test
