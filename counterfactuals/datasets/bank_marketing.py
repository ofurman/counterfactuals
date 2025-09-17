import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from counterfactuals.datasets.base import AbstractDataset

SAMPLES_KEEP = 30000


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
        self.numerical_features = [
            "age",
            "balance",
            "day",
            "duration",
            "campaign",
            "pdays",
            "previous",
        ]
        self.categorical_features = [
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
            self._init_base_column_transformer()
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

        X = data_df[self.numerical_features + self.categorical_features].to_numpy()
        y = data_df[target_column].to_numpy()

        self.numerical_columns = list(range(0, len(self.numerical_features)))
        self.categorical_columns = list(
            range(len(self.numerical_features), len(self.feature_columns))
        )
        return X, y

    def _init_base_column_transformer(self):
        """
        Initialize base ColumnTransformer with OneHotEncoder fitted on full dataset
        to ensure consistent features across CV folds.
        """
        self.base_onehot_encoder = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False
        )
        self.base_onehot_encoder.fit(self.X[:, self.categorical_columns])

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
        scaler = MinMaxScaler()

        X_train_numerical = scaler.fit_transform(X_train[:, self.numerical_columns])
        X_test_numerical = scaler.transform(X_test[:, self.numerical_columns])

        X_train_categorical = self.base_onehot_encoder.transform(
            X_train[:, self.categorical_columns]
        )
        X_test_categorical = self.base_onehot_encoder.transform(
            X_test[:, self.categorical_columns]
        )

        X_train = np.concatenate([X_train_numerical, X_train_categorical], axis=1)
        X_test = np.concatenate([X_test_numerical, X_test_categorical], axis=1)

        self.feature_transformer = ColumnTransformer(
            [
                ("MinMaxScaler", scaler, self.numerical_columns),
                ("OneHotEncoder", self.base_onehot_encoder, self.categorical_columns),
            ],
        )

        X_train = np.array(X_train.astype(np.float32))
        X_test = np.array(X_test.astype(np.float32))
        y_train = np.array(y_train.astype(np.int64))
        y_test = np.array(y_test.astype(np.int64))
        self.numerical_features = list(range(0, len(self.numerical_columns)))
        self.categorical_features = list(
            range(len(self.numerical_columns), X_train.shape[1])
        )
        self.actionable_features = list(range(0, X_train.shape[1]))

        return X_train, X_test, y_train, y_test

    @property
    def categorical_features_lists(self) -> list:
        """
        Override the base class property to return correct categorical feature groupings
        based on actual one-hot encoded features.

        Returns:
            list: List of lists, where each inner list contains the indices of
                  one-hot encoded features for each original categorical variable.
        """
        categorical_features_lists = []
        current_idx = len(self.numerical_columns)

        for categories in self.base_onehot_encoder.categories_:
            n_categories = len(categories)
            categorical_features_lists.append(
                list(range(current_idx, current_idx + n_categories))
            )
            current_idx += n_categories

        return categorical_features_lists
