import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from counterfactuals.datasets.base import AbstractDataset

SAMPLES_KEEP = 30000


class LendingClubDataset(AbstractDataset):
    def __init__(
        self,
        file_path: str = "data/lending_club.csv",
        transform=True,
        shuffle=True,
    ):
        """
        Initialize the GiveMeSomeCreditDataset dataset.
        """
        self.features = [
            "loan_amnt",
            "term",
            "int_rate",
            "installment",
            "grade",
            "emp_length",
            "home_ownership",
            "annual_inc",
            "dti",
            "delinq_2yrs",
            "fico_range_low",
            "fico_range_high",
            "loan_status",
        ]
        self.numerical_columns = [
            "loan_amnt",
            "int_rate",
            "installment",
            "annual_inc",
            "dti",
            "delinq_2yrs",
            "fico_range_low",
            "fico_range_high",
        ]
        self.categorical_columns = ["term", "grade", "emp_length", "home_ownership"]
        self.raw_data = self.load(file_path=file_path, index_col=False)
        self.raw_data = self.raw_data[
            self.numerical_columns + self.categorical_columns + ["loan_status"]
        ]

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
        self.feature_columns = self.features[:-1]  # Store original feature column names
        target_column = "loan_status"
        preprocess_columns = {"loan_status": ["Fully Paid", "Charged Off"]}
        for col, valid in preprocess_columns.items():
            raw_data[col] = raw_data[col].apply(lambda x: x if x in valid else None)
        raw_data[target_column] = raw_data[target_column].replace(
            {"Fully Paid": 1, "Charged Off": 0}
        )
        data_df = raw_data[self.features]
        data_df = data_df.dropna()
        data_df = data_df[:SAMPLES_KEEP]

        X = data_df[self.numerical_columns + self.categorical_columns].to_numpy()
        y = data_df[target_column].to_numpy()

        # Convert to indices for transformed data
        self.numerical_columns = list(range(0, len(self.numerical_columns)))
        self.categorical_columns = list(
            range(
                len(self.numerical_columns),
                len(self.feature_columns),
            )
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
