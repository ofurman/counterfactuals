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
        # Based on dataset info: numerical features are indices [0, 4, 11-22]
        # Categorical features are indices [1, 2, 3, 5-10]
        # Convert to indices in the feature_columns list
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

        self.numerical_feature_names = [
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
        self.categorical_feature_names = [
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

        self.target_column = "default payment next month"

        # Clean data by removing rows with missing values in feature columns
        self.raw_data = self.raw_data.dropna(subset=self.features)
        self.raw_data = self.raw_data[
            self.numerical_feature_names
            + self.categorical_feature_names
            + [self.target_column]
        ]
        self.X, self.y = self.preprocess(raw_data=self.raw_data)
        self.X_train, self.X_test, self.y_train, self.y_test = self.get_split_data(
            self.X, self.y
        )
        if transform:
            # Initialize base ColumnTransformer with OneHotEncoder fitted on full data
            self._init_base_column_transformer()
            self.X_train, self.X_test, self.y_train, self.y_test = self.transform(
                self.X_train, self.X_test, self.y_train, self.y_test
            )

    def _init_base_column_transformer(self):
        """
        Initialize base ColumnTransformer with OneHotEncoder fitted on full dataset
        to ensure consistent features across CV folds.
        """
        self.base_onehot_encoder = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False
        )
        self.base_onehot_encoder.fit(self.X[:, self.categorical_columns])

    def preprocess(self, raw_data: pd.DataFrame):
        """
        Preprocess the loaded data to X and y numpy arrays.
        """
        self.feature_columns = self.features[:-1]

        X = raw_data[self.feature_columns].to_numpy()
        y = raw_data[self.target_column].to_numpy()

        self.numerical_columns = list(range(0, len(self.numerical_feature_names)))
        self.categorical_columns = list(
            range(len(self.numerical_feature_names), len(self.feature_columns))
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
                ("OneHotEncoder", self.base_onehot_encoder, self.categorical_columns),
            ],
        )
        # self.feature_transformer.set_output(transform='pandas')

        X_train = self.feature_transformer.fit_transform(X_train)
        X_test = self.feature_transformer.transform(X_test)

        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
        y_train = y_train.astype(np.int64)
        y_test = y_test.astype(np.int64)

        self.numerical_features = list(range(0, len(self.numerical_columns)))
        self.categorical_features = list(
            range(len(self.numerical_columns), X_train.shape[1])
        )

        # Make all features actionable as requested
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
        if not hasattr(self, "feature_transformer") or not hasattr(
            self.feature_transformer, "transformers_"
        ):
            return []

        categorical_features_lists = []
        current_idx = len(self.numerical_columns)  # Start after numerical features

        # Get the OneHotEncoder transformer
        for name, transformer, columns in self.feature_transformer.transformers_:
            if name == "OneHotEncoder" and hasattr(transformer, "categories_"):
                for categories in transformer.categories_:
                    n_categories = len(categories)
                    categorical_features_lists.append(
                        list(range(current_idx, current_idx + n_categories))
                    )
                    current_idx += n_categories
                break

        return categorical_features_lists
