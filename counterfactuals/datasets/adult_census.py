import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from counterfactuals.datasets.base import AbstractDataset


class AdultCensusDataset(AbstractDataset):
    def __init__(self, file_path: str = "data/adult_census.csv"):
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

        self.feature_columns = [
            "age",
            "workclass",
            "education",
            "marital.status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "capital.gain",
            "capital.loss",
            "hours.per.week",
            "native.country",
        ]

        # Based on dataset info: numerical features are indices [0, 8, 9, 10]
        # Categorical features are indices [1, 2, 3, 4, 5, 6, 7, 11]
        # Convert to indices in the feature_columns list
        self.numerical_feature_names = [
            "age",
            "capital.gain",
            "capital.loss",
            "hours.per.week",
        ]
        self.categorical_feature_names = [
            "workclass",
            "education",
            "marital.status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native.country",
        ]
        self.target_column = "income"
        self.numerical_columns = list(range(0, len(self.numerical_feature_names)))
        self.categorical_columns = list(
            range(len(self.numerical_feature_names), len(self.feature_columns))
        )

        # Preprocess native.country column - keep only United-States, others become "Other"
        raw_data["native.country"] = raw_data["native.country"].apply(
            lambda x: x if x.strip() == "United-States" else "Other"
        )

        # Clean data by removing rows with missing values in feature columns
        raw_data = raw_data.dropna(subset=self.feature_columns)

        # Remove rows with '?' values which are missing value indicators in this dataset
        for col in self.feature_columns:
            raw_data = raw_data[raw_data[col].astype(str).str.strip() != "?"]

        # Process target column - convert to binary
        raw_data[self.target_column] = raw_data[self.target_column].apply(
            lambda x: 1 if x.strip() == ">50K" else 0
        )

        X = raw_data[
            self.numerical_feature_names + self.categorical_feature_names
        ].to_numpy()
        y = raw_data[self.target_column].to_numpy()

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

        self.numerical_features = list(range(0, len(self.numerical_columns)))
        self.categorical_features = list(
            range(len(self.numerical_columns), X_train.shape[1])
        )

        # Define actionable features - exclude immutable features: marital.status, sex, race
        # Map original feature names to their indices in self.feature_columns
        immutable_feature_names = ["marital.status", "sex", "race"]
        immutable_original_indices = []

        for feature_name in immutable_feature_names:
            if feature_name in self.feature_columns:
                immutable_original_indices.append(
                    self.feature_columns.index(feature_name)
                )

        # Get feature names after transformation
        feature_names = self.feature_transformer.get_feature_names_out()

        immutable_indices = []
        for i, feature_name in enumerate(feature_names):
            # Check if this transformed feature corresponds to an immutable original feature
            for orig_idx in immutable_original_indices:
                if f"x{orig_idx}" in feature_name:
                    immutable_indices.append(i)
                    break

        # Set actionable features as all features except immutable ones
        self.actionable_features = [
            i for i in range(X_train.shape[1]) if i not in immutable_indices
        ]

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
