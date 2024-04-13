import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from counterfactuals.datasets.base import AbstractDataset


class CompasDataset(AbstractDataset):
    def __init__(self, file_path: str = "data/compas_two_years.csv"):
        self.raw_data = self.load(file_path=file_path, index_col=False)
        self.X, self.y = self.preprocess(raw_data=self.raw_data)
        self.X_train, self.X_test, self.y_train, self.y_test = self.get_split_data(
            self.X, self.y
        )

    def preprocess(self, raw_data: pd.DataFrame):
        """
        Preprocess the loaded data to X and y numpy arrays.
        """

        self.feature_columns = [
            # Continuous
            "age",
            "priors_count",
            "days_b_screening_arrest",
            "length_of_stay",
            "is_recid",
            "is_violent_recid",
            "two_year_recid",
            # Categorical
            # "c_charge_degree",
            # "sex",
            # "race",
        ]
        self.numerical_columns = list(range(0, 7))
        self.categorical_columns = list(range(7, len(self.feature_columns)))
        target_column = "class"

        raw_data["days_b_screening_arrest"] = np.abs(
            raw_data["days_b_screening_arrest"]
        )
        raw_data["c_jail_out"] = pd.to_datetime(raw_data["c_jail_out"])
        raw_data["c_jail_in"] = pd.to_datetime(raw_data["c_jail_in"])
        raw_data["length_of_stay"] = np.abs(
            (raw_data["c_jail_out"] - raw_data["c_jail_in"]).dt.days
        )
        raw_data["length_of_stay"].fillna(
            raw_data["length_of_stay"].value_counts().index[0], inplace=True
        )
        raw_data["days_b_screening_arrest"].fillna(
            raw_data["days_b_screening_arrest"].value_counts().index[0], inplace=True
        )
        raw_data["length_of_stay"] = raw_data["length_of_stay"].astype(int)
        raw_data["days_b_screening_arrest"] = raw_data[
            "days_b_screening_arrest"
        ].astype(int)
        # raw_data = raw_data[raw_data["score_text"] != "Medium"]
        # raw_data["class"] = pd.get_dummies(raw_data["score_text"])["High"].astype(int)
        raw_data["class"] = (
            raw_data["score_text"].map({"Low": 0, "Medium": 1, "High": 2}).astype(int)
        )
        raw_data.drop(["c_jail_in", "c_jail_out", "score_text"], axis=1, inplace=True)

        # Downsample to minor class
        raw_data = raw_data.dropna(subset=self.feature_columns)
        rows_per_class = raw_data[target_column].value_counts().min()
        raw_data = pd.concat(
            [
                raw_data[raw_data[target_column] == class_label].sample(
                    rows_per_class, random_state=42
                )
                for class_label in raw_data[target_column].unique()
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

        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
        y_train = y_train.astype(np.int64)
        y_test = y_test.astype(np.int64)

        self.numerical_features = list(range(0, len(self.numerical_columns)))
        self.categorical_features = list(
            range(len(self.numerical_columns), X_train.shape[1])
        )
        self.actionable_features = list(range(0, X_train.shape[1]))

        return X_train, X_test, y_train, y_test
