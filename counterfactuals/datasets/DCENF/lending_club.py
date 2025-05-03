from typing import Union
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, QuantileTransformer
import torch
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from counterfactuals.datasets.base import AbstractDataset


# lending-club
# Total 100000
# Train 50000
# Test 50000
# Num 8
# Cat 4
# Numeric ['loan_amnt', 'int_rate', 'installment', 'annual_inc', 'dti', 'delinq_2yrs', 'fico_range_low', 'fico_range_high']
# Categorical ['term', 'grade', 'emp_length', 'home_ownership']
# Classification Target ['loan_status']

class LendingClubDataset(AbstractDataset):
    alpha = 1e-6

    def __init__(
        self, 
        train_file_path: str = "data/lending-club/train.csv",
        test_file_path: str = "data/lending-club/test.csv"
    ):
        self.train_data = self.load(file_path=train_file_path, index_col=False)
        self.test_data = self.load(file_path=test_file_path, index_col=False)
        
        # Preprocess train and test data separately
        self.X_train, self.y_train = self.preprocess(raw_data=self.train_data, pred_path="data/lending-club/y_train_pred.npy")
        self.X_test, self.y_test = self.preprocess(raw_data=self.test_data, pred_path="data/lending-club/y_test_pred.npy")
        
        # Transform the data
        self.X_train, self.X_test, self.y_train, self.y_test = self.transform(
            self.X_train, self.X_test, self.y_train, self.y_test
        )

    def preprocess(self, raw_data: pd.DataFrame, pred_path: str):
        """
        Preprocess the loaded data to X and y numpy arrays.
        """
        self.feature_columns = [
            # Continuous
            "loan_amnt",
            "int_rate",
            "installment",
            "annual_inc",
            "dti",
            "delinq_2yrs",
            "fico_range_low",
            "fico_range_high",
            # Categorical
            "term",
            "grade",
            "emp_length",
            "home_ownership",
        ]
        self.numerical_columns = list(range(0, 8))  # First 8 columns are numerical
        self.categorical_columns = list(range(8, len(self.feature_columns)))
        target_column = "loan_status"

        # Drop any rows with missing values
        raw_data = raw_data.dropna(subset=self.feature_columns)

        # Convert target to binary (1 for good loans, 0 for bad loans)
        # Assuming 'Fully Paid' and 'Current' are good loans, others are bad
        raw_data[target_column] = raw_data[target_column].apply(
            lambda x: 1 if x in ['Fully Paid', 'Current'] else 0
        ).astype(int)

        X = raw_data[self.feature_columns].to_numpy()
        y = raw_data[target_column].to_numpy()
        y = np.load(pred_path)
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
        X_train = self.feature_transformer.fit_transform(X_train)
        X_test = self.feature_transformer.transform(X_test)

        y_train = y_train.reshape(-1)
        y_test = y_test.reshape(-1)

        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
        y_train = y_train.astype(np.int64)
        y_test = y_test.astype(np.int64)

        self.numerical_features = list(range(0, len(self.numerical_columns)))
        self.categorical_features = list(
            range(len(self.numerical_columns), X_train.shape[1])
        )
        self.actionable_features = list(range(0, X_train.shape[1]))

        X_train[:, self.categorical_features] += (
            np.random.normal(
                0,
                0.05,
                size=(X_train.shape[0], len(self.categorical_features))
            )
        )
        self.qt = QuantileTransformer()
        X_train[:, self.categorical_features] = self.qt.fit_transform(X_train[:, self.categorical_features])
        X_test[:, self.categorical_features] = self.qt.transform(X_test[:, self.categorical_features])

        return X_train, X_test, y_train, y_test 