from typing import Union
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, QuantileTransformer
import torch
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from counterfactuals.datasets.base import AbstractDataset


# Total 100000
# Train 50000
# Test 50000
# Num 7
# Cat 3
# Numeric ['RevolvingUtilizationOfUnsecuredLines', 'age', 'DebtRatio', 'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines']
# Categorical ['NumberOfTime30-59DaysPastDueNotWorse', 'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents']
# Classification Target ['SeriousDlqin2yrs']

class GMCDataset(AbstractDataset):
    alpha = 1e-6

    def __init__(
        self, 
        train_file_path: str = "data/gmc/train.csv",
        test_file_path: str = "data/gmc/test.csv"
    ):
        self.train_data = self.load(file_path=train_file_path, index_col=False)
        self.test_data = self.load(file_path=test_file_path, index_col=False)
        
        # Preprocess train and test data separately
        self.X_train, self.y_train = self.preprocess(raw_data=self.train_data)
        self.X_test, self.y_test = self.preprocess(raw_data=self.test_data)
        
        # Transform the data
        self.X_train, self.X_test, self.y_train, self.y_test = self.transform(
            self.X_train, self.X_test, self.y_train, self.y_test
        )

    def preprocess(self, raw_data: pd.DataFrame):
        """
        Preprocess the loaded data to X and y numpy arrays.
        """
        self.feature_columns = [
            # Continuous
            "RevolvingUtilizationOfUnsecuredLines",
            "age",
            "DebtRatio",
            "MonthlyIncome",
            "NumberOfOpenCreditLinesAndLoans",
            "NumberOfTimes90DaysLate",
            "NumberRealEstateLoansOrLines",
            # Categorical
            "NumberOfTime30-59DaysPastDueNotWorse",
            "NumberOfTime60-89DaysPastDueNotWorse",
            "NumberOfDependents",
        ]
        self.numerical_columns = list(range(0, 7))  # First 7 columns are numerical
        self.categorical_columns = list(range(7, len(self.feature_columns)))
        target_column = "SeriousDlqin2yrs"

        # Convert target to binary if needed
        raw_data[target_column] = raw_data[target_column].astype(int)

        # Drop any rows with missing values
        raw_data = raw_data.dropna(subset=self.feature_columns)

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
                0.1,
                size=(X_train.shape[0], len(self.categorical_features))
            )
        )
        self.qt = QuantileTransformer()
        X_train[:, self.categorical_features] = self.qt.fit_transform(X_train[:, self.categorical_features])
        X_test[:, self.categorical_features] = self.qt.transform(X_test[:, self.categorical_features])

        return X_train, X_test, y_train, y_test
