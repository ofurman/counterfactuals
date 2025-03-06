import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler

from counterfactuals.datasets.base import AbstractDataset


class GiveMeSomeCreditDataset(AbstractDataset):
    def __init__(self, train_file: str = "data/GiveMeSomeCredit-training.csv",
                 test_file: str = "data/GiveMeSomeCredit-testing.csv"):
        self.raw_train_data = self.load(file_path=train_file, index_col=False)
        self.raw_test_data = self.load(file_path=test_file, index_col=False)

        self.X_train, self.y_train = self.preprocess(raw_data=self.raw_train_data)
        self.X_test, self.y_test = self.preprocess(raw_data=self.raw_test_data)

        self.X_train, self.X_test, self.y_train, self.y_test = self.transform(
            self.X_train, self.X_test, self.y_train, self.y_test
        )

    def preprocess(self, raw_data: pd.DataFrame):
        """
        Preprocess the loaded data to X and y numpy arrays.
        """
        self.feature_columns = [
            "RevolvingUtilizationOfUnsecuredLines",
            "age",
            "NumberOfTime30-59DaysPastDueNotWorse",
            "DebtRatio",
            "MonthlyIncome",
            "NumberOfOpenCreditLinesAndLoans",
            "NumberOfTimes90DaysLate",
            "NumberRealEstateLoansOrLines",
            "NumberOfTime60-89DaysPastDueNotWorse",
            "NumberOfDependents",
        ]
        self.numerical_columns = list(range(len(self.feature_columns)))
        target_column = "SeriousDlqin2yrs"

        # Drop NaN values (especially for MonthlyIncome & NumberOfDependents)
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
            ]
        )

        X_train = self.feature_transformer.fit_transform(X_train)
        X_test = self.feature_transformer.transform(X_test)

        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
        y_train = y_train.astype(np.int64)
        y_test = y_test.astype(np.int64)

        self.numerical_features = list(range(0, len(self.numerical_columns)))
        self.actionable_features = list(range(0, X_train.shape[1]))  # All features are actionable

        return X_train, X_test, y_train, y_test
