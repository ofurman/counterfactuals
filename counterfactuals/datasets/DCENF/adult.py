from typing import Union
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import torch
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from counterfactuals.datasets.base import AbstractDataset


# adult
# Total 48842
# Train 32561
# Test 16281
# Num 4
# Cat 8
# Numeric ['age', 'capital.gain', 'capital.loss', 'hours.per.week']
# Categorical ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
# Classification Target ['income']


class CustomCategoricalTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, categorical_mapping, y=None, ):
        self.categorical_mapping = categorical_mapping
        return self

    def transform(self, X):
        return X.apply(lambda x: self.categorical_mapping[x])

class AdultDataset(AbstractDataset):
    alpha = 1e-6

    def __init__(
        self, 
        train_file_path: str = "data/adult/train.csv",
        test_file_path: str = "data/adult/test.csv"
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
            "age",
            "hours_per_week",
            # Categorical
            "workclass",
            "education",
            "marital_status",
            "occupation",
            "race",
            "gender",
        ]
        self.numerical_columns = list(range(0, 2))
        self.categorical_columns = list(range(2, len(self.feature_columns)))
        target_column = "income"

        # Drop any rows with missing values
        raw_data = raw_data.dropna(subset=self.feature_columns)

        # Convert target to binary
        raw_data[target_column] = (raw_data[target_column] == ">50K").astype(int)

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
                    OneHotEncoder(sparse=False),
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

        return X_train, X_test, y_train, y_test
