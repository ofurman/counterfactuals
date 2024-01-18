import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

from counterfactuals.datasets.base import AbstractDataset


class LawDataset(AbstractDataset):
    def __init__(self, file_path: str = "data/law.csv", preprocess: bool = True):
        super().__init__(data=None)
        self.scaler = MinMaxScaler()

        # TODO: make from_filepath class method
        self.load(file_path=file_path)
        if preprocess:
            self.preprocess()

    def load(self, file_path):
        """
        Load data from a CSV file and store it in the 'data' attribute.
        """
        try:
            self.data = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error loading data from {file_path}: {e}")

    def save(self, file_path):
        """
        Save the processed data (including scaled features) to a CSV file.
        """
        if self.data is not None:
            try:
                self.data.to_csv(file_path, index=False)
                print(f"Data saved to {file_path}")
            except Exception as e:
                print(f"Error saving data to {file_path}: {e}")
        else:
            print("No data to save.")

    def preprocess(self):
        """
        Preprocess the loaded data by applying Min-Max scaling to all columns except the last one (target column).
        """
        if not isinstance(self.data, pd.DataFrame):
            raise Exception("Data is empy. Nothing to preprocess!")

        feature_columns = ["lsat", "gpa", "zfygpa", "sex", "race"]
        target_column = "pass_bar"
        self.numerical_columns = list(range(0, 3))
        self.categorical_columns = list(range(3, len(feature_columns)))

        # Downsample to minor class
        self.data = self.data.dropna(subset=feature_columns)
        row_per_class = sum(self.data[target_column] == 0)
        self.data = pd.concat([
            self.data[self.data[target_column] == 0],
            self.data[self.data[target_column] == 1].sample(row_per_class, random_state=42),
        ])

        X = self.data[feature_columns]
        y = self.data[target_column]

        X_train, X_test, y_train, y_test = train_test_split(X.to_numpy(), y.to_numpy(), random_state=4, train_size=0.9, shuffle=True, stratify=y)

        self.feature_transformer = ColumnTransformer(
            [
                ("MinMaxScaler", MinMaxScaler(), self.numerical_columns),
                ("OneHotEncoder", OneHotEncoder(drop='if_binary', sparse_output=False), self.categorical_columns)
            ],
        )
        self.X_train = self.feature_transformer.fit_transform(X_train)
        self.X_test = self.feature_transformer.transform(X_test)

        # self.target_transformer = LabelEncoder()
        # self.y_train = self.target_transformer.fit_transform(y_train.reshape(-1, 1))
        # self.y_test = self.target_transformer.transform(y_test.reshape(-1, 1))
        self.y_train = y_train
        self.y_test = y_test

        self.X_train = self.X_train.astype(np.float32)
        self.X_test = self.X_test.astype(np.float32)
        self.y_train = self.y_train.astype(np.float32)
        self.y_test = self.y_test.astype(np.float32)

        self.numerical_features = list(range(0, len(self.numerical_columns)))
        self.categorical_features = list(range(len(self.numerical_columns), self.X_train.shape[1]))
        self.actionable_features = list(range(0, self.X_train.shape[1]))

    def get_split_data(self) -> list:
        return self.X_train, self.X_test, self.y_train, self.y_test
