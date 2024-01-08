import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

from counterfactuals.datasets.base import AbstractDataset


class HelocDataset(AbstractDataset):
    def __init__(self, file_path: str, preprocess: bool = True):
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

        X = self.data[self.data.columns[1:]]
        y = self.data[self.data.columns[0]]
        X_train, X_test, y_train, y_test = train_test_split(X.to_numpy(), y.to_numpy(), random_state=4, train_size=0.9, shuffle=True)

        self.feature_transformer = MinMaxScaler()
        self.X_train = self.feature_transformer.fit_transform(X_train)
        self.X_test = self.feature_transformer.transform(X_test)

        self.target_transformer = LabelEncoder()
        self.y_train = self.target_transformer.fit_transform(y_train.reshape(-1, 1))
        self.y_test = self.target_transformer.transform(y_test.reshape(-1, 1))

        self.X_train = self.X_train.astype(np.float32)
        self.X_test = self.X_test.astype(np.float32)
        self.y_train = self.y_train.astype(np.float32)
        self.y_test = self.y_test.astype(np.float32)

    def get_split_data(self) -> list:
        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_dataloader(self, batch_size: int, shuffle: bool, **kwargs_dataloader):
        return DataLoader(
            TensorDataset(torch.from_numpy(self.X_train), torch.from_numpy(self.y_train)),
            batch_size=batch_size,
            shuffle=shuffle,
            **kwargs_dataloader,
        )

    def test_dataloader(self, batch_size: int, shuffle: bool, **kwargs_dataloader):
        return DataLoader(
            TensorDataset(torch.from_numpy(self.X_test), torch.from_numpy(self.y_test)),
            batch_size=batch_size,
            shuffle=shuffle,
            **kwargs_dataloader,
        )
