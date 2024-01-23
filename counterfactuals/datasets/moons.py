import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from counterfactuals.datasets.base import AbstractDataset


class MoonsDataset(AbstractDataset):
    def __init__(self, file_path: str = "data/moons.csv", preprocess: bool = True):
        # TODO: make from_filepath class method
        self.load(file_path=file_path)
        if preprocess:
            self.preprocess()

    def load(self, file_path):
        """
        Load data from a CSV file and store it in the 'data' attribute.
        """
        try:
            self.data = pd.read_csv(file_path, header=None)
        except Exception as e:
            raise FileNotFoundError(f"Error loading data from {file_path}: {e}")

    def save(self, file_path):
        """
        Save the processed data (including scaled features) to a CSV file.
        """
        if self.data is not None:
            try:
                self.data.to_csv(file_path, index=False, header=None)
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

        X = self.data[self.data.columns[:-1]]
        y = self.data[self.data.columns[-1]]

        self.numerical_features = [0, 1]
        self.categorical_features = []
        self.actionable_features = [0, 1]

        X_train, X_test, y_train, y_test = train_test_split(
            X.to_numpy(), y.to_numpy(), random_state=4, test_size=0.1, shuffle=True, stratify=y
        )

        self.X_train = X_train
        self.X_test = X_test

        self.y_train = y_train.reshape(-1, 1)
        self.y_test = y_test.reshape(-1, 1)

        self.X_train = self.X_train.astype(np.float32)
        self.X_test = self.X_test.astype(np.float32)
        self.y_train = self.y_train.astype(np.float32)
        self.y_test = self.y_test.astype(np.float32)

    @property
    def categorical_features_lists(self) -> list:
        return []

    def get_split_data(self) -> list:
        return self.X_train, self.X_test, self.y_train, self.y_test
