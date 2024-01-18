import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

from counterfactuals.datasets.base import AbstractDataset


class CompasDataset(AbstractDataset):
    def __init__(self, file_path: str = "data/compas_two_years.csv", preprocess: bool = True):
        super().__init__(data=None)

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

        feature_columns = [
            # Continuous
            'age', 'priors_count', 'days_b_screening_arrest',
            'length_of_stay', 'is_recid', 'is_violent_recid', 'two_year_recid',
            # Categorical
            'c_charge_degree', 'sex', 'race', 
        ]
        self.numerical_columns = list(range(0, 7))
        self.categorical_columns = list(range(7, len(feature_columns)))
        target_column = 'class'

        self.data['days_b_screening_arrest'] = np.abs(self.data['days_b_screening_arrest'])
        self.data['c_jail_out'] = pd.to_datetime(self.data['c_jail_out'])
        self.data['c_jail_in'] = pd.to_datetime(self.data['c_jail_in'])
        self.data['length_of_stay'] = np.abs((self.data['c_jail_out'] - self.data['c_jail_in']).dt.days)
        self.data['length_of_stay'].fillna(self.data['length_of_stay'].value_counts().index[0], inplace=True)
        self.data['days_b_screening_arrest'].fillna(self.data['days_b_screening_arrest'].value_counts().index[0], inplace=True)
        self.data['length_of_stay'] = self.data['length_of_stay'].astype(int)
        self.data['days_b_screening_arrest'] = self.data['days_b_screening_arrest'].astype(int)
        self.data = self.data[self.data['score_text'] != "Medium"]
        self.data["class"] = pd.get_dummies(self.data["score_text"])["High"].astype(int)
        self.data.drop(['c_jail_in', 'c_jail_out', 'score_text'], axis=1, inplace=True)
        

        # Downsample to minor class
        self.data = self.data.dropna(subset=feature_columns)
        row_per_class = sum(self.data[target_column] == 1)
        self.data = pd.concat([
            self.data[self.data[target_column] == 0].sample(row_per_class, random_state=42),
            self.data[self.data[target_column] == 1],
        ])

        X = self.data[feature_columns]
        y = self.data[target_column]


        X_train, X_test, y_train, y_test = train_test_split(X.to_numpy(), y.to_numpy(), random_state=4, test_size=0.3, shuffle=True)

        self.feature_transformer = ColumnTransformer(
            [
                ("MinMaxScaler", MinMaxScaler(), self.numerical_columns),
                ("OneHotEncoder", OneHotEncoder(drop='if_binary', sparse_output=False), self.categorical_columns)
            ],
        )
        # self.feature_transformer.set_output(transform='pandas')

        self.X_train = self.feature_transformer.fit_transform(X_train)
        self.X_test = self.feature_transformer.transform(X_test)

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
