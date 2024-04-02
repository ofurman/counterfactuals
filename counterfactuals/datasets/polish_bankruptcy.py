import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer

from counterfactuals.datasets.base import AbstractDataset


class PolishBankDataset(AbstractDataset):
    def __init__(
        self,
        file_path: str = "data/polish_bankruptcy.csv",
        preprocess: bool = True,
        dataset_part: int = None,
    ):
        super().__init__(data=None)
        if dataset_part is not None:
            file_path = f"{file_path.removesuffix('.csv')}{dataset_part}.csv"
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
            self.data = pd.read_csv(file_path, index_col=False)
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

        # self.data = self.data.replace("?", np.nan)
        # self.data = self.data.loc[:, self.data.isnull().mean() < 0.01].dropna()

        # corr_matrix = self.data.corr().abs()
        # upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        # to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
        # self.data.drop(to_drop, axis=1, inplace=True)

        self.imputer = SimpleImputer(
            strategy="constant", missing_values=np.nan, fill_value=0
        )

        target_column = self.data.columns[-1]
        self.feature_columns = list(self.data.columns[:-1])
        self.numerical_columns = list(range(0, len(self.feature_columns)))
        self.categorical_columns = []

        self.data.replace("?", np.nan, inplace=True)
        self.data = pd.DataFrame(
            self.imputer.fit_transform(self.data), columns=self.data.columns
        )
        # upsample data

        row_per_class = sum(self.data[target_column] == 1)
        self.data = pd.concat(
            [
                self.data[self.data[target_column] == 0].sample(
                    row_per_class, random_state=42
                ),
                self.data[self.data[target_column] == 1],
            ]
        )
        X = self.data[self.feature_columns]
        y = self.data[target_column]
        X_train, X_test, y_train, y_test = train_test_split(
            X.to_numpy(),
            y.to_numpy(),
            random_state=4,
            test_size=0.1,
            shuffle=True,
            stratify=y,
        )

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

        self.categorical_features = []
        self.numerical_features = list(range(0, len(self.feature_columns)))

    def get_split_data(self) -> list:
        return self.X_train, self.X_test, self.y_train, self.y_test
