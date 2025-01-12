from typing import Union

from sklearn.compose import ColumnTransformer
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from counterfactuals.datasets.base import AbstractDataset
from sklearn.base import BaseEstimator, TransformerMixin


class CustomCategoricalTransformer(BaseEstimator, TransformerMixin):
    def _dequantize(self, x, rng):
        """
        Adds noise to pixels to dequantize them.
        Ensures the output stays in the valid range [0, 1].
        """
        for i in range(x.shape[1]):
            data_with_noise = x[:, i] + rng.rand(*x[:, i].shape)
            divider = self.dividers[i]
            x[:, i] = data_with_noise / divider
        return x

    def _logit_transform(self, x):
        """
        Transforms pixel values with logit to be unconstrained.
        """
        x = CreditDefaultDataset.alpha + (1 - 2 * CreditDefaultDataset.alpha) * x
        return np.log(x / (1.0 - x))

    @staticmethod
    def inverse(x: Union[np.ndarray, torch.Tensor], dividers: list) -> np.ndarray:
        """
        Inverse transform the logit transformation.
        Handles both numpy arrays and torch tensors.
        """
        x = x.copy()
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = (torch.sigmoid(x) - 1e-6) / (1 - 2e-6)
        for i in range(x.shape[1]):
            bins = np.linspace(0, 1, dividers[i])
            x[:, i] = np.digitize(x[:, i], bins) - 1
        return x.numpy()

    def fit(self, X, y=None):
        self.dividers = [X[:, i].max() + 1 for i in range(X.shape[1])]
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed = self._dequantize(X_transformed, np.random.RandomState(42))
        X_transformed = self._logit_transform(X_transformed)
        return X_transformed

    def inverse_transform(self, X):
        """
        Inverse transform the logit transformation.
        Handles both numpy arrays and torch tensors.
        """
        x = X.copy()
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        x = (torch.sigmoid(torch.from_numpy(x)).numpy() - 1e-6) / (1 - 2e-6)
        for i in range(x.shape[1]):
            bins = np.linspace(0, 1, self.dividers[i] + 1)
            x[:, i] = np.digitize(x[:, i], bins) - 1
        return x


class CreditDefaultDataset(AbstractDataset):
    alpha = 1e-6

    def __init__(
        self, file_path: str = "data/credit_default.csv", transform=True, shuffle=True
    ):
        """
        Initialize the Credit Default dataset.
        """
        self.categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]
        self.features = [
            "LIMIT_BAL",
            "SEX",
            "EDUCATION",
            "MARRIAGE",
            "AGE",
            "PAY_0",
            "PAY_2",
            "PAY_3",
            "PAY_4",
            "PAY_5",
            "PAY_6",
            "BILL_AMT1",
            "BILL_AMT2",
            "BILL_AMT3",
            "BILL_AMT4",
            "BILL_AMT5",
            "BILL_AMT6",
            "PAY_AMT1",
            "PAY_AMT2",
            "PAY_AMT3",
            "PAY_AMT4",
            "PAY_AMT5",
            "PAY_AMT6",
            "default payment next month",
        ]
        self.raw_data = self.load(file_path=file_path, index_col=False)
        self.X, self.y = self.preprocess(raw_data=self.raw_data)
        self.X_train, self.X_test, self.y_train, self.y_test = self.get_split_data(
            self.X, self.y, shuffle=shuffle
        )
        if transform:
            self.X_train, self.X_test, self.y_train, self.y_test = self.transform(
                self.X_train, self.X_test, self.y_train, self.y_test
            )

    def load(self, file_path: str, index_col: bool = True):
        """
        Load the dataset from a CSV file.
        Original source: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
        """
        return pd.read_csv(file_path, index_col=index_col)

    @staticmethod
    def _dequantize(
        x: np.ndarray, categorical_cols: list, rng: np.random.Generator
    ) -> np.ndarray:
        """
        Adds noise to pixels to dequantize them.
        Ensures the output stays in the valid range [0, 1].
        """
        for col in categorical_cols:
            x[:, col] = (
                x[:, col] + rng.rand(*x[:, col].shape) / np.unique(x[:, col]).shape[0]
            )
        return x

    @staticmethod
    def _logit_transform(x: np.ndarray, categorical_cols: list) -> np.ndarray:
        """
        Transforms pixel values with logit to be unconstrained.
        """
        for col in categorical_cols:
            x[:, col] = (
                CreditDefaultDataset.alpha
                + (1 - 2 * CreditDefaultDataset.alpha) * x[:, col]
            )
            x[:, col] = np.log(x[:, col] / (1.0 - x[:, col]))
        return x

    def inverse_transform(self, x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Inverse transform the logit transformation.
        Handles both numpy arrays and torch tensors.
        """
        x = x.copy()
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        x = np.hstack(
            [
                self.feature_transformer["num0"].inverse_transform(
                    x[:, self.numerical_columns[0:1]]
                ),
                self.feature_transformer["cat"].inverse_transform(
                    x[:, self.categorical_columns]
                ),
                self.feature_transformer["num1"].inverse_transform(
                    x[:, self.numerical_columns[1:]]
                ),
            ]
        )
        return x

    def preprocess(self, raw_data: pd.DataFrame):
        """
        Preprocess the loaded data to X and y numpy arrays.
        PAY_0 to PAY_6 are the repayment status from April to September.
        -1 = pay duly;
        1 = payment delay for one month;
        2 = payment delay for two months;
        . . .;
        8 = payment delay for eight months;
        9 = payment delay for nine months and above.

        Features:
        0 - LIMIT_BAL: Amount of given credit.
        1 - SEX: Gender of the applicant.  (1 = male; 2 = female).
        2 - EDUCATION: Education level of the applicant. (1 = graduate school; 2 = university; 3 = high school; 4 = others).
        3 - MARRIAGE: Marital status of the applicant. (1 = married; 2 = single; 3 = others).
        4 - AGE: Age of the applicant.
        5 - PAY_0: Repayment status in September.
        6 - PAY_2: Repayment status in August.
        7 - PAY_3: Repayment status in July.
        8 - PAY_4: Repayment status in June.
        9 - PAY_5: Repayment status in May.
        10 - PAY_6: Repayment status in April.
        11 - BILL_AMT1: Bill statement amount in September.
        12 - BILL_AMT2: Bill statement amount in August.
        13 - BILL_AMT3: Bill statement amount in July.
        14 - BILL_AMT4: Bill statement amount in June.
        15 - BILL_AMT5: Bill statement amount in May.
        16 - BILL_AMT6: Bill statement amount in April.
        17 - PAY_AMT1: Previous payment amount in September.
        18 - PAY_AMT2: Previous payment amount in August.
        19 - PAY_AMT3: Previous payment amount in July.
        20 - PAY_AMT4: Previous payment amount in June.
        21 - PAY_AMT5: Previous payment amount in May.
        22 - PAY_AMT6: Previous payment amount in April.
        """
        self.feature_columns = [
            "LIMIT_BAL",
            "SEX",
            "EDUCATION",
            "MARRIAGE",
            "AGE",
            "PAY_0",
            "PAY_2",
            "PAY_3",
            "PAY_4",
            "PAY_5",
            "PAY_6",
            "BILL_AMT1",
            "BILL_AMT2",
            "BILL_AMT3",
            "BILL_AMT4",
            "BILL_AMT5",
            "BILL_AMT6",
            "PAY_AMT1",
            "PAY_AMT2",
            "PAY_AMT3",
            "PAY_AMT4",
            "PAY_AMT5",
            "PAY_AMT6",
        ]
        self.categorical_columns = [
            "SEX",
            "EDUCATION",
            "MARRIAGE",
            "PAY_0",
            "PAY_2",
            "PAY_3",
            "PAY_4",
            "PAY_5",
            "PAY_6",
        ]
        self.target_column = "default payment next month"
        raw_data = raw_data[raw_data["SEX"].isin([1, 2])]
        raw_data["SEX"] = raw_data["SEX"] - 1
        raw_data = raw_data[raw_data["MARRIAGE"].isin([1, 2])]
        raw_data["MARRIAGE"] = raw_data["MARRIAGE"] - 1
        raw_data = raw_data[raw_data["EDUCATION"].isin([1, 2, 3])]
        raw_data["EDUCATION"] = raw_data["EDUCATION"] - 1

        # Replace the PAY_i columns values -1, -2, 0 with 0
        pay_columns = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
        for col in pay_columns:
            raw_data[col] = raw_data[col].replace([-1, -2], 0)

        # # Downsample the majority class (0) to match the minority class (1)
        # count_class_0, count_class_1 = np.bincount(raw_data[self.target_column].values)
        # df_class_0 = raw_data[raw_data[self.target_column] == 0]
        # df_class_1 = raw_data[raw_data[self.target_column] == 1]
        # df_class_0_downsampled = df_class_0.sample(count_class_1, random_state=42)
        # raw_data = pd.concat([df_class_0_downsampled, df_class_1])
        raw_data = raw_data.dropna()
        X = raw_data[self.feature_columns].values
        y = raw_data[self.target_column].values
        # invert y to be 1 for non-default and 0 for default
        y = 1 - y

        self.numerical_columns = [0, 4, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
        self.categorical_columns = [
            1,
            2,
            3,
            5,
            6,
            7,
            8,
            9,
            10,
        ]
        self.numerical_features = [
            self.feature_columns[i] for i in self.numerical_columns
        ]
        self.categorical_features = [
            self.feature_columns[i] for i in self.categorical_columns
        ]

        # Actionable features are the features in august and september
        self.actionable_features = [5, 6, 11, 12, 17, 18]
        self.not_actionable_features = [
            i
            for i in range(len(self.feature_columns))
            if i not in self.actionable_features
        ]
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
            transformers=[
                ("num0", StandardScaler(), self.numerical_columns[0:1]),
                ("cat", CustomCategoricalTransformer(), self.categorical_columns),
                ("num1", StandardScaler(), self.numerical_columns[1:]),
            ],
            remainder="passthrough",
        )
        # self.feature_transformer = MinMaxScaler()
        X_train = self.feature_transformer.fit_transform(X_train.astype(np.float32))
        X_test = self.feature_transformer.transform(X_test.astype(np.float32))

        X_train = np.array(X_train.astype(np.float32))
        X_test = np.array(X_test.astype(np.float32))

        self.y_transformer = OneHotEncoder(sparse_output=False)
        y_train = self.y_transformer.fit_transform(y_train.reshape(-1, 1))
        y_test = self.y_transformer.transform(y_test.reshape(-1, 1))
        y_train = np.array(y_train.astype(np.int64))
        y_test = np.array(y_test.astype(np.int64))

        self.categorical_features = self.categorical_columns
        self.numerical_features = self.numerical_columns

        return X_train, X_test, y_train, y_test
