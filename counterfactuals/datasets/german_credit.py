import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder

from counterfactuals.datasets.base import AbstractDataset


class GermanCreditDataset(AbstractDataset):
    def __init__(self, file_path: str = "data/german_credit.csv", method=None, n_bins=None, train=False):
        self.raw_data = self.load(file_path=file_path, index_col=False)
        if method == "ares":
            self.categorical_features = [
                        "account_check_status", "credit_history", "purpose",
                        "savings", "present_emp_since", "personal_status_sex",
                        "other_debtors", "property", "other_installment_plans",
                        "housing", "job", "telephone", "foreign_worker",
                        # new
                        "installment_as_income_perc", "other_installment_plans",
                        "credits_this_bank", "people_under_maintenance", "present_res_since"]
            self.n_bins = n_bins
            self.X, self.y = self.one_hot(self.raw_data)
            if train:
                self.X, self.y = self.X.to_numpy().astype(np.float32), self.y.to_numpy()
            self.X_train, self.X_test, self.y_train, self.y_test = self.get_split_data(
                self.X, self.y
            )
        else:
            self.X, self.y = self.preprocess(raw_data=self.raw_data)
            self.X_train, self.X_test, self.y_train, self.y_test = self.get_split_data(
                self.X, self.y
            )
            self.X_train, self.X_test, self.y_train, self.y_test = self.transform(
                self.X_train, self.X_test, self.y_train, self.y_test
            )

    def preprocess(self, raw_data: pd.DataFrame):
        """
        Preprocess the loaded data to X and y numpy arrays.
        """

        self.feature_columns = [
            # Continuous
            "duration_in_month",
            "credit_amount",
            "installment_as_income_perc",
            "present_res_since",
            "age",
            "credits_this_bank",
            "people_under_maintenance",
            # Categorical
            "account_check_status",
            "credit_history",
            "purpose",
            "savings",
            "present_emp_since",
            "personal_status_sex",
            "other_debtors",
            "property",
            "other_installment_plans",
            "housing",
            "job",
            # "telephone",
            # "foreign_worker",
        ]
        self.numerical_columns = list(range(0, 7))
        self.categorical_columns = list(range(7, len(self.feature_columns)))
        target_column = "default"

        # Downsample to minor class
        raw_data = raw_data.dropna(subset=self.feature_columns)
        row_per_class = sum(raw_data[target_column] == 1)
        raw_data = pd.concat(
            [
                raw_data[raw_data[target_column] == 0].sample(
                    row_per_class, random_state=42
                ),
                raw_data[raw_data[target_column] == 1],
            ]
        )

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
        # self.feature_transformer.set_output(transform='pandas')

        X_train = self.feature_transformer.fit_transform(X_train)
        X_test = self.feature_transformer.transform(X_test)

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

    def one_hot(self, data):
        """
        Improvised method for one-hot encoding the data
        
        Input: data (whole dataset)
        Outputs: data_oh (one-hot encoded data)
                 features (list of feature values after one-hot encoding)
        """

        label_encoder = LabelEncoder()
        data_encode = data.copy()
        self.bins = {}
        self.bins_tree = {}
        self.features_tree = {}

        # Assign encoded features to one hot columns
        data_oh, features = [], []
        for x in data.columns[1:]:
            self.features_tree[x] = []
            categorical = x in self.categorical_features
            if categorical:
                data_encode[x] = label_encoder.fit_transform(data_encode[x])
                cols = label_encoder.classes_
            elif self.n_bins is not None:
                data_encode[x] = pd.cut(data_encode[x].apply(lambda x: float(x)),
                                        bins=self.n_bins)
                cols = data_encode[x].cat.categories
                self.bins_tree[x] = {}
            else:
                data_oh.append(data[x])
                features.append(x)
                continue
                
            one_hot = pd.get_dummies(data_encode[x])
            data_oh.append(one_hot)
            for col in cols:
                feature_value = x + " = " + str(col)
                features.append(feature_value)
                self.features_tree[x].append(feature_value)
                if not categorical:
                    self.bins[feature_value] = col.mid
                    self.bins_tree[x][feature_value] = col.mid
                
        data_oh = pd.concat(data_oh, axis=1, ignore_index=True)
        data_oh.columns = features
        self.features = features
        return data_oh, data["default"]
