import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from counterfactuals.datasets.base import AbstractDataset


class SyntheticDataset(AbstractDataset):
    def __init__(self, n_samples: int = 500, random_seed: int = 42):
        self.n_samples = n_samples
        self.random_seed = random_seed
        self.raw_data = self.generate_data()
        self.X, self.y = self.preprocess(raw_data=self.raw_data)
        self.X_train, self.X_test, self.y_train, self.y_test = self.get_split_data(
            self.X, self.y
        )
        self.X_train, self.X_test, self.y_train, self.y_test = self.transform(
            self.X_train, self.X_test, self.y_train, self.y_test
        )

    import numpy as np
    import pandas as pd


    def generate_data(self):
        np.random.seed(self.random_seed)

        # Class 0, Category 0: N(0, 1)
        class_0_cat0_continuous = np.random.normal(loc=0, scale=1, size=(self.n_samples, 1))
        class_0_cat0_categorical = np.zeros((self.n_samples, 1))  # Categorical = 0
        class_0_cat0_class = np.zeros((self.n_samples, 1))  # Class = 0

        # Class 1, Category 0: N(1, 1)
        class_1_cat0_continuous = np.random.normal(loc=1, scale=1, size=(self.n_samples, 1))
        class_1_cat0_categorical = np.zeros((self.n_samples, 1))  # Categorical = 0
        class_1_cat0_class = np.ones((self.n_samples, 1))  # Class = 1

        # Class 1, Category 1: N(2, 1)
        class_1_cat1_continuous = np.random.normal(loc=2, scale=1, size=(self.n_samples, 1))
        class_1_cat1_categorical = np.ones((self.n_samples, 1))  # Categorical = 1
        class_1_cat1_class = np.zeros((self.n_samples, 1))  # Class = 0

        # Class 0, Category 1: N(-1, 1)
        class_0_cat1_continuous = np.random.normal(loc=-1, scale=1, size=(self.n_samples, 1))
        class_0_cat1_categorical = np.ones((self.n_samples, 1))  # Categorical = 1
        class_0_cat1_class = np.ones((self.n_samples, 1))   # Class = 1

        # Combine all data
        class_0_cat0 = np.hstack([class_0_cat0_categorical, class_0_cat0_continuous, class_0_cat0_class])
        class_1_cat0 = np.hstack([class_1_cat0_categorical, class_1_cat0_continuous, class_1_cat0_class])
        class_1_cat1 = np.hstack([class_1_cat1_categorical, class_1_cat1_continuous, class_1_cat1_class])
        class_0_cat1 = np.hstack([class_0_cat1_categorical, class_0_cat1_continuous, class_0_cat1_class])

        # Stack and shuffle the dataset
        data = np.vstack([class_0_cat0, class_1_cat0, class_1_cat1, class_0_cat1])
        np.random.shuffle(data)

        # Create a DataFrame for readability
        return pd.DataFrame(data, columns=["categorical", "continuous", "class"])

    def preprocess(self, raw_data: pd.DataFrame):
        """
        Preprocess the loaded data to X and y numpy arrays.
        """
        self.categorical_columns = [0]  # Column index for categorical features
        X = raw_data[["categorical", "continuous"]].to_numpy()
        y = raw_data["class"].to_numpy()

        self.numerical_features = [1]  # Continuous feature index
        self.categorical_features = [0]  # Categorical feature index
        self.actionable_features = [1]  # Only the continuous feature is actionable
        return X, y

    def transform(
            self,
            X_train: np.ndarray,
            X_test: np.ndarray,
            y_train: np.ndarray,
            y_test: np.ndarray,
    ):
        """
        Transform the loaded data by applying Min-Max scaling to the continuous feature.
        """
        self.feature_transformer = MinMaxScaler()

        # Only scale the continuous feature (index 1)
        X_train[:, 1:] = self.feature_transformer.fit_transform(X_train[:, 1:])
        X_test[:, 1:] = self.feature_transformer.transform(X_test[:, 1:])

        y_train = y_train.reshape(-1)
        y_test = y_test.reshape(-1)

        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
        y_train = y_train.astype(np.int64)
        y_test = y_test.astype(np.int64)
        return X_train, X_test, y_train, y_test
