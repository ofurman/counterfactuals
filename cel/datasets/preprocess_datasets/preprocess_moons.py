import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def preprocess_moons(raw_data: pd.DataFrame):
    """
    Preprocess the loaded data to X and y numpy arrays.
    """
    categorical_columns = []
    X = raw_data[raw_data.columns[:-1]].to_numpy()
    y = raw_data[raw_data.columns[-1]].to_numpy()

    numerical_features = [0, 1]
    numerical_columns = [0, 1]
    categorical_features = []
    actionable_features = [0, 1]
    return X, y


def transform_moons(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
):
    """
    Transform the loaded data by applying Min-Max scaling to the features.
    """
    feature_transformer = MinMaxScaler()
    X_train = feature_transformer.fit_transform(X_train)
    X_test = feature_transformer.transform(X_test)

    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = y_train.astype(np.int64)
    y_test = y_test.astype(np.int64)
    return X_train, X_test, y_train, y_test
