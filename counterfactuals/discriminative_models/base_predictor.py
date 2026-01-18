from abc import ABC, abstractmethod

import numpy as np


class BasePredictor(ABC):

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray) -> None:
        """
        Fit the classifier to the data.

        Args:
            X_train: The input training data. # shape: (n_samples, n_features)
            y_train: The training target labels. # shape: (n_samples,)
            X_val: The input validation data. # shape: (n_samples, n_features)
            y_val: The validation target labels. # shape: (n_samples,)
        """

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict the labels of the data.

        Args:
            x: The input data. # shape: (n_samples, n_features)

        Returns:
            The predicted labels. # shape: (n_samples,)
        """

    @abstractmethod
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """
        Predict the class probabilities of the data.

        Args:
            x: The input data. # shape: (n_samples, n_features)

        Returns:
            The predicted class probabilities. # shape: (n_samples, n_classes)
        """

    @abstractmethod
    def eval(self) -> None:
        """
        Set the classifier to evaluation mode.
        """

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the classifier to a file.
        """

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load the classifier from a file.
        """
