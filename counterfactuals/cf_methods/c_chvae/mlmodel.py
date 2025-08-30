import warnings
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd
import torch

from .data import Data
from .steps import order_data


class MLModel(ABC):
    """
    Abstract class to implement custom black-box-model for a given dataset with encoding and scaling processing.

    Parameters
    ----------
    data: Data
        Dataset inherited from Data-wrapper

    Methods
    -------
    predict:
        One-dimensional prediction of ml model for an output interval of [0, 1].
    predict_proba:
        Two-dimensional probability prediction of ml model.

    Returns
    -------
    None
    """

    def __init__(
        self,
        data: Data,
    ) -> None:
        self._data: Data = data

    @property
    def data(self) -> Data:
        """
        Contains the data.api.Data dataset.

        Returns
        -------
        carla.data.Data
        """
        return self._data

    @data.setter
    def data(self, data: Data) -> None:
        self._data = data

    @property
    @abstractmethod
    def feature_input_order(self):
        """
        Saves the required order of features as list.

        Prevents confusion about correct order of input features in evaluation

        Returns
        -------
        list of str
        """
        pass

    @property
    @abstractmethod
    def backend(self):
        """
        Describes the type of backend which is used for the classifier.

        E.g., tensorflow, pytorch, sklearn, xgboost

        Returns
        -------
        str
        """
        pass

    @property
    @abstractmethod
    def raw_model(self):
        """
        Contains the raw ML model built on its framework

        Returns
        -------
        object
            Classifier, depending on used framework
        """
        pass

    @abstractmethod
    def predict(self, x: Union[np.ndarray, pd.DataFrame]):
        """
        One-dimensional prediction of ml model for an output interval of [0, 1].

        Shape of input dimension has to be always two-dimensional (e.g., (1, m), (n, m))

        Parameters
        ----------
        x : np.Array or pd.DataFrame
            Tabular data of shape N x M (N number of instances, M number of features)

        Returns
        -------
        iterable object
            Ml model prediction for interval [0, 1] with shape N x 1
        """
        pass

    @abstractmethod
    def predict_proba(self, x: Union[np.ndarray, pd.DataFrame]):
        """
        Two-dimensional probability prediction of ml model.

        Shape of input dimension has to be always two-dimensional (e.g., (1, m), (n, m))

        Parameters
        ----------
        x : np.Array or pd.DataFrame
            Tabular data of shape N x M (N number of instances, M number of features)

        Returns
        -------
        iterable object
            Ml model prediction with shape N x 2
        """
        pass

    def get_ordered_features(self, x):
        """
        Restores the correct input feature order for the ML model, this also drops the columns not in the
        feature order. So it drops the target column, and possibly other features, e.g. categorical.

        Only works for encoded data

        Parameters
        ----------
        x : pd.DataFrame
            Data we want to order

        Returns
        -------
        output : pd.DataFrame
            Whole DataFrame with ordered feature
        """
        if isinstance(x, pd.DataFrame):
            return order_data(self.feature_input_order, x)
        else:
            warnings.warn(
                f"cannot re-order features for non dataframe input: {type(x)}"
            )
            return x

    def get_mutable_mask(self):
        """
        Get mask of mutable features.

        For example with mutable feature "income" and immutable features "age", the
        mask would be [True, False] for feature_input_order ["income", "age"].

        This mask can then be used to index data to only get the columns that are (im)mutable.

        Returns
        -------
        mutable_mask: np.array(bool)
        """
        # get categorical features
        categorical = self.data.categorical
        # get the binary encoded categorical features
        encoded_categorical = self.data.encoder.get_feature_names(categorical)
        # get the immutables, where the categorical features are in encoded format
        immutable = [
            encoded_categorical[categorical.index(i)] if i in categorical else i
            for i in self.data.immutables
        ]
        # find the index of the immutables in the feature input order
        immutable = [self.feature_input_order.index(col) for col in immutable]
        # make a mask
        mutable_mask = np.ones(len(self.feature_input_order), dtype=bool)
        # set the immutables to False
        mutable_mask[immutable] = False
        return mutable_mask


class CustomMLModel(MLModel):
    """
    Custom implementation of MLModel for use with CCHVAE
    """

    def __init__(self, model, data: Data):
        """
        Initialize with a trained PyTorch model and dataset

        Parameters
        ----------
        model: torch.nn.Module
            A trained PyTorch model
        data: Data
            A Data object
        """
        super().__init__(data)
        self._model = model
        self._feature_input_order = [
            str(i) for i in range(len(data.categorical) + len(data.continuous))
        ]

    @property
    def feature_input_order(self):
        """Required order of features"""
        return self._feature_input_order

    @property
    def backend(self):
        """Type of backend used"""
        return "pytorch"

    @property
    def raw_model(self):
        """The raw ML model"""
        return self._model

    def predict(self, x: Union[np.ndarray, pd.DataFrame, torch.Tensor]):
        """One-dimensional prediction"""
        with torch.no_grad():
            if isinstance(x, pd.DataFrame):
                x = x[self.feature_input_order].values

            if isinstance(x, np.ndarray):
                x = torch.tensor(x, dtype=torch.float32)

            return self._model.predict(x)

    def predict_proba(self, x: Union[np.ndarray, pd.DataFrame, torch.Tensor]):
        """Two-dimensional probability prediction"""
        with torch.no_grad():
            if isinstance(x, pd.DataFrame):
                x = x[self.feature_input_order].values

            if isinstance(x, np.ndarray):
                x = torch.tensor(x, dtype=torch.float32)

            return self._model.predict_proba(x)

    def get_mutable_mask(self):
        """
        Get mask of mutable features based on actionable features.

        For example with actionable features [0, 2] in a 4-feature dataset,
        the mask would be [True, False, True, False] for feature_input_order ["0", "1", "2", "3"].

        This mask can then be used to index data to only get the columns that are mutable (actionable).

        Returns
        -------
        mutable_mask: np.array(bool)
        """
        # Check if data has actionable_features attribute
        if hasattr(self.data, "actionable_features"):
            # Get actionable features as strings (to match feature_input_order format)
            actionable_features = self.data.actionable_features
        else:
            # Fall back to default behavior (use immutable features)
            return super().get_mutable_mask()

        # Create mask with all features initially set to False (immutable)
        mutable_mask = np.zeros(len(self.feature_input_order), dtype=bool)

        # Set actionable features to True (mutable)
        for feature in actionable_features:
            try:
                feature_idx = self.feature_input_order.index(feature)
                mutable_mask[feature_idx] = True
            except ValueError:
                # Feature not found in feature_input_order, skip it
                continue

        return mutable_mask
