from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class Data(ABC):
    """
    Abstract class to implement arbitrary datasets, which are provided by the user. This is the general data object
    that is used in CARLA.
    """

    @property
    @abstractmethod
    def categorical(self):
        """
        Provides the column names of categorical data.
        Column names do not contain encoded information as provided by a get_dummy() method (e.g., sex_female)

        Label name is not included.

        Returns
        -------
        list of Strings
            List of all categorical columns
        """
        pass

    @property
    @abstractmethod
    def continuous(self):
        """
        Provides the column names of continuous data.

        Label name is not included.

        Returns
        -------
        list of Strings
            List of all continuous columns
        """
        pass

    @property
    @abstractmethod
    def immutables(self):
        """
        Provides the column names of immutable data.

        Label name is not included.

        Returns
        -------
        list of Strings
            List of all immutable columns
        """
        pass

    @property
    @abstractmethod
    def target(self):
        """
        Provides the name of the label column.

        Returns
        -------
        str
            Target label name
        """
        pass

    @property
    @abstractmethod
    def df(self):
        """
        The full Dataframe.

        Returns
        -------
        pd.DataFrame
        """
        pass

    @property
    @abstractmethod
    def df_train(self):
        """
        The training split Dataframe.

        Returns
        -------
        pd.DataFrame
        """
        pass

    @property
    @abstractmethod
    def df_test(self):
        """
        The testing split Dataframe.

        Returns
        -------
        pd.DataFrame
        """
        pass

    @abstractmethod
    def transform(self, df):
        """
        Data transformation, for example normalization of continuous features and encoding of categorical features.

        Parameters
        ----------
        df: pd.DataFrame

        Returns
        -------
        pd.Dataframe
        """
        pass

    @abstractmethod
    def inverse_transform(self, df):
        """
        Inverts transform operation.

        Parameters
        ----------
        df: pd.DataFrame

        Returns
        -------
        pd.Dataframe
        """
        pass


class CustomData(Data):
    """
    Custom implementation of Data class for use with CCHVAE
    """

    def __init__(self, dataset, target_column: str = "target") -> None:
        """
        Initialize with a dataset that has transformer and feature information
        """
        self._dataset = dataset
        self._categorical_columns = [str(i) for i in dataset.categorical_features]
        self._continuous_columns = [
            str(i)
            for i in range(dataset.X_train.shape[1])
            if i not in dataset.categorical_features
        ]
        self._df = pd.DataFrame(
            data=dataset.X_train,
            columns=[str(i) for i in range(dataset.X_train.shape[1])],
        )
        self._df_train = self._df.copy()

        self._df_test = pd.DataFrame(
            data=dataset.X_test,
            columns=[str(i) for i in range(dataset.X_test.shape[1])],
        )

        self._target_column = target_column
        self._df[self._target_column] = dataset.y_train
        self._df_train[self._target_column] = dataset.y_train
        self._df_test[self._target_column] = dataset.y_test

        self._scaler: Optional[MinMaxScaler] = None
        if self._continuous_columns:
            self._scaler = MinMaxScaler()
            self._scaler.fit(self._df_train[self._continuous_columns])
            self._apply_scaling()

        class Encoder:
            def get_feature_names(self, categorical):
                return [str(i) for i in dataset.categorical_features]

        self.encoder = Encoder()

    @property
    def categorical(self):
        """Column names of categorical features"""
        return self._categorical_columns

    @property
    def continuous(self):
        """Column names of continuous features"""
        return self._continuous_columns

    @property
    def immutables(self):
        """Column names of immutable features (example: demographic features)"""
        # This is application-specific - for demonstration we'll consider no features immutable
        return []

    @property
    def target(self):
        """Name of the target column"""
        return self._target_column

    @property
    def df(self):
        """Full dataframe"""
        return self._df

    @property
    def df_train(self):
        """Training dataframe"""
        return self._df_train

    @property
    def df_test(self):
        """Testing dataframe"""
        return self._df_test

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data (apply scaling/encoding)"""
        transformed = df.copy()
        if self._scaler and set(self._continuous_columns).issubset(transformed.columns):
            transformed.loc[:, self._continuous_columns] = self._scaler.transform(
                transformed[self._continuous_columns]
            )
        return transformed

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform (undo scaling/encoding)"""
        inversed = df.copy()
        if self._scaler and set(self._continuous_columns).issubset(inversed.columns):
            inversed.loc[:, self._continuous_columns] = self._scaler.inverse_transform(
                inversed[self._continuous_columns]
            )
        return inversed

    def _apply_scaling(self) -> None:
        """Apply min-max scaling to stored dataframes."""
        for df in (self._df, self._df_train, self._df_test):
            df.loc[:, self._continuous_columns] = self._scaler.transform(
                df[self._continuous_columns]
            )
