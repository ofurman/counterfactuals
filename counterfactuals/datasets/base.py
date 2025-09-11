from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Union, Optional, Tuple, Generator
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold


class MonotonicityDirection(Enum):
    """Enum representing monotonicity direction of a feature."""
    INCREASE = "INCREASE"
    DECREASE = "DECREASE"


@dataclass
class FeatureParameters:
    """Configuration parameters for a single feature.

    Attributes:
        actionable: Whether the feature can be changed in counterfactuals.
        top_limit: Upper bound for the feature value.
        bottom_limit: Lower bound for the feature value.
        direction: Monotonicity direction, if applicable.
    """
    actionable: bool
    top_limit: Optional[float] = None
    bottom_limit: Optional[float] = None
    direction: Optional[MonotonicityDirection] = None


@dataclass
class DatasetParameters:
    """Configuration parameters for a dataset.

    Attributes:
        raw_data_path: Path to the raw dataset file.
        features: List of feature names or indices used as input variables.
        continuous_features: List of continuous feature names or indices.
        categorical_features: List of categorical feature names or indices.
        feature_config: Mapping of feature name/index to FeatureParameters.
        target: Name of the target column.
    """
    raw_data_path: str
    features: List[Union[str, int]]
    continuous_features: List[Union[str, int]]
    categorical_features: List[Union[str, int]]
    feature_config: Dict[Union[str, int], FeatureParameters]
    target: str = "y"


class DatasetBase:
    """Base class for datasets.

    Provides functionality for preprocessing, splitting, and cross-validation.
    Transformation logic should be handled outside this class.
    """

    def __init__(self, config: DatasetParameters):
        """Initializes a dataset with configuration parameters.

        Args:
            config: Dataset configuration object.
        """
        self.config = config
        self.X: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None

    def preprocess(self, raw_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Extracts input features and target from raw data.

        Args:
            raw_data: Raw dataset as a pandas DataFrame.

        Returns:
            A tuple of (X, y) numpy arrays.
        """
        X = raw_data[self.config.features].to_numpy()
        y = raw_data[self.config.target].to_numpy()
        self.X, self.y = X, y
        return X, y

    def split_data(
        self, X: np.ndarray, y: np.ndarray, train_ratio: float = 0.8, stratify: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Splits data into train and test sets.

        Args:
            X: Input features as numpy array.
            y: Target labels as numpy array.
            train_ratio: Proportion of data to include in the train split.
            stratify: Whether to stratify the split by target variable.

        Returns:
            A tuple (X_train, X_test, y_train, y_test).
        """
        return train_test_split(
            X,
            y,
            train_size=train_ratio,
            stratify=y if stratify else None,
            random_state=42,
        )

    def get_cv_splits(
        self, n_splits: int = 5, shuffle: bool = True
    ) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], None, None]:
        """Generates stratified cross-validation splits.

        Args:
            n_splits: Number of folds.
            shuffle: Whether to shuffle data before splitting.

        Yields:
            Tuples of (X_train, X_test, y_train, y_test) for each fold.

        Raises:
            ValueError: If preprocess has not been called before.
        """
        if self.X is None or self.y is None:
            raise ValueError("Call preprocess() before generating CV splits.")
        cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=42)
        for train_idx, test_idx in cv.split(self.X, self.y):
            yield (
                self.X[train_idx],
                self.X[test_idx],
                self.y[train_idx],
                self.y[test_idx],
            )
            
