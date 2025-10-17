from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

try:
    from omegaconf import DictConfig
except ImportError:  # pragma: no cover - optional dependency during testing
    DictConfig = None


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
    target: str
    target_mapping: Dict[str, int]
    samples_keep: int = -1


class DatasetBase:
    """Base class for datasets.

    Provides functionality for preprocessing, splitting, and cross-validation.
    Transformation logic should be handled outside this class.
    """

    def __init__(self, config_path: Path):
        """Initializes a dataset with configuration parameters from a YAML file.

        Args:
            config_path: Path to the YAML config file.
        """
        self.config = self._load_config(config_path)
        self.X: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        # Note: AReS/GLOBE_CE expect features to include target at the end
        self.features: List[Any] = self.config.features + [self.config.target]
        self.numerical_features: List[Any] = self.config.continuous_features
        self.numerical_features_indices: List[int] = [
            self.features.index(f) for f in self.numerical_features
        ]
        self.categorical_features: List[Any] = self.config.categorical_features
        self.categorical_features_indices: List[int] = [
            self.features.index(f) for f in self.categorical_features
        ]
        self.actionable_features: List[Any] = [
            k for k, v in self.config.feature_config.items() if v.actionable
        ]
        self.task_type: str = "classification"

        # Aliases for backward compatibility with AReS/GLOBE_CE
        self.numerical_columns: List[int] = self.numerical_features_indices
        self.categorical_columns: List[int] = self.categorical_features_indices

    @property
    def features_tree(self) -> Dict[Any, List[str]]:
        """Generate features_tree for compatibility with AReS/GLOBE_CE.

        For datasets without one-hot encoding, this returns a dictionary
        mapping each feature to an empty list (for continuous features)
        or to their categorical values (for categorical features).

        Returns:
            Dictionary mapping feature names to their one-hot encoded column names
            (empty list for continuous features).
        """
        if not hasattr(self, "_features_tree"):
            self._features_tree = {}
            # Only iterate over actual features, not the target
            for feature in self.config.features:
                # For now, assume no one-hot encoding in base datasets
                # Continuous features map to empty list
                # Categorical features would need one-hot encoding handled separately
                if feature in self.categorical_features:
                    # For categorical features without explicit one-hot encoding,
                    # return empty list (will be handled by preprocessing if needed)
                    self._features_tree[feature] = []
                else:
                    # Continuous features
                    self._features_tree[feature] = []
        return self._features_tree

    @features_tree.setter
    def features_tree(self, value: Dict[Any, List[str]]) -> None:
        """Allow setting features_tree explicitly (e.g., from pipelines)."""
        self._features_tree = value

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

    def _resolve_data_path(self, file_path: Union[str, Path]) -> Path:
        project_root = Path(__file__).resolve().parent.parent.parent
        return project_root / Path(file_path)

    def _load_csv(self, file_path: str, **read_csv_kwargs: Any) -> pd.DataFrame:
        path = self._resolve_data_path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        return pd.read_csv(path, **read_csv_kwargs)

    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_ratio: float = 0.8,
        stratify: bool = True,
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

        if self.task_type == "classification":
            cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=42)
            for train_idx, test_idx in cv.split(self.X, self.y):
                yield (
                    self.X[train_idx],
                    self.X[test_idx],
                    self.y[train_idx],
                    self.y[test_idx],
                )
        else:
            cv = KFold(n_splits=n_splits, shuffle=shuffle, random_state=42)
            for train_idx, test_idx in cv.split(self.X):
                yield (
                    self.X[train_idx],
                    self.X[test_idx],
                    self.y[train_idx],
                    self.y[test_idx],
                )

    def _load_config(self, yaml_path: Path) -> DatasetParameters:
        """Loads dataset parameters from YAML config.

        Args:
            yaml_path: Path to the YAML config file.

        Returns:
            DatasetParameters object containing the loaded configuration.

        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(yaml_path, "r") as f:
            cfg = yaml.safe_load(f)

        # Parse feature_config into FeatureParameters
        feature_config = {
            k: FeatureParameters(**v) for k, v in cfg.get("feature_config", {}).items()
        }

        return DatasetParameters(
            raw_data_path=cfg["raw_data_path"],
            features=cfg["features"],
            continuous_features=cfg.get("continuous_features", []),
            categorical_features=cfg.get("categorical_features", []),
            feature_config=feature_config,
            target=cfg.get("target", "y"),
            target_mapping=cfg.get("target_mapping", {}),
            samples_keep=cfg.get("samples_keep", 1000),
        )
