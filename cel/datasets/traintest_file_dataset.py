from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from cel.datasets.base import DatasetBase
from cel.datasets.initial_transforms import (
    InitialTransformContext,
    InitialTransformPipeline,
    build_initial_transform_pipeline,
)


class TrainTestFileDataset(DatasetBase):
    """Dataset loader for pre-split train and test files.

    Unlike FileDataset which loads a single file and splits it internally,
    this class takes separate paths for train and test data, allowing users
    to provide their own pre-defined splits.
    """

    def __init__(
        self,
        config_path: Path,
        train_data_path: str,
        test_data_path: str,
        samples_keep: Optional[int] = None,
    ):
        """Initializes the dataset with separate train and test files.

        Args:
            config_path: Path to the dataset configuration file (defines features,
                target, feature_config, etc.). The raw_data_path in config is ignored.
            train_data_path: Path to the training data CSV file.
            test_data_path: Path to the test data CSV file.
            samples_keep: Optional limit on number of samples to keep from each file.
        """
        super().__init__(config_path=config_path)
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.samples_keep = samples_keep if samples_keep is not None else self.config.samples_keep
        self.initial_transform_pipeline: Optional[InitialTransformPipeline] = (
            build_initial_transform_pipeline(self.config.initial_transforms)
        )
        self.one_hot_feature_groups: dict[str, list[str]] = {}

        train_raw = self._load_csv(self.train_data_path)
        test_raw = self._load_csv(self.test_data_path)

        train_context = self._apply_initial_transforms(train_raw)
        test_context = self._apply_initial_transforms(test_raw)

        if self.samples_keep > 0 and len(train_context.data) > self.samples_keep:
            train_context.data = train_context.data.sample(
                self.samples_keep, random_state=42
            ).reset_index(drop=True)

        if self.samples_keep > 0 and len(test_context.data) > self.samples_keep:
            test_context.data = test_context.data.sample(
                self.samples_keep, random_state=42
            ).reset_index(drop=True)

        self.raw_train_data = train_context.data
        self.raw_test_data = test_context.data
        self._update_metadata_from_context(train_context)

        self.X_train, self.y_train = self._preprocess_split(self.raw_train_data)
        self.X_test, self.y_test = self._preprocess_split(self.raw_test_data)

        self.X = np.vstack([self.X_train, self.X_test])
        self.y = np.concatenate([self.y_train, self.y_test])

    def _preprocess_split(self, raw_data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Preprocesses raw data into feature and target arrays.

        Args:
            raw_data: Raw dataset as a pandas DataFrame.

        Returns:
            Tuple (X, y) as numpy arrays.
        """
        data = raw_data.copy()
        if self.config.target_mapping:
            data[self.config.target] = data[self.config.target].replace(self.config.target_mapping)

        X = data[self.features].to_numpy()
        y = data[self.config.target].to_numpy()
        return X, y

    def _apply_initial_transforms(self, raw_data: pd.DataFrame) -> InitialTransformContext:
        """Apply configured initial transforms to the raw dataframe."""
        context = InitialTransformContext(
            data=raw_data.copy(),
            features=list(self.config.features),
            continuous_features=list(self.config.continuous_features),
            categorical_features=list(self.config.categorical_features),
            feature_config=dict(self.config.feature_config),
            target=self.config.target,
            task_type=self.task_type,
        )

        if self.initial_transform_pipeline is None:
            return context
        return self.initial_transform_pipeline.fit_transform(context)

    def _update_metadata_from_context(self, context: InitialTransformContext) -> None:
        """Update dataset metadata after applying initial transforms."""
        self.config.features = list(context.features)
        self.config.continuous_features = list(context.continuous_features)
        self.config.categorical_features = list(context.categorical_features)
        self.config.feature_config = context.feature_config

        self.features = list(context.features)
        self.numerical_features = list(context.continuous_features)
        self.categorical_features = list(context.categorical_features)
        self.numerical_features_indices = [self.features.index(f) for f in self.numerical_features]
        self.categorical_features_indices = [
            self.features.index(f) for f in self.categorical_features
        ]
        self.target_index = len(self.features)
        self.actionable_features = [
            feature
            for feature, params in context.feature_config.items()
            if params.actionable and feature in self.features
        ]
        self.one_hot_feature_groups = context.one_hot_feature_groups

    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_ratio: float = 0.8,
        stratify: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns pre-loaded train/test splits (ignores X, y arguments).

        This method overrides the base class to return the pre-loaded splits
        from separate train and test files instead of splitting X and y.
        """
        return self.X_train, self.X_test, self.y_train, self.y_test
