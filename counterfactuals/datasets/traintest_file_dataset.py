from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from counterfactuals.datasets.base import DatasetBase
from counterfactuals.datasets.initial_transforms import (
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
        val_data_path: Optional[str] = None,
    ):
        """Initializes the dataset with separate train and test files.

        Args:
            config_path: Path to the dataset configuration file (defines features,
                target, feature_config, etc.). The raw_data_path in config is ignored.
            train_data_path: Path to the training data CSV file.
            test_data_path: Path to the test data CSV file.
            samples_keep: Optional limit on number of samples to keep from each file.
            val_data_path: Optional path to a validation CSV. When given (and the
                file exists) it is loaded through the same train-fitted transforms
                and exposed as `X_val`/`y_val` for early stopping. A path that does
                not exist is treated as absent, so callers can point at a
                conventional location without checking first.
        """
        super().__init__(config_path=config_path)
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.val_data_path = val_data_path
        self.samples_keep = samples_keep if samples_keep is not None else self.config.samples_keep
        self.initial_transform_pipeline: Optional[InitialTransformPipeline] = (
            build_initial_transform_pipeline(self.config.initial_transforms)
        )
        self.one_hot_feature_groups: dict[str, list[str]] = {}

        train_raw = self._load_csv(self.train_data_path)
        test_raw = self._load_csv(self.test_data_path)

        train_context, test_context = self._apply_initial_transforms_paired(train_raw, test_raw)

        # The validation split has to be transformed here, while the config still
        # holds the pre-one-hot feature names that the transforms expect;
        # `_update_metadata_from_context` below replaces them with the expanded
        # column names.
        val_context = None
        if val_data_path is not None and Path(val_data_path).exists():
            val_context = self._build_transform_context(self._load_csv(val_data_path))
            if self.initial_transform_pipeline is not None:
                val_context = self.initial_transform_pipeline.transform(val_context)
            val_context = self._align_context_to_train(val_context, train_context)

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

        self.raw_val_data: Optional[pd.DataFrame] = None
        self.X_val: Optional[np.ndarray] = None
        self.y_val: Optional[np.ndarray] = None
        if val_context is not None:
            self.raw_val_data = val_context.data
            self.X_val, self.y_val = self._preprocess_split(self.raw_val_data)

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

    def _build_transform_context(self, raw_data: pd.DataFrame) -> InitialTransformContext:
        return InitialTransformContext(
            data=raw_data.copy(),
            features=list(self.config.features),
            continuous_features=list(self.config.continuous_features),
            categorical_features=list(self.config.categorical_features),
            feature_config=dict(self.config.feature_config),
            target=self.config.target,
            task_type=self.task_type,
        )

    def _apply_initial_transforms_paired(
        self, train_raw: pd.DataFrame, test_raw: pd.DataFrame
    ) -> tuple[InitialTransformContext, InitialTransformContext]:
        """Fit transforms on train, then apply to both splits so columns align."""
        train_ctx = self._build_transform_context(train_raw)
        test_ctx = self._build_transform_context(test_raw)

        if self.initial_transform_pipeline is None:
            return train_ctx, test_ctx

        train_ctx = self.initial_transform_pipeline.fit_transform(train_ctx)
        test_ctx = self.initial_transform_pipeline.transform(test_ctx)

        return train_ctx, self._align_context_to_train(test_ctx, train_ctx)

    def _align_context_to_train(
        self,
        context: InitialTransformContext,
        train_ctx: InitialTransformContext,
    ) -> InitialTransformContext:
        """Give a transformed split the train split's exact column set and metadata.

        One-hot columns absent from the split are filled with zeros so every split
        lands in the same feature space regardless of which categories it happens
        to contain.
        """
        train_cols = list(train_ctx.data.columns)
        for col in train_cols:
            if col not in context.data.columns:
                context.data[col] = 0
        context.data = context.data[train_cols]
        context.features = list(train_ctx.features)
        context.continuous_features = list(train_ctx.continuous_features)
        context.categorical_features = list(train_ctx.categorical_features)
        context.feature_config = dict(train_ctx.feature_config)
        context.one_hot_feature_groups = dict(train_ctx.one_hot_feature_groups)
        return context

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
        self.monotonic_features = {
            feature: params.direction
            for feature, params in context.feature_config.items()
            if params.direction is not None and feature in self.features
        }
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
