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


class FileDataset(DatasetBase):
    """File dataset loader compatible with DatasetBase."""

    def __init__(
        self,
        config_path: Path,
        samples_keep: Optional[int] = None,
    ):
        """Initializes the File dataset with OmegaConf config.
        Args:
            config_path: Path to the dataset configuration file.
            dataset_name: Optional name for the dataset (used for model paths).
        """
        super().__init__(config_path=config_path)
        self.samples_keep = samples_keep if samples_keep is not None else self.config.samples_keep
        self.initial_transform_pipeline: Optional[InitialTransformPipeline] = (
            build_initial_transform_pipeline(self.config.initial_transforms)
        )
        self.one_hot_feature_groups: dict[str, list[str]] = {}

        raw_data = self._load_csv(self.config.raw_data_path)
        context = self._apply_initial_transforms(raw_data)

        if self.samples_keep > 0 and len(context.data) > self.samples_keep:
            context.data = context.data.sample(self.samples_keep, random_state=42).reset_index(
                drop=True
            )

        self.raw_data = context.data
        self._update_metadata_from_context(context)
        self.X, self.y = self.preprocess(self.raw_data)

    def _load_csv(self, file_path: str) -> pd.DataFrame:
        """Load dataset from CSV file.

        Args:
            file_path: Path to the CSV file (relative to project root).

        Returns:
            Loaded dataset as a pandas DataFrame.
        """
        # Resolve path relative to project root
        project_root = Path(__file__).resolve().parent.parent.parent
        path = project_root / file_path

        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        return pd.read_csv(path, index_col=False)

    def preprocess(self, raw_data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
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
        self.X, self.y = X, y
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
