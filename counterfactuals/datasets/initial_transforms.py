"""Initial dataset transforms applied before train/test split."""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

import numpy as np
import pandas as pd

from counterfactuals.datasets.base import FeatureParameters


@dataclass
class InitialTransformContext:
    """Container holding dataset state for initial transforms."""

    data: pd.DataFrame
    features: List[str]
    continuous_features: List[str]
    categorical_features: List[str]
    feature_config: Dict[str, FeatureParameters]
    target: str
    task_type: str = "classification"
    one_hot_feature_groups: Dict[str, List[str]] = field(default_factory=dict)

    def copy(self) -> "InitialTransformContext":
        """Deep copy context to avoid shared state between steps."""
        return InitialTransformContext(
            data=self.data.copy(),
            features=list(self.features),
            continuous_features=list(self.continuous_features),
            categorical_features=list(self.categorical_features),
            feature_config=copy.deepcopy(self.feature_config),
            target=self.target,
            task_type=self.task_type,
            one_hot_feature_groups=copy.deepcopy(self.one_hot_feature_groups),
        )


class InitialTransformStep(ABC):
    """Interface for individual initial dataset transforms."""

    @abstractmethod
    def fit(self, context: InitialTransformContext) -> "InitialTransformStep":
        """Fit the transform based on full dataset."""

    @abstractmethod
    def transform(self, context: InitialTransformContext) -> InitialTransformContext:
        """Apply the transform and return updated context."""


class InitialTransformPipeline:
    """Chain multiple initial transforms sequentially."""

    def __init__(self, steps: List[Tuple[str, InitialTransformStep]]):
        self.steps = steps
        self._validate_steps()

    def _validate_steps(self) -> None:
        for name, step in self.steps:
            if not isinstance(step, InitialTransformStep):
                raise TypeError(f"Initial transform '{name}' must implement InitialTransformStep.")

    def fit(self, context: InitialTransformContext) -> "InitialTransformPipeline":
        current = context
        for _, step in self.steps:
            step.fit(current)
            current = step.transform(current)
        return self

    def transform(self, context: InitialTransformContext) -> InitialTransformContext:
        current = context
        for _, step in self.steps:
            current = step.transform(current)
        return current

    def fit_transform(self, context: InitialTransformContext) -> InitialTransformContext:
        current = context
        for _, step in self.steps:
            step.fit(current)
            current = step.transform(current)
        return current

    def __repr__(self) -> str:
        steps_repr = "\n  ".join([f"{name}: {type(step).__name__}" for name, step in self.steps])
        return f"InitialTransformPipeline(\n  {steps_repr}\n)"


class DropNaStep(InitialTransformStep):
    """Drop rows with missing values in features and target column."""

    def __init__(self, subset: Optional[Sequence[str]] = None):
        """Initialize DropNaStep.

        Args:
            subset: Optional list of column names to check for NaN values.
                   If None, will check all feature columns plus the target.
        """
        self.subset = list(subset) if subset is not None else None

    def fit(self, context: InitialTransformContext) -> "DropNaStep":
        return self

    def transform(self, context: InitialTransformContext) -> InitialTransformContext:
        # Determine which columns to check for NaN values
        subset = self.subset or context.features
        missing = [col for col in subset if col not in context.data.columns]
        if missing:
            raise ValueError(f"Columns not found for DropNaStep: {missing}")

        # Include target column in dropna to remove rows where label (y) contains NaN
        columns_to_check = list(subset)
        if context.target not in columns_to_check:
            columns_to_check.append(context.target)

        context.data = context.data.dropna(subset=columns_to_check)
        return context


class ReorderColumnsStep(InitialTransformStep):
    """Reorder feature columns while keeping target at the end."""

    def __init__(self, order: Optional[Sequence[str]] = None):
        self.order = list(order) if order is not None else None

    def fit(self, context: InitialTransformContext) -> "ReorderColumnsStep":
        return self

    def transform(self, context: InitialTransformContext) -> InitialTransformContext:
        order = self.order or (
            list(context.continuous_features) + list(context.categorical_features)
        )
        missing = [col for col in order if col not in context.data.columns]
        if missing:
            raise ValueError(f"Columns not found for ReorderColumnsStep: {missing}")

        if context.target not in context.data.columns:
            raise ValueError("Target column missing during column reordering.")

        context.data = context.data[order + [context.target]]
        context.features = order
        return context


class DownsampleStep(InitialTransformStep):
    """Balance classes by downsampling to the minority class size."""

    def __init__(self, target_column: Optional[str] = None, random_state: int = 42):
        self.target_column = target_column
        self.random_state = random_state

    def fit(self, context: InitialTransformContext) -> "DownsampleStep":
        return self

    def transform(self, context: InitialTransformContext) -> InitialTransformContext:
        if context.task_type != "classification":
            return context

        target = self.target_column or context.target
        if target not in context.data.columns:
            raise ValueError("Target column missing for downsampling.")

        counts = context.data[target].value_counts()
        if counts.empty:
            return context

        min_count = counts.min()
        balanced = []
        for class_value in counts.index:
            class_df = context.data[context.data[target] == class_value]
            if len(class_df) > min_count:
                class_df = class_df.sample(min_count, random_state=self.random_state)
            balanced.append(class_df)

        context.data = pd.concat(balanced, ignore_index=True)
        return context


class OneHotEncodingStep(InitialTransformStep):
    """One-hot encode categorical features before splitting."""

    def __init__(
        self,
        columns: Optional[Sequence[str]] = None,
        drop_first: bool = False,
        prefix_sep: str = "__",
        dtype: type = np.float64,
    ):
        self.columns = list(columns) if columns is not None else None
        self.drop_first = drop_first
        self.prefix_sep = prefix_sep
        self.dtype = dtype

    def fit(self, context: InitialTransformContext) -> "OneHotEncodingStep":
        return self

    def transform(self, context: InitialTransformContext) -> InitialTransformContext:
        categorical = self.columns or context.categorical_features
        if not categorical:
            return context

        missing = [col for col in categorical if col not in context.features]
        if missing:
            raise ValueError(f"Columns not found for OneHotEncodingStep: {missing}")

        features_df = context.data[context.features]
        encoded = pd.get_dummies(
            features_df,
            columns=categorical,
            drop_first=self.drop_first,
            prefix_sep=self.prefix_sep,
            dtype=self.dtype,
        )

        encoded = encoded.reset_index(drop=True)
        target_df = context.data[[context.target]].reset_index(drop=True)
        context.data = pd.concat([encoded, target_df], axis=1)
        context.features = list(encoded.columns)

        new_feature_config: Dict[str, FeatureParameters] = {}
        categorical_groups: Dict[str, List[str]] = {}

        for feature in encoded.columns:
            base_feature = self._base_feature_name(feature, categorical)
            if base_feature is None:
                new_feature_config[feature] = copy.deepcopy(
                    context.feature_config.get(feature, FeatureParameters(actionable=True))
                )
                continue

            params = context.feature_config.get(base_feature)
            if params is None:
                params = FeatureParameters(actionable=True)
            new_feature_config[feature] = copy.deepcopy(params)
            categorical_groups.setdefault(base_feature, []).append(feature)

        context.feature_config = new_feature_config
        categorical_columns = {col for cols in categorical_groups.values() for col in cols}
        context.categorical_features = [
            col for col in encoded.columns if col in categorical_columns
        ]
        context.continuous_features = [
            col for col in encoded.columns if col not in categorical_columns
        ]
        context.one_hot_feature_groups = categorical_groups
        return context

    def _base_feature_name(self, feature_name: str, candidates: Sequence[str]) -> Optional[str]:
        for candidate in candidates:
            if feature_name.startswith(f"{candidate}{self.prefix_sep}"):
                return candidate
        return None


INITIAL_TRANSFORM_REGISTRY: Dict[str, Type[InitialTransformStep]] = {
    "dropna": DropNaStep,
    "drop_na": DropNaStep,
    "reorder_columns": ReorderColumnsStep,
    "reorder_features": ReorderColumnsStep,
    "downsample": DownsampleStep,
    "one_hot_encode": OneHotEncodingStep,
    "one_hot_encoding": OneHotEncodingStep,
}


def build_initial_transform_pipeline(
    steps_config: Optional[Sequence[Dict[str, Any]]],
) -> Optional[InitialTransformPipeline]:
    """Create a pipeline instance from YAML configuration."""
    if not steps_config:
        return None

    steps: List[Tuple[str, InitialTransformStep]] = []
    for step_cfg in steps_config:
        name = step_cfg.get("name")
        if name is None:
            raise ValueError("Initial transform step missing 'name' field.")
        params = step_cfg.get("params", {})
        step_class = INITIAL_TRANSFORM_REGISTRY.get(name)
        if step_class is None:
            raise ValueError(f"Unknown initial transform step: {name}")
        step_instance = step_class(**params)
        steps.append((name, step_instance))

    return InitialTransformPipeline(steps) if steps else None
