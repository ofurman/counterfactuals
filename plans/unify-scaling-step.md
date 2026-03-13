# Plan: Unify Scaling Steps via Strategy Pattern

## Problem

`MinMaxScalingStep` and `StandardScalingStep` in `counterfactuals/preprocessing/scalers.py`
are **90% identical** (268 lines total). They share the same `fit`, `transform`,
`inverse_transform`, `_transform_array`, and `_inverse_transform_array` logic — the only
difference is which sklearn scaler gets instantiated.

## Current State

```
scalers.py  (268 lines)
├── MinMaxScalingStep(PreprocessingStep)    # lines 10–140, uses SklearnMinMaxScaler
└── StandardScalingStep(PreprocessingStep)  # lines 143–267, uses SklearnStandardScaler
```

- `MinMaxScalingStep` accepts a `feature_range` kwarg and passes it to `SklearnMinMaxScaler`.
- `StandardScalingStep` takes no extra kwargs; instantiates a bare `SklearnStandardScaler()`.
- Everything else — storing indices, copying arrays, slicing continuous columns,
  creating new `PreprocessingContext` — is copy-pasted between the two.

## Usage Across the Codebase

- **27 pipeline files** import `MinMaxScalingStep` and instantiate it as
  `("minmax", MinMaxScalingStep())`.
- **0 pipeline files** currently use `StandardScalingStep`, but it is exported from
  `preprocessing/__init__.py` and available for user code.

## Proposed Design

Replace both classes with a single `ScalingStep` that takes an sklearn scaler class
(and optional kwargs) as a constructor argument — the **Strategy pattern**.

### New `scalers.py` (target: ~95 lines)

```python
from typing import Any, Optional

import numpy as np
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from counterfactuals.preprocessing.base import PreprocessingContext, PreprocessingStep


class ScalingStep(PreprocessingStep):
    """Scaling for continuous features using any sklearn-compatible scaler.

    Wraps an sklearn scaler (e.g. MinMaxScaler, StandardScaler) and applies it
    only to the continuous feature columns identified in the PreprocessingContext.

    Args:
        scaler_cls: An sklearn transformer class. Must implement fit/transform/
            inverse_transform (sklearn's TransformerMixin protocol).
        **scaler_kwargs: Keyword arguments forwarded to the scaler constructor.

    Examples:
        ScalingStep()                                    # MinMaxScaler(0, 1)
        ScalingStep(scaler_cls=StandardScaler)           # zero-mean, unit-var
        ScalingStep(feature_range=(-1, 1))               # MinMaxScaler(-1, 1)
        ScalingStep(scaler_cls=RobustScaler)             # median / IQR
    """

    def __init__(
        self,
        scaler_cls: type[TransformerMixin] = MinMaxScaler,
        **scaler_kwargs: Any,
    ):
        self.scaler_cls = scaler_cls
        self.scaler_kwargs = scaler_kwargs
        self.scaler: Optional[TransformerMixin] = None
        self._continuous_indices: Optional[list[int]] = None
        self._categorical_indices: Optional[list[int]] = None

    def fit(self, context: PreprocessingContext) -> "ScalingStep":
        """Fit the scaler on continuous features from training data."""
        self._continuous_indices = context.continuous_indices
        self._categorical_indices = context.categorical_indices

        if len(self._continuous_indices) > 0:
            X_cont = context.X_train[:, self._continuous_indices]
            self.scaler = self.scaler_cls(**self.scaler_kwargs)
            self.scaler.fit(X_cont)

        return self

    def transform(self, context: PreprocessingContext) -> PreprocessingContext:
        """Scale continuous features."""
        if len(self._continuous_indices) == 0 or self.scaler is None:
            return context

        X_train_transformed = self._transform_array(context.X_train)
        X_test_transformed = (
            self._transform_array(context.X_test) if context.X_test is not None else None
        )

        return PreprocessingContext(
            X_train=X_train_transformed,
            X_test=X_test_transformed,
            y_train=context.y_train,
            y_test=context.y_test,
            categorical_indices=context.categorical_indices,
            continuous_indices=context.continuous_indices,
        )

    def inverse_transform(self, context: PreprocessingContext) -> PreprocessingContext:
        """Reverse the scaling transformation."""
        if len(self._continuous_indices) == 0 or self.scaler is None:
            return context

        X_train_inv = self._inverse_transform_array(context.X_train)
        X_test_inv = (
            self._inverse_transform_array(context.X_test)
            if context.X_test is not None
            else None
        )

        return PreprocessingContext(
            X_train=X_train_inv,
            X_test=X_test_inv,
            y_train=context.y_train,
            y_test=context.y_test,
            categorical_indices=context.categorical_indices,
            continuous_indices=context.continuous_indices,
        )

    def _transform_array(self, X: np.ndarray) -> np.ndarray:
        X_transformed = X.copy()
        X_transformed[:, self._continuous_indices] = self.scaler.transform(
            X[:, self._continuous_indices]
        )
        return X_transformed

    def _inverse_transform_array(self, X: np.ndarray) -> np.ndarray:
        X_inv = X.copy()
        X_inv[:, self._continuous_indices] = self.scaler.inverse_transform(
            X[:, self._continuous_indices]
        )
        return X_inv
```

### Backward-Compatible Aliases

Keep the old names as thin aliases so existing imports don't break:

```python
# Backward-compatible convenience aliases
MinMaxScalingStep = ScalingStep  # default scaler_cls is already MinMaxScaler


def StandardScalingStep() -> ScalingStep:
    """Create a ScalingStep using sklearn's StandardScaler."""
    return ScalingStep(scaler_cls=StandardScaler)
```

> **Note:** `MinMaxScalingStep` is a direct alias (the class itself), because
> `ScalingStep()` already defaults to `MinMaxScaler`. `StandardScalingStep` is a
> factory function because it needs to set `scaler_cls=StandardScaler`.

### Updated `__init__.py`

```python
from counterfactuals.preprocessing.scalers import (
    MinMaxScalingStep,
    ScalingStep,
    StandardScalingStep,
)
```

No changes needed in the 27 pipeline files — `MinMaxScalingStep()` still works.

## Step-by-Step Implementation

### Step 1: Write `ScalingStep` class

- File: `counterfactuals/preprocessing/scalers.py`
- Replace the two classes with the unified `ScalingStep` class shown above.
- Add `MinMaxScalingStep` alias and `StandardScalingStep` factory function at the
  bottom of the file.

### Step 2: Update `__init__.py`

- File: `counterfactuals/preprocessing/__init__.py`
- Add `ScalingStep` to the import and `__all__`.
- Keep `MinMaxScalingStep` and `StandardScalingStep` in `__all__` for backward compat.

### Step 3: Run existing tests

```bash
uv run pytest tests/ -x -q
```

No pipeline files need to change — they all import `MinMaxScalingStep` which is now
an alias for `ScalingStep`.

### Step 4: Add unit tests for `ScalingStep`

- File: `tests/test_preprocessing/test_scalers.py`

Test cases:
1. `ScalingStep()` — default MinMaxScaler, verify output in [0, 1]
2. `ScalingStep(feature_range=(-1, 1))` — custom range
3. `ScalingStep(scaler_cls=StandardScaler)` — zero mean, unit variance
4. Inverse transform round-trip: `X ≈ inverse_transform(transform(X))`
5. Only continuous features are scaled; categorical columns unchanged
6. Empty continuous indices — returns context unchanged
7. Works with `X_test=None`
8. `MinMaxScalingStep()` alias produces identical results
9. `StandardScalingStep()` factory produces identical results

### Step 5: Lint and format

```bash
uv run ruff check counterfactuals/preprocessing/scalers.py --fix
uv run ruff format counterfactuals/preprocessing/scalers.py
```

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| `StandardScalingStep` becomes a function, not a class — breaks `isinstance` checks | Grep the codebase: no `isinstance(..., StandardScalingStep)` calls exist. Safe. |
| Serialized pipelines (pickle) reference old class names | No evidence of pickle-based serialization in the codebase. Hydra configs instantiate fresh objects. |
| Third-party code imports `StandardScalingStep` as a type annotation | Unlikely for an internal library. If needed, make it a subclass instead of a factory. |

## Result

| Metric | Before | After |
|--------|--------|-------|
| Lines in `scalers.py` | 268 | ~105 |
| Classes | 2 (copy-pasted) | 1 (parameterized) |
| Supported scalers | 2 (hardcoded) | Any sklearn-compatible transformer |
| Breaking changes | — | None (aliases preserve API) |
| New flexibility | — | Users can pass `RobustScaler`, `MaxAbsScaler`, `PowerTransformer`, etc. |
