# GLOBE-CE

**Global Counterfactual Explanations**

GLOBE_CE finds a single global translation (delta) that, when scaled per
input, flips predictions of affected instances toward a target class.

## Overview

GLOBE_CE samples candidate translation directions over the feature space,
respecting categorical structure and optional monotonicity constraints,
selects the best-performing delta, and per-input scales it to balance
coverage and cost. The resulting counterfactuals are returned for all
affected inputs (those not already predicted as `target_class`).

## Key Features

- Single global direction with per-input scaling.
- Handles one-hot encoded categorical features and continuous features.
- Optional `monotonicity` mask, dropped features, ordinal features, and
  affected-subgroup filtering.
- Cost computation via `feature_costs_vector` with optional bin widths.

## Basic Usage

```python
from cel.cf_methods.global_methods import GLOBE_CE

method = GLOBE_CE(
    predict_fn=classifier.predict,
    dataset=dataset,
    X=X_train,
    affected_subgroup=None,
    dropped_features=[],
    ordinal_features=[],
    delta_init="zeros",
    normalise=None,
    bin_widths=None,
    monotonicity=None,
    p=1,
    target_class=1,
)

result = method.explain()

# result is an ExplanationResult with x_cfs, y_cf_targets, x_origs, y_origs, logs
```

## Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `predict_fn` | Callable | required | Black-box predict function returning a class label. |
| `dataset` | dataset wrapper | required | Custom dataset exposing `features`, `features_tree`, and categorical/numerical column metadata. |
| `X` | `pandas.DataFrame` | required | Inputs used to identify affected instances (predictions != `target_class`). |
| `affected_subgroup` | `str \| None` | `None` | Optional feature-value name selecting a subgroup of interest. |
| `dropped_features` | `list[str]` | `[]` | Feature names excluded from the translation. |
| `ordinal_features` | `list[str]` | `[]` | Categorical features treated as ordinal when computing costs. |
| `delta_init` | `str \| np.ndarray` | `"zeros"` | Initial delta. `"zeros"` initialises to a zero vector; otherwise an array is copied. |
| `normalise` | `tuple \| None` | `None` | Reserved normalisation hook (currently disabled internally). |
| `bin_widths` | `dict \| None` | `None` | Optional mapping from continuous feature name to bin width for cost weighting. |
| `monotonicity` | array-like \| `None` | `None` | Optional sign mask applied to sampled deltas. |
| `p` | `int` | `1` | Norm order used for cost computation on continuous-only deltas. |
| `target_class` | `int` | `1` | Desired target class for affected inputs. |

## `explain` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `y_origin` | `np.ndarray \| None` | `None` | Optional original labels for affected inputs. Defaults to current predictions. |
| `y_target` | `np.ndarray \| None` | `None` | Optional target labels. Defaults to a vector filled with `target_class`. |

`explain` returns an `ExplanationResult` containing `x_cfs`, `y_cf_targets`,
`x_origs`, `y_origs`, and `logs`.

## API Reference

::: cel.cf_methods.global_methods.GLOBE_CE
