# GLANCE

**Global Action-based Counterfactual Explanations via clustering**

GLANCE clusters non-target instances, generates per-cluster counterfactual actions with an underlying explainer (DiCE), and iteratively merges clusters to produce a small set of representative group-level actions.

## Overview

GLANCE starts from k-means clusters over instances whose label differs from the target class. For each cluster centroid it generates counterfactuals using a wrapped explainer, then greedily merges cluster pairs minimizing centroid distance plus average action cosine dissimilarity until at most `s` clusters remain. The averaged action of each surviving cluster is applied (optionally with line search) to new instances.

## Key Features

- Group-level counterfactuals via clustering and action averaging
- Greedy bottom-up merging based on centroid distance and action similarity
- Optional line search to push instances across the decision boundary
- Exposes merge history and final cluster actions for inspection

## Usage

```python
from cel.cf_methods.group_methods import GLANCE

method = GLANCE(
    X_test=X_test,
    y_test=y_test,
    model=classifier,
    features=feature_names,
    k=-1,           # initial number of clusters; -1 uses len(X_test)
    s=4,            # final number of groups after merging
    m=1,            # counterfactuals generated per centroid
    target_class=1,
)

result = method.explain(
    X=X_test,
    y_origin=y_test,
    y_target=y_target,
    X_train=X_train,
    y_train=y_train,
)
```

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `X_test` | array-like | Instances used for clustering (only those with `y_test != target_class` are kept). |
| `y_test` | array-like | Labels aligned to `X_test`. |
| `model` | classifier | Model exposing `predict_crisp`. |
| `features` | iterable[str] | Feature names. |
| `k` | int | Initial cluster count; `-1` uses one cluster per kept instance. |
| `s` | int | Target number of groups after merging. |
| `m` | int | Counterfactuals generated per cluster centroid. |
| `target_class` | int | Desired class label for counterfactuals. |

### `explain` arguments

| Parameter | Type | Description |
|-----------|------|-------------|
| `X` | `np.ndarray` | Instances to explain. |
| `y_origin` | `np.ndarray` | Original labels for `X`. |
| `y_target` | `np.ndarray` | Desired target labels. |
| `X_train` | `np.ndarray \| None` | Training features used to fit the underlying explainer. |
| `y_train` | `np.ndarray \| None` | Training labels used to fit the underlying explainer. |
| `**kwargs` | dict | Supports `method_to_use` (default `"dice"`). |

## API Reference

::: cel.cf_methods.group_methods.glance.glance.GLANCE
