# SACE

**Search-based Abstract Counterfactual Explainer**

SACE is an abstract base class with multiple concrete variants that search for counterfactuals using nearest-neighbor, prototype, tree, and distribution-based strategies.

## Overview

`SACE` defines `fit(b, X)` and `get_counterfactuals(x, k=5, ...)`. Concrete subclasses live alongside the base in `cel.cf_methods.local_methods.sace`:

- `casebased_sace.CaseBasedSACE`
- `distr_sace.DistrSACE`
- `feature_sace.FeatureSACE`
- `neighbor_sace.NeighborSACE`
- `random_sace.RandomSACE`
- `tree_sace.TreeSACE`

## Usage

```python
from cel.cf_methods.local_methods.sace.neighbor_sace import NeighborSACE

method = NeighborSACE(
    metric="euclidean",
    feature_names=feature_names,
    continuous_features=continuous_idx,
    normalize=True,
)
method.fit(b=classifier, X=X_train)

cfs = method.get_counterfactuals(x=instance, k=5)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `variable_features` | `list[int] \| None` | `None` | Indices of features that may change. |
| `weights` | `np.ndarray \| None` | `None` | Per-feature weights for distance. |
| `metric` | `str \| tuple` | `"euclidean"` | Distance metric (or pair for continuous/categorical). |
| `feature_names` | `list[str] \| None` | `None` | Optional feature names. |
| `continuous_features` | `list[int] \| None` | `None` | Indices of continuous features. |
| `categorical_features_index_lists` | `list[list[int]] \| None` | `None` | One-hot groups for categorical features. |
| `normalize` | `bool` | `True` | Standardize features during fit. |
| `pooler` | object | `None` | Optional dimensionality-reducing pooler. |
| `tol` | `float` | `0.01` | Rounding tolerance. |

## API Reference

::: cel.cf_methods.local_methods.sace.sace.SACE
