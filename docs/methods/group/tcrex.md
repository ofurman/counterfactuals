# TCREx

**Tree-based Counterfactual Rule Explanations**

TCREx fits a surrogate decision tree on the data, extracts hyperrectangle rules from its leaves, filters them by accuracy and feasibility, partitions the input space into a grid, assigns an optimal rule per cell, and trains a metarule tree that maps new instances to a counterfactual rule.

## Overview

The method produces axis-aligned region rules: each rule is a `Hyperrectangle` of per-feature bounds together with an accuracy and a feasibility score. At inference, an input is routed to a leaf of the metarule tree, the associated rule is retrieved, and the input is projected onto the rule's bounds to form the counterfactual point.

## Key Features

- Surrogate decision tree extraction of hyperrectangle rules
- Filtering by accuracy threshold `tau` and feasibility threshold `rho`
- Maximal-rule selection plus grid partitioning of the input space
- Metarule decision tree assigning rules to regions for fast inference

## Usage

```python
from cel.cf_methods.group_methods import TCREx

method = TCREx(
    target_model=classifier,
    tau=0.9,
    rho=0.02,
    surrogate_tree_params={"max_leaf_nodes": 8},
)

method.fit(X_train, y_train)
X_cf = method.explain(X_test)
```

## Parameters

### Constructor

| Parameter | Type | Description |
|-----------|------|-------------|
| `target_model` | classifier | Model exposing `predict`; used to score surrogate-tree leaves. |
| `tau` | float | Minimum leaf accuracy for a rule to be kept (default `0.9`). |
| `rho` | float | Minimum leaf feasibility (sample fraction) for a rule (default `0.02`). |
| `surrogate_tree_params` | dict \| None | Kwargs forwarded to the surrogate `DecisionTreeClassifier` (default `{"max_leaf_nodes": 8}`). |

### `fit(X, y)`

Trains the surrogate tree, extracts and filters maximal rules, partitions the input space into grid cells, assigns optimal rules per cell, and fits the metarule tree. Returns `self`.

### `explain(X)`

Routes each row of `X` through the metarule tree and projects it onto the bounds of the selected rule. Returns an array of counterfactual points with shape `X.shape`.

### `explain_dataloader(dataloader)`

Reads `dataloader.dataset.tensors` and delegates to `explain`.

## Helper Classes

- `Hyperrectangle(bounds)`: list of `(lower, upper)` tuples per feature with a `contains` check.
- `CounterfactualRule(hyperrectangle, accuracy, feasibility)`: container for a candidate rule.

## API Reference

::: cel.cf_methods.group_methods.tcrex.tcrex.TCREx

::: cel.cf_methods.group_methods.tcrex.tcrex.Hyperrectangle

::: cel.cf_methods.group_methods.tcrex.tcrex.CounterfactualRule
