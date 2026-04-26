# AReS

**Actionable Recourse Summaries**

AReS produces rule-based global counterfactual explanations: a small set of
two-level recourse rules of the form ``If <subgroup> and <inner-if>, Then <then>``
that summarise how predictions can be flipped across an affected population.

## Overview

AReS operates on tabular data with one-hot encoded categorical features and
binned continuous features. It mines candidate predicates with the Apriori
algorithm, builds a ground set of triples (subgroup descriptor, inner-if,
then), and runs a submodular maximisation step to select a compact rule set
under the configured constraints. The mined rules are then materialised into
counterfactual instances for the affected (negatively predicted) inputs.

## Key Features

- Rule-based, human-readable global explanations.
- Supports dropped features and ordinal categorical features.
- Configurable rule budget via `constraints = [e1, e2, e3]` (number of rules,
  maximum rule width, number of unique subgroup descriptors).
- Optional input normalisation prior to calling `predict_fn`.

## Basic Usage

```python
from cel.cf_methods.global_methods import AReS

method = AReS(
    predict_fn=classifier.predict,
    dataset=dataset,
    X=X_train,
    dropped_features=[],
    n_bins=10,
    ordinal_features=[],
    normalise=False,
    constraints=[20, 7, 10],
    correctness=False,
)

result = method.explain(
    apriori_threshold=0.6,
    max_triples_eval=5000,
    max_triples_select=5000,
    disable_tqdm=False,
)

# result is an ExplanationResult with x_cfs, y_cf_targets, x_origs, y_origs, logs
```

## Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `predict_fn` | Callable | required | Black-box predict function returning a binary prediction. |
| `dataset` | dataset wrapper | required | Custom dataset object exposing `features`, `features_tree`, and `categorical_features`. |
| `X` | `pandas.DataFrame` | required | Training data (positive and negative predictions) used to derive affected inputs. |
| `dropped_features` | `list[str]` | `[]` | Feature names to exclude from rule generation. |
| `n_bins` | `int` | `10` | Number of equal-width bins for continuous features. |
| `ordinal_features` | `list[str]` | `[]` | Categorical features that should be treated as ordinal when computing costs. |
| `normalise` | `bool` | `False` | If `True`, standardise inputs before calling `predict_fn`. |
| `constraints` | `list[int]` | `[20, 7, 10]` | `[e1, e2, e3]` constraints from the AReS paper. |
| `correctness` | `bool` | `False` | Reserved flag controlling correctness-only evaluation. |

## `explain` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `apriori_threshold` | `float` | `0.6` | Minimum support threshold for Apriori itemset mining. |
| `y_origin` | `np.ndarray \| None` | `None` | Optional original labels for affected inputs. |
| `y_target` | `np.ndarray \| None` | `None` | Optional target labels (defaults to ones). |
| `max_triples_eval` | `int` | `5000` | Cap on the number of triples evaluated. |
| `max_triples_select` | `int` | `5000` | Cap on the number of triples kept after selection. |
| `disable_tqdm` | `bool` | `False` | Disable progress bars. |

`explain` returns an `ExplanationResult` containing `x_cfs`, `y_cf_targets`,
`x_origs`, `y_origs`, and `logs`.

## API Reference

::: cel.cf_methods.global_methods.AReS
