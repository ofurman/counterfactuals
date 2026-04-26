# DICE

**Diverse Counterfactual Explanations**

DICE wraps the `dice-ml` library to generate diverse counterfactuals via gradient descent.

## Overview

DICE produces multiple diverse counterfactuals by jointly optimizing for validity, proximity, and diversity. This wrapper uses `dice_ml.Dice(..., method="gradient")`.

## Usage

```python
from cel.cf_methods.local_methods import DICE

method = DICE(
    X_train=X_train,
    y_train=y_train,
    features=feature_names,   # ordered list; last entry is treated as the outcome column
    disc_model=classifier,
)

result = method.explain(
    Xs=instances,             # 2D numpy array
    ys=y_origin,
    total_CFs=1,
    desired_class="opposite",
)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X_train` | `np.ndarray` | required | Training features. |
| `y_train` | `np.ndarray` | required | Training labels. |
| `features` | `list[str]` | required | Feature names. |
| `disc_model` | model | required | PyTorch classifier (passed to `dice_ml.Model(..., backend="PYT")`). |

`explain` and `explain_dataloader` accept the standard DiCE arguments: `total_CFs`, `desired_class`, `desired_range`, `permitted_range`, `features_to_vary`, `stopping_threshold`, `posthoc_sparsity_param`, `posthoc_sparsity_algorithm`, `verbose`.

## References

- Mothilal et al., "Explaining Machine Learning Classifiers through Diverse Counterfactual Explanations"

## API Reference

::: cel.cf_methods.local_methods.dice.dice.DICE
