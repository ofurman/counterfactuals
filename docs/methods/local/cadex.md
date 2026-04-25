# CADEX

**Constrained Adversarial Counterfactual Explanations**

CADEX optimizes an additive input perturbation under categorical, ordinal, and direction constraints to flip a classifier's prediction toward a target class.

## Overview

A trainable `delta` is added to the input and updated with Adam to minimize a classification loss. Categorical groups are projected to one-hot, and ordinal features can be rounded after applying inverse scaling.

## Usage

```python
from cel.cf_methods.local_methods import CADEX

method = CADEX(disc_model=classifier, device="cpu")

result = method.explain(
    X=X,                  # shape: (n_instances, n_features)
    y_origin=y_origin,
    y_target=y_target,
    max_epochs=1000,
)
```

`explain_dataloader` is not implemented and raises `NotImplementedError`.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `disc_model` | `torch.nn.Module` | required | Classifier producing logits. |
| `categorical_attributes` | `list[list[int]] \| None` | `None` | Index groups for one-hot categorical features. |
| `ordinal_attributes` | `list[int] \| None` | `None` | Indices of ordinal features (requires `scale`/`unscale`). |
| `scale` | `Callable \| None` | `None` | Forward scaling function for ordinal handling. |
| `unscale` | `Callable \| None` | `None` | Inverse scaling function for ordinal handling. |
| `device` | `str \| None` | `"cpu"` | Torch device. |

## API Reference

::: cel.cf_methods.local_methods.cadex.cadex.CADEX
