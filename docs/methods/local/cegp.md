# CEGP

**Counterfactuals Guided by Prototypes (alibi)**

CEGP wraps `alibi.explainers.CounterFactualProto`, generating counterfactuals guided by class prototypes.

## Overview

A TensorFlow session is initialized internally, and the wrapped alibi explainer optimizes a perturbation guided by an auto-encoder/prototype loss.

## Usage

```python
from cel.cf_methods.local_methods import CEGP

method = CEGP(
    disc_model=classifier,
    beta=0.01,
    c_init=1.0,
    c_steps=5,
    max_iterations=500,
    feature_range=(0.0, 1.0),
)
method.fit(X_train)

result = method.explain(
    X=instance,           # shape: (n_features,)
    y_origin=0,
    y_target=1,
)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `disc_model` | `PytorchBase` | required | Classifier exposing `predict_proba` and `num_inputs`. |
| `beta` | `float` | `0.01` | L1 regularization weight. |
| `c_init` | `float` | `1.0` | Initial value of the attack-loss coefficient. |
| `c_steps` | `int` | `5` | Number of `c` adjustment steps. |
| `max_iterations` | `int` | `500` | Optimization iterations. |
| `feature_range` | `tuple[float, float]` | `(0.0, 1.0)` | Bounds for features. |
| `d_type` | `str` | `"abdm"` | Distance metric for categorical encoding. |
| `disc_perc` | `list[int] \| None` | `[25, 50, 75]` | Percentiles used to discretize numerical features. |
| `device` | `str \| None` | `None` | Torch device. |

## API Reference

::: cel.cf_methods.local_methods.cegp.cegp.CEGP
