# CEM

**Contrastive Explanation Method (alibi)**

`CEM_CF` wraps `alibi.explainers.CEM`, returning pertinent negatives (PN) as counterfactuals.

## Overview

CEM optimizes a perturbation that flips the prediction while penalizing L1/L2 distance and (optionally) reconstruction error from an auto-encoder. The PyTorch classifier is exposed as a TF callable via `predict_proba`.

## Usage

```python
from cel.cf_methods.local_methods import CEM_CF

method = CEM_CF(
    disc_model=classifier,
    mode="PN",
    kappa=0.2,
    beta=0.1,
    c_init=10.0,
    c_steps=5,
    max_iterations=200,
    learning_rate_init=1e-2,
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
| `mode` | `str` | `"PN"` | `"PN"` (pertinent negative) or `"PP"`. |
| `kappa` | `float` | `0.2` | Confidence margin. |
| `beta` | `float` | `0.1` | L1 regularization weight. |
| `c_init` | `float` | `10.0` | Initial attack-loss coefficient. |
| `c_steps` | `int` | `5` | Number of `c` adjustment steps. |
| `max_iterations` | `int` | `200` | Optimization iterations. |
| `learning_rate_init` | `float` | `1e-2` | Initial learning rate. |
| `no_info_type` | `str` | `"median"` | Reference value strategy. |
| `feature_range` | `tuple[float, float]` | `(0.0, 1.0)` | Feature bounds. |
| `clip` | `tuple[float, float]` | `(-1000.0, 1000.0)` | Gradient clipping bounds. |
| `device` | `str \| None` | `None` | Torch device. |

## API Reference

::: cel.cf_methods.local_methods.cem.cem.CEM_CF
