# CEARM

**Counterfactual Explanations via Bayesian Optimization**

CEARM uses Bayesian optimization (GPyOpt) with an exponential potential acquisition to search for counterfactuals that shift the regressor's output toward a target value.

## Overview

For each test instance, a Gaussian-process Bayesian optimizer maximizes an exponential potential between the current and target prediction over the unit hypercube `[0, 1]^d`.

## Usage

```python
from cel.cf_methods.local_methods import CEARM

method = CEARM(disc_model=regressor, device="cpu")

# Pipelines/runners use the dataloader entry point.
result = method.explain_dataloader(
    dataloader=test_loader,
    target_change=0.2,
)
```

`explain` raises `NotImplementedError`. `explain_dataloader` returns a tuple `(x_cfs, x_origs, y_origs, y_cf_targets, None)` rather than an `ExplanationResult`.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `disc_model` | `PytorchBase` | required | Regression/classification model used as the BO objective. |
| `device` | `str \| None` | `"cpu"` | Torch device. |

`explain_dataloader` accepts `epochs`, `lr`, `patience_eps`, and `target_change` (default `0.2`).

## Notes

CEARM requires the optional `GPyOpt`/`GPy` stack: `pip install 'ce-library[cearm]'`.

## API Reference

::: cel.cf_methods.local_methods.cearm.cearm.CEARM
