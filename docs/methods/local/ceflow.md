# CeFlow

**Normalizing-flow-based counterfactual generator**

CeFlow encodes inputs via a normalizing flow, shifts them in latent space along the difference between class means, and decodes back to input space.

## Overview

`fit` computes per-class means in the latent space (or reuses learned means from the flow). `explain` searches over a grid of step sizes `alpha` between origin- and target-class means, picking the closest decoded sample that flips the prediction.

## Usage

```python
from cel.cf_methods.local_methods.ceflow.ceflow import CeFlow, CeFlowParams

method = CeFlow(
    flow_model=flow,
    disc_model=classifier,
    params=CeFlowParams(alpha_steps=9, distance_metric="original"),
    device="cpu",
)
method.fit(X_train=X_train, y_train=y_train)

result = method.explain(
    X=X,
    y_origin=y_origin,
    y_target=y_target,
)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `flow_model` | `torch.nn.Module` | required | Normalizing flow. Must expose `.inverse()` unless `decode_fn` is given. |
| `disc_model` | `PytorchBase` | required | Predictive model. |
| `params` | `CeFlowParams \| None` | `None` | Search configuration (alpha grid, batch size, distance metric, clamping). |
| `encode_fn` | `Callable \| None` | `None` | Optional input-to-latent override. |
| `decode_fn` | `Callable \| None` | `None` | Optional latent-to-input override. |
| `device` | `str \| None` | `"cpu"` | Torch device. |

## API Reference

::: cel.cf_methods.local_methods.ceflow.ceflow.CeFlow
