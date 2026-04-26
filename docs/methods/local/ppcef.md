# PPCEF

**Plausible Probabilistic Counterfactual Explanations with Flows**

PPCEF generates counterfactuals that are both valid and plausible by combining a discriminator loss with a normalizing-flow log-density constraint.

## Overview

For each batch, a per-instance perturbation `delta` is optimized with Adam to minimize:

```
loss = ||delta||_2 + alpha * (disc_loss + relu(plausibility_weight * log_prob_threshold + plausibility_bias - log p(x + delta | y_target)))
```

`explain` is not implemented; pipelines use `explain_dataloader`.

## Usage

```python
import torch
from cel.cf_methods.local_methods import PPCEF

method = PPCEF(
    gen_model=flow_model,
    disc_model=classifier,
    disc_model_criterion=torch.nn.BCEWithLogitsLoss(),
    device="cpu",
)

result = method.explain_dataloader(
    dataloader=train_loader,
    epochs=1000,
    lr=5e-4,
    alpha=1.0,
    log_prob_threshold=-10.0,
)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gen_model` | `GenerativePytorchMixin` | required | Trained generative model exposing `forward(x, context=...)` returning log-density. |
| `disc_model` | `PytorchBase` | required | Trained classifier. |
| `disc_model_criterion` | callable | required | Loss between discriminator logits and target context. |
| `device` | `str \| None` | `"cpu"` | Torch device. |

`explain_dataloader` requires `alpha` and `log_prob_threshold` in `search_step_kwargs`. Optional: `plausibility_weight` (default `1.0`), `plausibility_bias` (default `0.0`), `categorical_intervals`.

## API Reference

::: cel.cf_methods.local_methods.ppcef.ppcef.PPCEF
