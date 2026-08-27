# WACH

**Wachter et al. counterfactual explanations**

WACH implements the classic gradient-based counterfactual objective: minimize L2 distance to the input subject to the classifier predicting the target class.

## Overview

For each batch, a per-instance perturbation `delta` is optimized with Adam to minimize:

```
loss = ||delta||_2 + alpha * disc_loss(disc_model(x + delta), y_target)
```

`explain` is not implemented; pipelines use `explain_dataloader`.

## Usage

```python
import torch
from cel.cf_methods.local_methods import WACH

method = WACH(
    disc_model=classifier,
    disc_model_criterion=torch.nn.BCEWithLogitsLoss(),
    device="cpu",
)

result = method.explain_dataloader(
    dataloader=train_loader,
    epochs=1000,
    lr=5e-4,
    alpha=1.0,
)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `disc_model` | `PytorchBase` | required | Trained classifier. |
| `disc_model_criterion` | callable | required | Loss between discriminator logits and target context. |
| `device` | `str \| None` | `"cpu"` | Torch device. |

`explain_dataloader` requires `alpha` in `search_step_kwargs`. Stops early when `disc_loss < patience_eps`.

## API Reference

::: cel.cf_methods.local_methods.wach.wach.WACH
