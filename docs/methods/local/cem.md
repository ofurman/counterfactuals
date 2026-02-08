# CEM

**Contrastive Explanation Method**

CEM generates counterfactuals by finding contrastive perturbations.

## Overview

CEM identifies both pertinent positives (features that must be present) and pertinent negatives (features that must be absent) for a prediction.

## Usage

```python
from cel.cf_methods.local_methods.cem import CEM_CF

method = CEM_CF(
    gen_model=gen_model,
    disc_model=classifier,
    disc_model_criterion=criterion,
    device="cuda"
)

result = method.explain(
    X=instance,
    y_origin=0,
    y_target=1,
    X_train=X_train,
    y_train=y_train
)
```

## API Reference

::: counterfactuals.cf_methods.local_methods.cem.cem.CEM_CF
