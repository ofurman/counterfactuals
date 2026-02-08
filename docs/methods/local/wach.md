# WACH

**Weighted Actionable Counterfactual Explanations**

WACH focuses on generating actionable counterfactuals with weighted feature importance.

## Overview

WACH emphasizes actionability by weighting features based on their modifiability.

## Usage

```python
from cel.cf_methods.local_methods import WACH

method = WACH(
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

::: counterfactuals.cf_methods.local_methods.wach.wach.WACH
