# GLANCE

**Group-Level Anchor-based Counterfactual Explanations**

GLANCE uses anchors to define groups and generate group-level counterfactuals.

## Overview

GLANCE identifies anchor points that define natural groupings and generates counterfactuals applicable to each group.

## Usage

```python
from counterfactuals.cf_methods.group_methods import GLANCE

method = GLANCE(
    gen_model=gen_model,
    disc_model=classifier,
    disc_model_criterion=criterion,
    device="cuda"
)

result = method.explain(
    X=X_test,
    y_origin=y_test,
    y_target=target_class,
    X_train=X_train,
    y_train=y_train
)
```

## API Reference

::: counterfactuals.cf_methods.group_methods.glance.glance.GLANCE
