# ReViCE

**Regional Variant of PPCEF (Group PPCEF)**

ReViCE generates counterfactuals for groups of similar instances.

## Overview

ReViCE extends PPCEF to handle groups, finding counterfactual transformations that work for clusters of similar instances.

## Usage

```python
from cel.cf_methods.group_methods.group_ppcef import RPPCEF

method = RPPCEF(
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
    y_train=y_train,
    n_groups=5
)
```

## API Reference

::: counterfactuals.cf_methods.group_methods.group_ppcef.rppcef.RPPCEF
