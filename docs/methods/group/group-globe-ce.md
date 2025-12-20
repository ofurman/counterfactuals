# Group GLOBE-CE

**Group Variant of GLOBE-CE**

Group GLOBE-CE extends GLOBE-CE to find transformations for subpopulations.

## Overview

Group GLOBE-CE partitions the data into groups and finds optimal transformations for each group.

## Usage

Group GLOBE-CE can be implemented by combining GLOBE-CE with group clustering. See the [RPPCEF](revice.md) implementation for a similar approach.

```python
# Group GLOBE-CE is implemented via the RPPCEF method
# with GLOBE-CE style deltas
from counterfactuals.cf_methods.group_methods.group_ppcef import RPPCEF

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

## Related Methods

- [GLOBE-CE](../global/globe-ce.md) - Global counterfactual explanations
- [ReViCE](revice.md) - Regional/group PPCEF
