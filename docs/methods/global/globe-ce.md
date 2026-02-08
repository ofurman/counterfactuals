# GLOBE-CE

**Global Counterfactual Explanations**

GLOBE-CE finds universal transformations that apply across an entire dataset.

## Overview

GLOBE-CE identifies a single transformation direction that, when applied to instances, changes their predictions to the target class.

## Usage

```python
from cel.cf_methods.global_methods.globe_ce import GLOBE_CE

method = GLOBE_CE(
    disc_model=classifier,
    dataset_config=dataset_config
)

result = method.explain(
    X=X_test,
    y_target=target_class
)
```

## API Reference

::: counterfactuals.cf_methods.global_methods.globe_ce.globe_ce.GLOBE_CE
