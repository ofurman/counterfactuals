# CEGP

**Counterfactual Explanations via Genetic Programming**

CEGP uses evolutionary algorithms to search for counterfactual explanations.

## Overview

CEGP applies genetic programming to evolve counterfactual candidates through mutation and crossover operations.

## Usage

```python
from cel.cf_methods.local_methods import CEGP

method = CEGP(
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

::: counterfactuals.cf_methods.local_methods.cegp.cegp.CEGP
