# DiCoFlex

**Diverse Counterfactuals with Flexible Constraints**

DiCoFlex generates diverse counterfactual explanations while respecting flexible feature constraints.

## Overview

DiCoFlex extends standard counterfactual generation by:

1. Producing multiple diverse counterfactuals
2. Supporting flexible actionability constraints
3. Balancing diversity with validity

## Usage

```python
from counterfactuals.cf_methods.local_methods.DiCoFlex import DiCoFlex

method = DiCoFlex(
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

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gen_model` | BaseGenerator | required | Trained generative model |
| `disc_model` | BaseClassifier | required | Trained classifier |
| `n_counterfactuals` | int | 5 | Number of CFs to generate |

## API Reference

::: counterfactuals.cf_methods.local_methods.DiCoFlex.method.DiCoFlex
