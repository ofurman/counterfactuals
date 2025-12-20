# AReS

**Anchor/Rule-based Explanations**

AReS generates rule-based global explanations.

## Overview

AReS identifies interpretable rules that describe when and how predictions can be changed.

## Usage

```python
from counterfactuals.cf_methods.global_methods import AReS

method = AReS(
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

::: counterfactuals.cf_methods.global_methods.ares.ares.AReS
