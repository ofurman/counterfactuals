# CET

**Counterfactual Explanation Trees**

CET uses tree structures to generate interpretable counterfactual explanations.

## Overview

CET builds decision trees that guide the counterfactual generation process.

## Usage

```python
from cel.cf_methods.local_methods.cet import CounterfactualExplanationTree

method = CounterfactualExplanationTree(
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

::: counterfactuals.cf_methods.local_methods.cet.cet.CounterfactualExplanationTree
