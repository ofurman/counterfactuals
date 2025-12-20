# DICE

**Diverse Counterfactual Explanations**

DICE is a popular method for generating diverse counterfactual explanations, integrated via the `dice-ml` library.

## Overview

DICE generates multiple diverse counterfactuals by optimizing for both validity and diversity simultaneously.

## Usage

```python
from counterfactuals.cf_methods.local_methods import DICE

method = DICE(
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
| `disc_model` | BaseClassifier | required | Trained classifier |
| `n_counterfactuals` | int | 5 | Number of CFs to generate |

## References

- Mothilal et al., "Explaining Machine Learning Classifiers through Diverse Counterfactual Explanations"

## API Reference

::: counterfactuals.cf_methods.local_methods.dice.dice.DICE
