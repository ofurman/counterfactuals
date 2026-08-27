# Artelt

**Plausible counterfactuals via density estimation**

Artelt fits per-class density estimators (Kernel Density + Gaussian Mixture) and generates plausible counterfactuals constrained to high-density ellipsoids of the target class. Designed for linear (hyperplane) classifiers.

## Overview

Density estimators are fitted on the training data per label, then used to constrain a plausible counterfactual generator that operates on the linear model's coefficients.

## Usage

```python
from cel.cf_methods.local_methods import Artelt

method = Artelt(disc_model=classifier)

result = method.explain(
    X=instance,           # shape: (n_features,) or (1, n_features)
    y_origin=0,
    y_target=1,
    X_train=X_train,
    y_train=y_train,
)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `disc_model` | `PytorchBase` | required | Linear classifier whose first two parameter tensors are interpreted as `coef_` and `intercept_`. |

## API Reference

::: cel.cf_methods.local_methods.artelt.artelt.Artelt
