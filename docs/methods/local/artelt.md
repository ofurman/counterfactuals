# Artelt

**Artelt's Heuristic Counterfactual Method**

Artelt's method uses heuristics for fast counterfactual generation.

## Variants

- **Artelt**: Base implementation
- **Heuristic20**: Optimized variant with 20 heuristics

## Usage

```python
from cel.cf_methods.local_methods import Artelt

method = Artelt(
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

::: counterfactuals.cf_methods.local_methods.artelt.artelt.Artelt
