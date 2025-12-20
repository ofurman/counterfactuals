# SACE

**SACE Counterfactual Methods**

SACE provides multiple variants for generating counterfactual explanations.

## Variants

- **Standard SACE**: Base implementation
- **Case-based SACE**: Uses case-based reasoning
- **Feature SACE**: Feature-focused approach
- **Random SACE**: Randomized search
- **Neighbor SACE**: Neighbor-based generation
- **Tree SACE**: Tree-structured search
- **Distribution SACE**: Distribution-aware generation

## Usage

```python
from counterfactuals.cf_methods.local_methods.sace import SACE

method = SACE(
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

::: counterfactuals.cf_methods.local_methods.sace.sace.SACE
