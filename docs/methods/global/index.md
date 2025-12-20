# Global Methods

Global counterfactual methods find **universal transformations** that apply across an entire dataset or population. They answer: "What systematic changes would alter predictions for many instances?"

## Available Methods

| Method | Description | Key Feature |
|--------|-------------|-------------|
| [GLOBE-CE](globe-ce.md) | Global counterfactual explanations | Dataset-wide transformations |
| [AReS](ares.md) | Anchor/rule-based explanations | Interpretable rules |

## When to Use Global Methods

Global methods are ideal when you need to:

- Understand **systematic model behavior**
- Identify **policy-level interventions**
- Find transformations that work for **many instances**
- Gain **high-level insights** into the model

## Comparison with Local Methods

| Aspect | Local Methods | Global Methods |
|--------|--------------|----------------|
| Scope | Single instance | Entire dataset |
| Output | Individual counterfactual | Universal transformation |
| Use case | Personal recourse | Policy insights |
| Interpretability | Instance-specific | Broadly applicable |

## Example Usage

```python
from counterfactuals.cf_methods.global_methods import GLOBECE

# Initialize method
method = GLOBECE(
    gen_model=flow_model,
    disc_model=classifier,
    disc_model_criterion=criterion,
    device="cuda"
)

# Find global counterfactual transformation
result = method.explain(
    X=X_test,
    y_origin=y_test,
    y_target=target_class,
    X_train=X_train,
    y_train=y_train
)

# The transformation applies to multiple instances
print(f"Global transformation found for {len(X_test)} instances")
```
