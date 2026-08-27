# Global Methods

Global counterfactual methods find **universal transformations** that apply
across an entire dataset or population. They answer: "What systematic changes
would alter predictions for many instances?"

## Available Methods

| Method | Description | Key Feature |
|--------|-------------|-------------|
| [GLOBE-CE](globe-ce.md) | Global counterfactual explanations | Dataset-wide translation direction |
| [AReS](ares.md) | Rule-based recourse summaries | Interpretable two-level rules |

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
| Output | Individual counterfactual | Universal transformation or rule set |
| Use case | Personal recourse | Policy insights |
| Interpretability | Instance-specific | Broadly applicable |

## Example Usage

```python
from cel.cf_methods.global_methods import GLOBE_CE

# Initialize method against a fitted classifier and a dataset wrapper
method = GLOBE_CE(
    predict_fn=classifier.predict,
    dataset=dataset,
    X=X_train,
    target_class=1,
)

# Find a global counterfactual transformation and apply it
result = method.explain()

# result is an ExplanationResult; x_cfs holds the generated counterfactuals
print(f"Generated counterfactuals for {result.x_cfs.shape[0]} affected instances")
```
