# Group Methods

Group counterfactual methods generate explanations for **clusters or subgroups** of similar instances. They balance individual precision with broader applicability.

## Available Methods

| Method | Description | Key Feature |
|--------|-------------|-------------|
| [ReViCE](revice.md) | Regional variant of PPCEF | Flow-based group CFs |
| [GLANCE](glance.md) | Group-level anchor counterfactuals | Anchor-based grouping |
| [Group GLOBE-CE](group-globe-ce.md) | Group variant of GLOBE-CE | Subpopulation focus |

## When to Use Group Methods

Group methods are ideal when you need to:

- Provide explanations for **similar users**
- Balance **personalization** with **scalability**
- Identify **subpopulation-specific** patterns
- Generate **semi-personalized recourse**

## How Groups Are Formed

Group methods typically cluster instances based on:

- Feature similarity
- Prediction confidence
- Demographic attributes
- Custom grouping criteria

```mermaid
flowchart LR
    A[Dataset] --> B[Clustering]
    B --> C[Group 1]
    B --> D[Group 2]
    B --> E[Group N]
    C --> F[Group CF 1]
    D --> G[Group CF 2]
    E --> H[Group CF N]
```

## Example Usage

```python
from counterfactuals.cf_methods.group_methods import GroupPPCEF

# Initialize method
method = GroupPPCEF(
    gen_model=flow_model,
    disc_model=classifier,
    disc_model_criterion=criterion,
    device="cuda"
)

# Generate group counterfactuals
result = method.explain(
    X=X_test,
    y_origin=y_test,
    y_target=target_class,
    X_train=X_train,
    y_train=y_train,
    n_groups=5  # Number of groups
)

# Each instance is assigned to a group
print(f"Group assignments: {result.cf_group_ids}")
```
