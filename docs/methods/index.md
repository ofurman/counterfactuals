# Counterfactual Methods

The library implements counterfactual explanation methods organized into three categories based on their scope: local (per-instance), global (dataset-wide), and group (subgroup-level).

## Method Categories

### Local Methods

Local methods generate counterfactuals for **individual instances**. Given a single input, they find the minimal change needed to alter the model's prediction.

[Explore Local Methods :material-arrow-right:](local/index.md){ .md-button }

### Global Methods

Global methods find **universal transformations** that work across an entire dataset or subpopulation, providing insights into systematic patterns.

[Explore Global Methods :material-arrow-right:](global/index.md){ .md-button }

### Group Methods

Group methods generate counterfactuals for **clusters or subgroups** of similar instances, balancing individual precision with broader applicability.

[Explore Group Methods :material-arrow-right:](group/index.md){ .md-button }

## Method Comparison

| Method | Category | Plausibility | Diversity | Actionability | Speed |
|--------|----------|--------------|-----------|---------------|-------|
| PPCEF | Local | High | Medium | Yes | Medium |
| DICE | Local | Medium | High | Yes | Fast |
| WACH | Local | Medium | Low | Yes | Fast |
| CCHVAE | Local | High | Medium | Yes | Medium |
| GLOBE-CE | Global | Medium | N/A | Yes | Fast |
| AReS | Global | Medium | N/A | Yes | Medium |
| GLANCE | Group | High | Medium | Yes | Medium |
| TCREx | Group | Medium | Medium | Yes | Fast |

## Choosing a Method

```mermaid
flowchart TD
    A[What scope do you need?] --> B{Single instance?}
    B -->|Yes| C[Local Methods]
    B -->|No| D{Entire dataset?}
    D -->|Yes| E[Global Methods]
    D -->|No| F[Group Methods]

    C --> G{Need plausibility?}
    G -->|Yes| H[PPCEF, CCHVAE]
    G -->|No| I[DICE, WACH]

    E --> J[GLOBE-CE, AReS]
    F --> K[GLANCE, TCREx]
```

## Common Interface

All methods inherit from `BaseCounterfactualMethod` and share a common interface:

```python
from cel.cf_methods import BaseCounterfactualMethod

class YourMethod(BaseCounterfactualMethod):
    def fit(self, X_train, y_train, **kwargs):
        """Prepare the method (optional)."""
        pass

    def explain(self, X, y_origin, y_target, **kwargs):
        """Generate counterfactual explanations."""
        return ExplanationResult(...)
```
