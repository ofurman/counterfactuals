# Counterfactual Methods

The library implements **17+ counterfactual explanation methods** organized into three categories based on their scope.

## Method Categories

### Local Methods

Local methods generate counterfactuals for **individual instances**. Given a single input, they find the minimal change needed to alter the model's prediction.

[Explore Local Methods →](local/index.md){ .md-button }

### Global Methods

Global methods find **universal transformations** that work across an entire dataset or subpopulation, providing insights into systematic patterns.

[Explore Global Methods →](global/index.md){ .md-button }

### Group Methods

Group methods generate counterfactuals for **clusters or subgroups** of similar instances, balancing individual precision with broader applicability.

[Explore Group Methods →](group/index.md){ .md-button }

## Method Comparison

| Method | Category | Plausibility | Diversity | Actionability | Speed | Best For |
|--------|----------|--------------|-----------|---------------|-------|----------|
| PPCEF | Local | High | Medium | Yes | Medium | Normalizing flow-based CF |
| DiCoFlex | Local | High | High | Yes | Medium | Flexible, diverse CFs |
| DICE | Local | Medium | High | Yes | Fast | Fast, diverse CFs |
| WACH | Local | Low | Low | No | Fast | Simple gradient-based CF |
| CEM | Local | Medium | Low | Yes | Medium | Contrastive explanations |
| CCHVAE | Local | High | Medium | Yes | Slow | VAE-based CF |
| GLOBE-CE | Global | Medium | N/A | Yes | Fast | Dataset-wide patterns |
| AReS | Global | Medium | N/A | Yes | Medium | Actionable rules |
| ReViCE | Group | High | Medium | Yes | Medium | Subgroup explanations |
| GLANCE | Group | Medium | High | Yes | Fast | Cluster-level CFs |

## Choosing a Method

```mermaid
flowchart TD
    A[What scope do you need?] --> B{Single instance?}
    B -->|Yes| C[Local Methods]
    B -->|No| D{Entire dataset?}
    D -->|Yes| E[Global Methods]
    D -->|No| F[Group Methods]

    C --> G{Need plausibility?}
    G -->|Yes| H[PPCEF, DiCoFlex, CCHVAE]
    G -->|No| I[DICE, WACH, CEM]

    E --> J[GLOBE-CE, AReS]
    F --> K[ReViCE, GLANCE]
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
