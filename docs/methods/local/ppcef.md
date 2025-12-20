# PPCEF

**Plausible Probabilistic Counterfactual Explanations with Flows**

PPCEF is the flagship method of this library, generating counterfactuals that are both valid and plausible by leveraging normalizing flows.

## Overview

PPCEF optimizes counterfactuals to lie in high-density regions of the data distribution, ensuring they represent realistic inputs rather than adversarial examples.

!!! note "Key Innovation"
    Unlike proximity-only methods, PPCEF uses a generative model (normalizing flow) to assess and maximize the plausibility of generated counterfactuals.

## Algorithm

The method minimizes a combined objective:

$$
\mathcal{L} = \alpha \cdot \mathcal{L}_{\text{validity}} + \beta \cdot \mathcal{L}_{\text{proximity}} + \gamma \cdot \mathcal{L}_{\text{plausibility}}
$$

Where:
- $\mathcal{L}_{\text{validity}}$: Cross-entropy loss for target class
- $\mathcal{L}_{\text{proximity}}$: Distance to original instance
- $\mathcal{L}_{\text{plausibility}}$: Negative log-likelihood under the flow

## Usage

```python
from counterfactuals.cf_methods.local_methods import PPCEF
from counterfactuals.models.generators import MaskedAutoregressiveFlow
from counterfactuals.models.classifiers import MLPClassifier

# Initialize models
gen_model = MaskedAutoregressiveFlow(...)
classifier = MLPClassifier(...)

# Create PPCEF instance
method = PPCEF(
    gen_model=gen_model,
    disc_model=classifier,
    disc_model_criterion=torch.nn.CrossEntropyLoss(),
    device="cuda"
)

# Generate counterfactual
result = method.explain(
    X=instance,
    y_origin=0,
    y_target=1,
    X_train=X_train,
    y_train=y_train,
    epochs=100,
    lr=0.01,
    alpha=1.0,
    beta=0.5
)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gen_model` | BaseGenerator | required | Trained generative model (flow) |
| `disc_model` | BaseClassifier | required | Trained classifier |
| `epochs` | int | 100 | Optimization iterations |
| `lr` | float | 0.01 | Learning rate |
| `alpha` | float | 1.0 | Validity loss weight |
| `beta` | float | 0.5 | Proximity loss weight |

## Strengths

- High plausibility of generated counterfactuals
- Works well with tabular data
- Supports actionability constraints

## Limitations

- Requires training a generative model
- Slower than simple optimization methods
- Performance depends on flow quality

## References

- [Paper citation placeholder]

## API Reference

::: counterfactuals.cf_methods.local_methods.ppcef.ppcef.PPCEF
