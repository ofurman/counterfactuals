# Generating Counterfactuals

The core workflow for generating counterfactual explanations.

## Basic Usage

```python
from counterfactuals.cf_methods.local_methods import PPCEF

# Initialize method
method = PPCEF(
    gen_model=flow,
    disc_model=classifier,
    disc_model_criterion=torch.nn.CrossEntropyLoss(),
    device="cuda"
)

# Generate counterfactual
result = method.explain(
    X=instance,           # Instance to explain
    y_origin=0,           # Current prediction
    y_target=1,           # Desired prediction
    X_train=X_train,      # Training data (for some methods)
    y_train=y_train
)
```

## Understanding ExplanationResult

```python
from counterfactuals.cf_methods import ExplanationResult

# Result structure
result.x_cfs         # Generated counterfactuals
result.y_cf_targets  # Target labels
result.x_origs       # Original instances
result.y_origs       # Original labels
result.logs          # Training logs (optional)
result.cf_group_ids  # Group assignments (for group methods)
```

## Batch Processing

```python
# Create dataloader
from counterfactuals.datasets import TorchDataLoader

loader = TorchDataLoader(X_test, y_test, batch_size=32)

# Generate for multiple instances
result = method.explain_dataloader(
    dataloader=loader,
    epochs=100,
    lr=0.01
)
```

## Common Parameters

| Parameter | Description |
|-----------|-------------|
| `epochs` | Number of optimization iterations |
| `lr` | Learning rate |
| `alpha` | Validity loss weight |
| `beta` | Proximity loss weight |
| `K` | Number of counterfactuals per instance |

## Next Steps

- [Evaluating Results](evaluation.md) - Assess counterfactual quality
