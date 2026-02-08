# Generating Counterfactuals

The core workflow for generating counterfactual explanations.

!!! note "Multiple Methods Available"
    CEL provides 14 counterfactual methods. The example below demonstrates PPCEF, 
    but the same `explain()` interface works for all local methods. See [Local Methods](../methods/local/index.md) 
    for a complete list and comparison.

## Basic Usage

This example uses PPCEF (Probabilistically Plausible Counterfactual Explanations with Flows):

```python
from cel.cf_methods.local_methods import PPCEF

# Initialize method
method = PPCEF(
    gen_model=flow,
    disc_model=classifier,
    disc_model_criterion=torch.nn.CrossEntropyLoss(),
    device="cuda"
)

# Generate counterfactual
result = method.explain(
    X=instance,  # Instance to explain
    y_origin=0,  # Current prediction
    y_target=1,  # Desired prediction
    X_train=X_train,  # Training data (for some methods)
    y_train=y_train
)
```

## Using Different Methods

All local methods share the same interface. Simply change the import:

```python
# Option 1: PPCEF (used in example above)
from cel.cf_methods.local_methods import PPCEF

method = PPCEF(gen_model=flow, disc_model=classifier, ...)

# Option 2: DICE (diverse counterfactuals)
from cel.cf_methods.local_methods import DICE

method = DICE(model=classifier, ...)

# Option 3: WACH (gradient-based)
from cel.cf_methods.local_methods import WACH

method = WACH(disc_model=classifier, ...)

# Option 4: CEM (contrastive)
from cel.cf_methods.local_methods import CEM

method = CEM(model=classifier, ...)

# All methods use the same explain() interface:
result = method.explain(X=instance, y_origin=0, y_target=1, ...)
```

## Understanding ExplanationResult

```python
from cel.cf_methods import ExplanationResult

# Result structure
result.x_cfs  # Generated counterfactuals
result.y_cf_targets  # Target labels
result.x_origs  # Original instances
result.y_origs  # Original labels
result.logs  # Training logs (optional)
result.cf_group_ids  # Group assignments (for group methods)
```

## Batch Processing

```python
# Create dataloader
from cel.datasets import TorchDataLoader

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
