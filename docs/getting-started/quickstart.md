# Quick Start

Generate your first counterfactual explanation in just a few steps.

!!! note "Multiple Methods Available"
    This tutorial demonstrates **PPCEF**, one of 17+ counterfactual methods available in CEL. 
    The same workflow applies to other methods like DiCE, WACH, and CEM—just import a different class from `counterfactuals.cf_methods.local_methods`.
    [Explore all methods &rarr;](../methods/index.md)

## Overview

This tutorial walks you through:

1. Loading a dataset
2. Training a classifier
3. Training a generative model (flow)
4. Generating counterfactual explanations (using PPCEF)
5. Evaluating the results

## Step 1: Load a Dataset

```python
from counterfactuals.datasets import FileDataset
import torch

# Load the Adult income dataset
dataset = FileDataset(config_path="config/datasets/adult.yaml")

# Access train/test splits
X_train, X_test = dataset.X_train, dataset.X_test
y_train, y_test = dataset.y_train, dataset.y_test

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Features: {dataset.features}")
```

## Step 2: Prepare Data Loaders

```python
from torch.utils.data import DataLoader, TensorDataset

# Convert to PyTorch tensors
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.LongTensor(y_train)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.LongTensor(y_test)

# Create data loaders
train_dataset = TensorDataset(X_train_t, y_train_t)
test_dataset = TensorDataset(X_test_t, y_test_t)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256)
```

## Step 3: Train a Classifier

```python
from counterfactuals.models.classifiers import MLPClassifier

# Get dimensions
n_features = X_train.shape[1]
n_classes = len(set(y_train))

# Create and train classifier
classifier = MLPClassifier(
    input_dim=n_features,
    hidden_dims=[128, 64],
    output_dim=n_classes
)

classifier.fit(
    train_loader=train_loader,
    test_loader=test_loader,
    epochs=50,
    lr=0.001
)

# Check accuracy
accuracy = classifier.score(X_test_t, y_test_t)
print(f"Test accuracy: {accuracy:.2%}")
```

## Step 4: Train a Generative Model

```python
from counterfactuals.models.generators import MaskedAutoregressiveFlow

# Create and train flow model
flow = MaskedAutoregressiveFlow(
    input_dim=n_features,
    hidden_dims=[128, 128],
    n_layers=5
)

flow.fit(
    train_loader=train_loader,
    test_loader=test_loader,
    epochs=100,
    lr=0.0001
)
```

## Step 5: Generate Counterfactuals

```python
from counterfactuals.cf_methods.local_methods import PPCEF
import torch.nn as nn

# Select an instance to explain (someone denied a loan)
idx = (y_test == 0).nonzero()[0][0]  # First instance with class 0
instance = X_test_t[idx:idx+1]
original_class = y_test[idx]
target_class = 1  # We want to find what would get approval

print(f"Original prediction: {original_class}")

# Create PPCEF method
device = "cuda" if torch.cuda.is_available() else "cpu"
method = PPCEF(
    gen_model=flow,
    disc_model=classifier,
    disc_model_criterion=nn.CrossEntropyLoss(),
    device=device
)

# Generate counterfactual
result = method.explain(
    X=instance.numpy(),
    y_origin=original_class,
    y_target=target_class,
    X_train=X_train,
    y_train=y_train,
    epochs=100,
    lr=0.01
)

# Show results
print(f"\nOriginal instance:\n{instance.numpy()}")
print(f"\nCounterfactual:\n{result.x_cfs}")
```

## Step 6: Analyze Changes

```python
import numpy as np

# Compare original and counterfactual
original = instance.numpy().flatten()
counterfactual = result.x_cfs.flatten()
changes = counterfactual - original

print("\nFeature changes:")
for i, (feat, change) in enumerate(zip(dataset.features, changes)):
    if abs(change) > 0.01:  # Only show significant changes
        print(f"  {feat}: {original[i]:.3f} -> {counterfactual[i]:.3f} ({change:+.3f})")
```

## Step 7: Evaluate Quality

```python
from counterfactuals.metrics import MetricsOrchestrator

# Compute metrics
orchestrator = MetricsOrchestrator(
    metrics=["validity", "proximity_l2", "sparsity"],
    gen_model=flow
)

scores = orchestrator.compute(
    x_cfs=result.x_cfs,
    x_origs=result.x_origs,
    y_targets=result.y_cf_targets,
    classifier=classifier
)

print("\nMetrics:")
for metric, value in scores.items():
    print(f"  {metric}: {value:.4f}")
```

## Complete Example

Here's the full code in one block:

??? example "Full Quick Start Code"

    ```python
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    from counterfactuals.datasets import FileDataset
    from counterfactuals.models.classifiers import MLPClassifier
    from counterfactuals.models.generators import MaskedAutoregressiveFlow
    from counterfactuals.cf_methods.local_methods import PPCEF

    # 1. Load dataset
    dataset = FileDataset(config_path="config/datasets/adult.yaml")
    X_train, X_test = dataset.X_train, dataset.X_test
    y_train, y_test = dataset.y_train, dataset.y_test

    # 2. Prepare data loaders
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.LongTensor(y_test)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=256, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(X_test_t, y_test_t),
        batch_size=256
    )

    n_features = X_train.shape[1]
    n_classes = len(set(y_train))

    # 3. Train classifier
    classifier = MLPClassifier(
        input_dim=n_features,
        hidden_dims=[128, 64],
        output_dim=n_classes
    )
    classifier.fit(train_loader, test_loader, epochs=50, lr=0.001)

    # 4. Train flow
    flow = MaskedAutoregressiveFlow(
        input_dim=n_features,
        hidden_dims=[128, 128],
        n_layers=5
    )
    flow.fit(train_loader, test_loader, epochs=100, lr=0.0001)

    # 5. Generate counterfactual
    device = "cuda" if torch.cuda.is_available() else "cpu"
    method = PPCEF(
        gen_model=flow,
        disc_model=classifier,
        disc_model_criterion=nn.CrossEntropyLoss(),
        device=device
    )

    idx = (y_test == 0).nonzero()[0][0]
    result = method.explain(
        X=X_test[idx:idx+1],
        y_origin=0,
        y_target=1,
        X_train=X_train,
        y_train=y_train,
        epochs=100,
        lr=0.01
    )

    print("Counterfactual generated successfully!")
    print(f"Original: {X_test[idx]}")
    print(f"Counterfactual: {result.x_cfs}")
    ```

## Next Steps

- [Core Concepts](concepts.md) - Understand the theory behind counterfactuals
- [User Guide](../user-guide/index.md) - Detailed usage instructions
- [Methods](../methods/index.md) - Explore all 17+ available methods

### Try Other Methods

CEL supports many counterfactual methods. To use a different method, simply change the import:

```python
# Instead of:
from counterfactuals.cf_methods.local_methods import PPCEF
method = PPCEF(...)

# Try:
from counterfactuals.cf_methods.local_methods import DICE
method = DICE(...)

# Or:
from counterfactuals.cf_methods.local_methods import WACH
method = WACH(...)
```

Each method has different strengths—experiment to find the best fit for your use case!
