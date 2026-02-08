# Datasets

The library includes **22 pre-configured datasets** for counterfactual explanation research, covering both classification and regression tasks.

## Dataset Categories

<div class="feature-grid" markdown>

<div class="feature-item" markdown>
**Classification Datasets**

15 datasets for binary and multi-class classification tasks.

[View Datasets →](classification.md)
</div>

<div class="feature-item" markdown>
**Regression Datasets**

7 datasets for continuous prediction tasks.

[View Datasets →](regression.md)
</div>

<div class="feature-item" markdown>
**Custom Datasets**

Add your own datasets with YAML configuration.

[Learn How →](custom.md)
</div>

</div>

## Quick Start

```python
from cel.datasets import FileDataset

# Load a pre-configured dataset
dataset = FileDataset(config_path="config/datasets/adult.yaml")

# Access data
X_train, X_test = dataset.X_train, dataset.X_test
y_train, y_test = dataset.y_train, dataset.y_test

# Get feature information
print(f"Features: {dataset.features}")
print(f"Numerical: {dataset.numerical_features}")
print(f"Categorical: {dataset.categorical_features}")
print(f"Actionable: {dataset.actionable_features}")
```

## Dataset Features

All datasets support:

| Feature | Description |
|---------|-------------|
| **Automatic splitting** | 80/20 train/test split (stratified for classification) |
| **Feature typing** | Numerical and categorical feature distinction |
| **Actionability** | Mark which features can be modified |
| **Constraints** | Define bounds and monotonicity constraints |
| **Cross-validation** | Built-in CV split generation |

## Available Datasets

### Classification

| Dataset | Features | Classes | Size | Domain |
|---------|----------|---------|------|--------|
| Adult | 14 | 2 | 48K | Income |
| COMPAS | 12 | 2 | 7K | Recidivism |
| German Credit | 20 | 2 | 1K | Credit |
| HELOC | 23 | 2 | 10K | Credit |
| Give Me Some Credit | 10 | 2 | 150K | Credit |
| ... | | | | |

### Regression

| Dataset | Features | Size | Domain |
|---------|----------|------|--------|
| Concrete | 8 | 1K | Engineering |
| Wine Quality | 11 | 6K | Food |
| Diabetes | 10 | 442 | Health |
| ... | | | |

See the full list in [Classification Datasets](classification.md) and [Regression Datasets](regression.md).
