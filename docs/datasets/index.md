# Datasets

The library includes **18 pre-configured datasets** for counterfactual explanation research, covering both classification and regression tasks.

## Dataset Categories

<div class="feature-grid" markdown>

<div class="feature-item" markdown>
**Classification Datasets**

13 datasets for binary and multi-class classification tasks.

[View Datasets →](classification.md)
</div>

<div class="feature-item" markdown>
**Regression Datasets**

5 datasets for continuous prediction tasks.

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
dataset = FileDataset(config_path="config/datasets/adult_census.yaml")

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
| Adult Census | 12 | 2 | 32,561 | Income |
| Audit | 23 | 2 | 775 | Risk |
| Bank Marketing | 16 | 2 | 40,004 | Marketing |
| Blobs | 2 | 3 | 1,500 | Synthetic |
| Credit Default | 23 | 2 | 30,000 | Credit |
| Digits | 64 | 10 | 1,797 | Images |
| German Credit | 20 | 2 | 1,000 | Credit |
| GMC | 10 | 2 | 16,714 | Credit |
| HELOC | 23 | 2 | 10,459 | Credit |
| Law | 5 | 2 | 2,216 | Education |
| Lending Club | 12 | 2 | 93,888 | Credit |
| Moons | 2 | 2 | 1,024 | Synthetic |
| Wine | 13 | 3 | 178 | Food |

### Regression

| Dataset | Features | Size | Domain |
|---------|----------|------|--------|
| Concrete | 8 | 1,030 | Engineering |
| Diabetes | 10 | 442 | Health |
| Yacht | 6 | 308 | Engineering |
| Synthetic | 2 | 1,000 | Synthetic |
| SCM20D | 61 | 8,966 | Multi-target |

See the full list in [Classification Datasets](classification.md) and [Regression Datasets](regression.md).
