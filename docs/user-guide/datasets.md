# Working with Datasets

Learn how to load, configure, and use datasets for counterfactual generation.

## Loading Pre-configured Datasets

```python
from cel.datasets import FileDataset

# Load a dataset from YAML config
dataset = FileDataset(config_path="config/datasets/adult.yaml")

# Access splits
X_train, X_test = dataset.X_train, dataset.X_test
y_train, y_test = dataset.y_train, dataset.y_test
```

## Dataset Properties

```python
# Feature information
print(dataset.features)              # All feature names
print(dataset.numerical_features)    # Continuous features
print(dataset.categorical_features)  # Discrete features
print(dataset.actionable_features)   # Modifiable features
```

## Cross-Validation

```python
# Get CV splits
cv_splits = dataset.get_cv_splits(n_splits=5)

for fold, (train_idx, val_idx) in enumerate(cv_splits):
    X_train_fold = dataset.X[train_idx]
    X_val_fold = dataset.X[val_idx]
```

## Custom Train/Test Splits

```python
# Custom split ratio
X_train, X_test, y_train, y_test = dataset.split_data(
    dataset.X,
    dataset.y,
    train_ratio=0.7,
    stratify=True
)
```

## Next Steps

- [Training Models](models.md) - Train classifiers and generative models
- [Custom Datasets](../datasets/custom.md) - Add your own datasets
