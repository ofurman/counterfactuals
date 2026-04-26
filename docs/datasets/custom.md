# Custom Datasets

Add your own datasets to the library.

## YAML Configuration Template

Create a new YAML file in `config/datasets/`:

```yaml
# config/datasets/my_dataset.yaml

# Dataset metadata
name: my_dataset
task_type: classification  # or regression

# Data file path
data_path: data/my_dataset.csv
target_column: target

# Feature definitions
features:
  - age
  - income
  - education
  - category

numerical_features:
  - age
  - income

categorical_features:
  - education
  - category

# Actionability (optional)
actionable_features:
  - income
  - education

# Feature constraints (optional)
feature_constraints:
  age:
    min: 18
    max: 100
    monotonicity: increasing  # only increase allowed
  income:
    min: 0
    max: null  # no upper bound

# Train/test split
train_ratio: 0.8
stratify: true  # for classification
random_state: 42
```

## Loading Custom Dataset

```python
from cel.datasets import FileDataset

dataset = FileDataset(config_path="config/datasets/my_dataset.yaml")
```

## Required Fields

| Field | Description |
|-------|-------------|
| `name` | Dataset identifier |
| `data_path` | Path to CSV file |
| `target_column` | Name of target column |
| `features` | List of feature names |
| `numerical_features` | Continuous features |
| `categorical_features` | Discrete features |

## Optional Fields

| Field | Description |
|-------|-------------|
| `actionable_features` | Features that can be modified |
| `feature_constraints` | Bounds and monotonicity |
| `train_ratio` | Train split proportion |
| `stratify` | Stratified splitting |

## Data Format

Your CSV file should have:
- Header row with column names
- Target column matching `target_column`
- All features listed in `features`
