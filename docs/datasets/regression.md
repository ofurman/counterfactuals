# Regression Datasets

Pre-configured datasets for regression tasks.

## Available Datasets

| Dataset | Features | Size | Description |
|---------|----------|------|-------------|
| `concrete.yaml` | 8 | 1,030 | Concrete compressive strength |
| `wine_quality_regression.yaml` | 11 | 6,497 | Wine quality score |
| `diabetes.yaml` | 10 | 442 | Diabetes progression |
| `yacht.yaml` | 6 | 308 | Yacht hydrodynamics |
| `toy_regression.yaml` | varies | synthetic | Synthetic regression task |

## Usage Example

```python
from counterfactuals.datasets import FileDataset

# Load Concrete dataset
dataset = FileDataset(config_path="config/datasets/concrete.yaml")

print(f"Training samples: {len(dataset.X_train)}")
print(f"Test samples: {len(dataset.X_test)}")
print(f"Target range: [{dataset.y.min():.2f}, {dataset.y.max():.2f}]")
```

## Regression-Specific Methods

For regression tasks, use:
- **PPCEFR**: PPCEF for Regression
- Regression-specific metrics via `RegressionCFMetrics`
