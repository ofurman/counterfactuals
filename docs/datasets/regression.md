# Regression Datasets

Pre-configured datasets for regression tasks.

## Available Datasets

| Dataset | Features | Size | Description |
|---------|----------|------|-------------|
| `concrete.yaml` | 8 | 1,030 | Concrete compressive strength |
| `diabetes.yaml` | 10 | 442 | Diabetes progression |
| `yacht.yaml` | 6 | 308 | Yacht hydrodynamics |

## Usage Example

```python
from cel.datasets import FileDataset

# Load Concrete dataset
dataset = FileDataset(config_path="config/datasets/concrete.yaml")

print(f"Training samples: {len(dataset.X_train)}")
print(f"Test samples: {len(dataset.X_test)}")
print(f"Target range: [{dataset.y.min():.2f}, {dataset.y.max():.2f}]")
```
