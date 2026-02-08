# Classification Datasets

Pre-configured datasets for classification tasks.

## Available Datasets

| Dataset | Features | Classes | Size | Description |
|---------|----------|---------|------|-------------|
| `adult_census.yaml` | 12 | 2 | 32,561 | Income prediction (>50K) |
| `audit.yaml` | 23 | 2 | 775 | Audit risk assessment |
| `bank_marketing.yaml` | 16 | 2 | 40,004 | Term deposit subscription |
| `blobs.yaml` | 2 | 3 | 1,500 | Gaussian blobs (synthetic) |
| `credit_default.yaml` | 23 | 2 | 30,000 | Credit card default |
| `digits.yaml` | 64 | 10 | 1,797 | Handwritten digits |
| `german_credit.yaml` | 20 | 2 | 1,000 | Credit risk assessment |
| `give_me_some_credit.yaml` | 10 | 2 | 16,714 | Credit scoring (GMC) |
| `heloc.yaml` | 23 | 2 | 10,459 | Home equity line of credit |
| `law.yaml` | 5 | 2 | 2,216 | Law school admission |
| `lending_club.yaml` | 12 | 2 | 93,888 | Loan approval |
| `moons.yaml` | 2 | 2 | 1,024 | Two interleaving half circles (synthetic) |
| `wine.yaml` | 13 | 3 | 178 | Wine classification |

## Usage Example

```python
from cel.datasets import FileDataset

# Load Adult Census dataset
dataset = FileDataset(config_path="config/datasets/adult_census.yaml")

print(f"Training samples: {len(dataset.X_train)}")
print(f"Test samples: {len(dataset.X_test)}")
print(f"Features: {len(dataset.features)}")
```
