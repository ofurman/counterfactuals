# Classification Datasets

Pre-configured datasets for classification tasks.

## Financial/Credit

| Dataset | Features | Classes | Size | Description |
|---------|----------|---------|------|-------------|
| `adult.yaml` | 14 | 2 | 48,842 | Income prediction (>50K) |
| `german_credit.yaml` | 20 | 2 | 1,000 | Credit risk assessment |
| `credit_default.yaml` | 23 | 2 | 30,000 | Credit card default |
| `give_me_some_credit.yaml` | 10 | 2 | 150,000 | Credit scoring |
| `heloc.yaml` | 23 | 2 | 10,459 | Home equity line of credit |
| `lending_club.yaml` | varies | 2 | varies | Loan approval |

## Criminal Justice

| Dataset | Features | Classes | Size | Description |
|---------|----------|---------|------|-------------|
| `compas.yaml` | 12 | 2 | 7,214 | Recidivism prediction |

## Banking/Marketing

| Dataset | Features | Classes | Size | Description |
|---------|----------|---------|------|-------------|
| `bank_marketing.yaml` | 16 | 2 | 45,211 | Term deposit subscription |

## Other

| Dataset | Features | Classes | Size | Description |
|---------|----------|---------|------|-------------|
| `wine.yaml` | 13 | 3 | 178 | Wine classification |
| `digits.yaml` | 64 | 10 | 1,797 | Handwritten digits |
| `moons.yaml` | 2 | 2 | synthetic | Two interleaving half circles |
| `blobs.yaml` | 2 | 2 | synthetic | Gaussian blobs |
| `law.yaml` | varies | 2 | varies | Law school admission |
| `audit.yaml` | varies | 2 | varies | Audit risk |

## Usage Example

```python
from cel.datasets import FileDataset

# Load Adult dataset
dataset = FileDataset(config_path="config/datasets/adult.yaml")

print(f"Training samples: {len(dataset.X_train)}")
print(f"Test samples: {len(dataset.X_test)}")
print(f"Features: {len(dataset.features)}")
```
