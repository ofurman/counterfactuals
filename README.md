# Datasets and Method Results

## Available Datasets

See the detailed table in [docs/datasets/classification.md](docs/datasets/classification.md) for all pre-configured classification datasets, including features, classes, and descriptions.

## Method Support

| Dataset                | PPCEF | GLOBE-CE |
|------------------------|:-----:|:--------:|
| adult                  |   ✓   |    ✓     |
| adult_census           |   ✓   |    ✓     |
| audit                  |   ✓   |    ✓     |
| bank_marketing         |   ✓   |    ✓     |
| blobs                  |   ✓   |    ✓     |
| compas                 |   ✓   |    ✓     |
| credit_default         |   ✓   |    ✓     |
| digits                 |   ✓   |    ✓     |
| german_credit          |   ✓   |    ✓     |
| give_me_some_credit    |   ✓   |    ✓     |
| heloc                  |   ✓   |    ✓     |
| law                    |   ✓   |    ✓     |
| lending_club           |   ✓   |    ✓     |
| moons                  |   ✓   |    ✓     |
| wine                   |   ✓   |    ✓     |

Both PPCEF and GLOBE-CE support all datasets listed above, as long as a YAML config and CSV are present.

For dataset details, see [docs/datasets/classification.md](docs/datasets/classification.md).

---

## Logging Results

To log results for PPCEF and GLOBE-CE, please add your results in the table below.

### Benchmark Scope
- **Datasets**: All 15 classification datasets are covered (regression datasets and MNIST are excluded).
- **Binary Classification (13 datasets)**: Both **GLOBE-CE** and **ARES** methods are evaluated.
- **Multi-class Classification (2 datasets)**: Only **GLOBE-CE** is executed for `wine` and `digits`. **ARES** is excluded as it does not support multi-class targets (rule-based method restricted to binary classification).


### GLOBE-CE Results
| Dataset             | Validity | Proximity | Sparsity | Plausibility | Diversity | Time (s) | Notes                       |
|---------------------|----------|-----------|----------|--------------|-----------|----------|-----------------------------|
| adult               |          |           |          |              |           |          |                             |
| adult_census        |          |           |          |              |           |          |                             |
| audit               |          |           |          |              |           |          | Target=0                    |
| bank_marketing      |          |           |          |              |           |          |                             |
| blobs               |          |           |          |              |           |          | Target=0                    |
| compas              |          |           |          |              |           |          |                             |
| credit_default      |          |           |          |              |           |          |                             |
| digits              |          |           |          |              |           |          |                             |
| german_credit       |          |           |          |              |           |          |                             |
| give_me_some_credit |          |           |          |              |           |          |                             |
| heloc               |          |           |          |              |           |          |                             |
| law                 |          |           |          |              |           |          |                             |
| lending_club        |          |           |          |              |           |          |                             |
| moons               |          |           |          |              |           |          | Target=0                    |
| wine                |          |           |          |              |           |          |                             |

### ARES Results
| Dataset             | Validity | Proximity | Sparsity | Plausibility | Diversity | Time (s) | Notes                       |
|---------------------|----------|-----------|----------|--------------|-----------|----------|-----------------------------|
| adult               |          |           |          |              |           |          |                             |
| adult_census        |          |           |          |              |           |          |                             |
| audit               |          |           |          |              |           |          | Target=0                    |
| bank_marketing      |          |           |          |              |           |          |                             |
| blobs               |          |           |          |              |           |          | Target=0                    |
| compas              |          |           |          |              |           |          |                             |
| credit_default      |          |           |          |              |           |          |                             |
| digits              | N/A      | N/A       | N/A      | N/A          | N/A       | N/A      | Multi-class (Not Supported) |
| german_credit       |          |           |          |              |           |          |                             |
| give_me_some_credit |          |           |          |              |           |          |                             |
| heloc               |          |           |          |              |           |          |                             |
| law                 |          |           |          |              |           |          |                             |
| lending_club        |          |           |          |              |           |          |                             |
| moons               |          |           |          |              |           |          | Target=0                    |
| wine                | N/A      | N/A       | N/A      | N/A          | N/A       | N/A      | Multi-class (Not Supported) |

---

For more information, see the documentation in [docs/datasets/classification.md](docs/datasets/classification.md).
