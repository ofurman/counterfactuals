# CEL: Counterfactual Explanations Library

A comprehensive Python framework for generating and evaluating counterfactual explanations in machine learning models. **CEL** (Counterfactual Explanations Library) provides a unified interface for multiple state-of-the-art counterfactual methods, including local (instance-level), global (model-level), and group (cohort-level) approaches.

## Overview

Counterfactual explanations offer a way to understand machine learning model decisions by explaining what minimal changes would alter a prediction. This library provides a unified framework for generating, evaluating, and comparing different counterfactual explanation methods across various datasets and model types.

The library includes multiple counterfactual methods, from gradient-based approaches like Wachter to advanced methods using normalizing flows for density estimation. It emphasizes plausibility, ensuring that generated explanations are coherent and realistic within the context of the original data.

## Table of Contents

- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Library Structure](#library-structure)
- [Counterfactual Methods](#counterfactual-methods)
- [Datasets](#datasets)
- [Models](#models)
- [Metrics](#metrics)
- [Running Experiments](#running-experiments)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Citation](#citation)
- [Contact](#contact)

## Key Features

- **Multiple CF Method Families**: Local, global, and group counterfactual methods
- **Normalizing Flow Integration**: State-of-the-art density estimation for plausibility
- **Comprehensive Metrics**: 17+ evaluation metrics for counterfactual quality
- **Hydra Configuration**: Flexible experiment management with YAML configs
- **16 Built-in Datasets**: Classification and regression tasks
- **Extensible Architecture**: Easy to add new methods, models, and metrics
- **PyTorch-based**: Modern deep learning framework
- **Cross-validation Support**: Robust evaluation with k-fold CV
- **Preprocessing Pipeline**: Composable feature transformations

## Installation

Clone the repository and set up the environment:

```shell
git clone git@github.com:ofurman/cel.git
cd cel
./setup_env.sh
```

Or install dependencies manually with [uv](https://github.com/astral-sh/uv):

```shell
uv sync
```

**Requirements**: Python >= 3.10

## Quick Start

```python
import torch
from cel.datasets import MethodDataset
from cel.cf_methods import PPCEF
from cel.models import MaskedAutoregressiveFlow, MLPClassifier
from cel.losses import BinaryDiscLoss
from cel.metrics import evaluate_cf

# Load dataset with preprocessing
dataset = MethodDataset.from_config("config/datasets/moons.yaml")
train_loader = dataset.train_dataloader(batch_size=128, shuffle=True)
test_loader = dataset.test_dataloader(batch_size=128, shuffle=False)

# Train discriminative model (classifier)
disc_model = MLPClassifier(
    input_size=dataset.input_size,
    hidden_layer_sizes=[256, 256],
    target_size=1,
    dropout=0.2,
)
disc_model.fit(train_loader, test_loader, epochs=5000, patience=300, lr=1e-3)

# Train generative model (normalizing flow)
gen_model = MaskedAutoregressiveFlow(
    features=dataset.input_size,
    hidden_features=8,
    context_features=1,
)
gen_model.fit(train_loader, test_loader, num_epochs=1000)

# Generate counterfactuals
cf_method = PPCEF(
    gen_model=gen_model,
    disc_model=disc_model,
    disc_model_criterion=BinaryDiscLoss(),
)
log_prob_threshold = torch.quantile(gen_model.predict_log_prob(test_loader), 0.25)
result = cf_method.explain_dataloader(
    test_loader,
    alpha=100,
    log_prob_threshold=log_prob_threshold,
    epochs=4000,
)

# Evaluate results
X_cf = result.x_origs + result.x_cfs
metrics = evaluate_cf(
    disc_model=disc_model,
    gen_model=gen_model,
    X_cf=X_cf,
    X_train=dataset.X_train,
    y_train=dataset.y_train,
    X_test=result.x_origs,
    y_test=result.y_origs,
    y_target=result.y_cf_targets,
    continuous_features=dataset.numerical_features,
    categorical_features=dataset.categorical_features,
    median_log_prob=log_prob_threshold,
)
```

## Library Structure

```
counterfactuals/
├── cf_methods/           # Counterfactual explanation methods
│   ├── local/            # Instance-level methods (PPCEF, DiCE, WACH, etc.)
│   ├── global_/          # Model-level methods (GLOBE-CE, AReS)
│   └── group/            # Cohort-level methods (GLANCE, T-CREx)
├── models/               # ML models
│   ├── discriminative/   # Classifiers (MLP, LogisticRegression, NODE)
│   ├── generative/       # Density estimators (MAF, RealNVP, NICE, KDE)
│   └── regression/       # Regressors (MLP, LinearRegression)
├── datasets/             # Dataset loading and configuration
├── preprocessing/        # Feature transformation pipeline
├── dequantization/       # Categorical feature handling for flows
├── losses/               # Loss functions for CF optimization
├── metrics/              # Evaluation metrics
├── pipelines/            # Experiment orchestration
│   ├── nodes/            # Pipeline components
│   └── conf/             # Hydra configuration files
├── plotting/             # Visualization utilities
└── utils.py              # Helper functions

config/
└── datasets/             # Dataset YAML configurations (16 datasets)

docs/
├── library_overview.md   # Comprehensive package documentation
└── ppcef_pipeline.md     # Pipeline guide
```

## Counterfactual Methods

### Local Methods (Instance-level)

| Method | Class | Description |
|--------|-------|-------------|
| **WACH** | `WACH` | Wachter-style gradient-based CF |
| **Artelt** | `Artelt` | Heuristic-based CF method |
| **DiCE** | `DICE` | Diverse Counterfactual Explanations |
| **CCHVAE** | `CCHVAE` | Conditional Heterogeneous VAE |
| **PPCEF** | `PPCEF` | Probabilistically Plausible CF with normalizing flows |
| **CEM** | `CEM_CF` | Contrastive Explanation Method |
| **CEGP** | `CEGP` | Counterfactual with Gaussian Processes |
| **CADEX** | `CADEX` | Counterfactual explanations via optimization |
| **SACE** | `SACE` | Several SACE variants |
| **CEARM** | `CEARM` | Counterfactual explanation through association rule mining |

### Global Methods (Model-level)

| Method | Class | Description |
|--------|-------|-------------|
| **GLOBE-CE** | `GLOBE_CE` | Global Counterfactual Explanations |
| **AReS** | `AReS` | Actionable Recourse Summaries |

### Group Methods (Cohort-level)

| Method | Class | Description |
|--------|-------|-------------|
| **GLANCE** | `GLANCE` | Group-level CF method |
| **T-CREx** | `TCREx` | Temporal Counterfactual Rule Extraction |

## Datasets

The library includes 16 pre-configured datasets:

**Classification (13):**
`adult_census`, `audit`, `bank_marketing`, `blobs`, `credit_default`, `digits`, `german_credit`, `give_me_some_credit` (GMC), `heloc`, `law`, `lending_club`, `moons`, `wine`

**Regression (3):**
`concrete`, `diabetes`, `yacht`

Dataset configurations are in `config/datasets/*.yaml` and support:
- Automatic feature type detection (continuous/categorical)
- Actionability flags for features
- Cross-validation splits
- Train/test split configuration

## Models

### Discriminative Models

| Model | Class | Use Case |
|-------|-------|----------|
| MLP Classifier | `MLPClassifier` | General classification |
| Logistic Regression | `LogisticRegression` | Binary classification |
| Multinomial LR | `MultinomialLogisticRegression` | Multiclass |
| NODE | `NODE` | Neural Oblivious Decision Ensembles |

### Generative Models

| Model | Class | Description |
|-------|-------|-------------|
| MAF | `MaskedAutoregressiveFlow` | Primary normalizing flow |
| RealNVP | `RealNVP` | Real-valued Non-Volume Preserving |
| NICE | `NICE` | Non-linear Independent Components |
| KDE | `KDE` | Kernel Density Estimation baseline |

### Regression Models

| Model | Class |
|-------|-------|
| MLP Regressor | `MLPRegressor` |
| Linear Regression | `LinearRegression` |

## Metrics

The library provides comprehensive evaluation metrics:

| Category | Metrics |
|----------|---------|
| **Validity** | `coverage`, `validity`, `actionability` |
| **Sparsity** | `sparsity` |
| **Distance** | `proximity_continuous_euclidean`, `proximity_continuous_manhattan`, `proximity_continuous_mad`, `proximity_categorical_hamming`, `proximity_categorical_jaccard`, `proximity_l2_jaccard`, `proximity_mad_hamming` |
| **Plausibility** | `prob_plausibility`, `log_density_cf`, `log_density_test` |
| **Outlier Detection** | `lof_scores_cf`, `lof_scores_test`, `isolation_forest_scores_cf`, `isolation_forest_scores_test` |

## Running Experiments

### Using Hydra Pipelines

```shell
# Run PPCEF pipeline
uv run python cel/pipelines/run_ppcef_pipeline.py

# With custom configuration
uv run python cel/pipelines/run_ppcef_pipeline.py \
  dataset.config_path=config/datasets/heloc.yaml \
  disc_model.model=disc_model/mlp_large \
  counterfactuals_params.target_class=1
```

### Available Pipelines

| Pipeline | Method |
|----------|--------|
| `run_ppcef_pipeline.py` | PPCEF |
| `run_dice_pipeline.py` | DiCE |
| `run_cem_pipeline.py` | CEM |
| `run_cchvae_pipeline.py` | C-CHVAE |
| `run_wach_pipeline.py` | WACH |
| `run_artelt_pipeline.py` | Artelt |
| `run_cegp_pipeline.py` | CEGP |
| `run_cadex_pipeline.py` | CADEX |
| `run_sace_pipeline.py` | SACE |
| `run_cearm_pipeline.py` | CEARM |
| `run_globe_ce_pipeline.py` | GLOBE-CE |
| `run_ares_pipeline.py` | AReS |
| `run_glance_pipeline.py` | GLANCE |
| `run_tcrex_pipeline.py` | T-CREx |

## Documentation

**Live Docs**: https://ofurman.github.io/counterfactuals/

## Contributing

Contributions are welcome! Before opening a PR:

1. Read [`AGENTS.md`](AGENTS.md) and [`docs/ppcef_pipeline.md`](docs/ppcef_pipeline.md) to understand the workflow
2. Use `uv` for all operations:
   ```shell
   uv sync                     # Install dependencies
   uv run ruff check --fix     # Lint and fix
   uv run pytest               # Run tests
   ```
3. Follow the coding standards:
   - Python 3.10+, PEP 8 compliant
   - Full type hints everywhere
   - Google-style docstrings
   - Line length: 100 characters
4. Keep patches small and well-documented
5. Update or add tests when behavior changes

To add new dependencies:
```shell
uv add <package>
```

## Contact

For questions or comments, please contact via LinkedIn: TBA
