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
- **21 Built-in Datasets**: Classification and regression tasks
- **Extensible Architecture**: Easy to add new methods, models, and metrics
- **PyTorch-based**: Modern deep learning framework
- **Cross-validation Support**: Robust evaluation with k-fold CV
- **Preprocessing Pipeline**: Composable feature transformations

## Installation

Clone the repository and set up the environment:

```shell
git clone git@github.com:ofurman/cel.git
cd counterfactuals
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

from cel.datasets import FileDataset, MethodDataset
from cel.cf_methods.local_methods import PPCEF
from cel.models import MaskedAutoregressiveFlow, MLPClassifier
from cel.losses import BinaryDiscLoss
from cel.metrics import evaluate_cf
from cel.preprocessing import (
    MinMaxScalingStep,
    PreprocessingPipeline,
    TorchDataTypeStep,
)

# Load dataset with preprocessing
dataset = MethodDataset.from_config("config/datasets/moons.yaml")
train_loader = dataset.train_dataloader(batch_size=128, shuffle=True)
test_loader = dataset.test_dataloader(batch_size=128, shuffle=False)

file_dataset = FileDataset(config_path="config/datasets/moons.yaml")
preprocessing = PreprocessingPipeline([
    ("minmax", MinMaxScalingStep()),
    ("torch_dtype", TorchDataTypeStep()),
])
dataset = MethodDataset(file_dataset, preprocessing)
train_dataloader = dataset.train_dataloader(batch_size=128, shuffle=True)
test_dataloader = dataset.test_dataloader(batch_size=128, shuffle=False)

disc_model = MLPClassifier(
    num_inputs=dataset.X_train.shape[1],
    num_targets=1,
    hidden_layer_sizes=[256, 256],
    dropout=0.2,
)
disc_model.fit(
    train_dataloader,
    test_dataloader,
    epochs=5000,
    patience=300,
    lr=1e-3,
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
    model_returned=np.ones(X_cf.shape[0]),
    continuous_features=dataset.numerical_features_indices,
    categorical_features=dataset.categorical_features_indices,
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
├── config/                # Dataset configuration YAML files
├── data/                  # Datasets
├── models/                # Pre-trained models
├── notebooks/             # Jupyter notebooks for analysis and examples
├── docs/                  # MkDocs documentation
├── tests/                 # Test suite
├── counterfactuals/       # Source code for the framework
│   ├── cf_methods/        # Counterfactual methods
│   │   ├── global_methods/
│   │   │   ├── ares/
│   │   │   └── globe_ce/
│   │   ├── group_methods/
│   │   │   ├── glance/
│   │   │   ├── pumal/
│   │   │   └── tcrex/
│   │   └── local_methods/
│   │       ├── artelt/
│   │       ├── c_chvae/
│   │       ├── cadex/
│   │       ├── casebased_sace/
│   │       ├── ceflow/
│   │       ├── cegp/
│   │       ├── cem/
│   │       ├── cet/
│   │       ├── dice/
│   │       ├── dicoflex/
│   │       ├── lice/
│   │       ├── ppcef/
│   │       ├── regression_ppcef/
│   │       ├── sace/
│   │       ├── tabdce/
│   │       └── wach/
│   ├── models/            # Neural network models
│   │   ├── classifier/    # Discriminative models (MLP, LR, NODE)
│   │   ├── generative/    # Generative models (MAF, RealNVP, NICE, KDE, CNF)
│   │   └── regression/    # Regression models (LinearRegression, MLPRegressor)
│   ├── datasets/          # Dataset loading and preprocessing
│   ├── preprocessing/     # Preprocessing pipeline (scaling, encoding, torch dtype)
│   ├── dequantization/    # Dequantization utilities for categorical features
│   ├── losses/            # Loss functions (BinaryDiscLoss, MulticlassDiscLoss)
│   ├── metrics/           # Evaluation metrics (validity, proximity, plausibility, etc.)
│   ├── pipelines/         # End-to-end experiment pipelines
│   │   ├── runners/       # Method-specific pipeline runners
│   │   ├── conf/          # Hydra configuration files
│   │   └── nodes/         # Pipeline building blocks
│   └── plotting/          # Visualization utilities
├── README.md              # This document
└── ...
```

## Counterfactual Methods

### Local Methods (Instance-level)

| Method | Class | Description |
|--------|-------|-------------|
| **PPCEF** | `PPCEF` | Probabilistically Plausible CF with normalizing flows |
| **PPCEFR** | `PPCEFR` | PPCEF for regression tasks |
| **DiCE** | `DICE` | Diverse Counterfactual Explanations |
| **CEM** | `CEM_CF` | Contrastive Explanation Method |
| **CET** | `CET` | Counterfactual Explanation Tree |
| **WACH** | `WACH` | Wachter-style gradient-based CF |
| **Artelt** | `Artelt` | Artelt's CF method |
| **SACE** | `SACE`, `CaseBasedSACE` | (Case-based) SACE methods |
| **CEGP** | `CEGP` | CF with Gaussian Processes |
| **C-CHVAE** | `CCHVAE` | Conditional Heterogeneous VAE |
| **DiCoFlex** | `DiCoFlex` | Diverse Counterfactual Flex |
| **LiCE** | `LiCE` | LIME-style CF (requires pyomo/onnx/omlt) |

### Global Methods (Model-level)

| Method | Class | Description |
|--------|-------|-------------|
| **GLOBE-CE** | `GLOBE_CE` | Global Counterfactual Explanations |
| **AReS** | `AReS` | Actionable Recourse Summaries |

### Group Methods (Cohort-level)

| Method | Class | Description |
|--------|-------|-------------|
| **RPPCEF** | `RPPCEF` | Regional PPCEF with shared interventions |
| **GLANCE** | `GLANCE` | Group-level CF method |

## Datasets

The library includes 21 pre-configured datasets:

**Classification:**
`adult`, `adult_census`, `audit`, `bank_marketing`, `compas`, `credit_default`, `diabetes`, `digits`, `german_credit`, `give_me_some_credit`, `heloc`, `law`, `lending_club`, `mnist`, `moons`, `wine`, `blobs`

**Regression:**
`concrete`, `toy_regression`, `wine_quality_regression`, `yacht`

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
uv run python -m cel.pipelines.run_ppcef_pipeline
```

## Documentation

Full documentation is available via MkDocs. To build and serve locally:

```shell
uv run mkdocs serve
```

Key sections:

- **[User Guide](docs/user-guide/index.md)** — datasets, models, generating counterfactuals, evaluation, pipelines.
- **[Methods](docs/methods/index.md)** — per-method descriptions for local, global, and group methods.
- **[Pipelines](docs/user-guide/pipelines.md)** — how `PipelineRunner` orchestrates CV folds, Hydra config structure, and how to add new runners.

## Contributing

Contributions are welcome! Before opening a PR:

- Read `AGENTS.md` and `docs/user-guide/pipelines.md` to understand the workflow, required typing,
  docstrings, and logging conventions.
- Use `uv` for everything (`uv sync`, `uv run ruff check --fix`, `uv run pytest`).
- Keep patches small, fully type-hinted, and Ruff-clean (line length 100, Google docstrings).
- Update or add documentation/tests whenever behavior or configuration changes.

To add new dependencies:
```shell
uv add <package>
```

## Contact

For questions or comments, please contact via LinkedIn: TBA
