# Counterfactuals

A comprehensive Python library for generating and evaluating counterfactual explanations.

---

**Counterfactuals** provides a unified framework for counterfactual explanation methods, datasets, and evaluation metrics. Whether you're researching new methods, benchmarking existing approaches, or building explainable AI systems, this library has you covered.

## Features

<div class="feature-grid" markdown>

<div class="feature-item" markdown>
**17+ Explanation Methods**

Local, global, and group-level counterfactual methods including PPCEF, DICE, GLOBE-CE, and more.

[Explore Methods :material-arrow-right:](methods/index.md)
</div>

<div class="feature-item" markdown>
**22 Pre-configured Datasets**

Ready-to-use datasets for classification and regression tasks with built-in preprocessing.

[View Datasets :material-arrow-right:](datasets/index.md)
</div>

<div class="feature-item" markdown>
**18+ Evaluation Metrics**

Comprehensive metrics for validity, proximity, sparsity, plausibility, and diversity.

[See Metrics :material-arrow-right:](benchmarks/metrics.md)
</div>

<div class="feature-item" markdown>
**End-to-End Pipelines**

Hydra-based configuration system for reproducible experiments with MLflow logging.

[Run Pipelines :material-arrow-right:](user-guide/pipelines.md)
</div>

</div>

## Quick Start

### Installation

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### Generate Your First Counterfactual

```python
from counterfactuals.datasets import FileDataset
from counterfactuals.models.classifiers import MLPClassifier
from counterfactuals.models.generators import MaskedAutoregressiveFlow
from counterfactuals.cf_methods.local_methods import PPCEF

# 1. Load dataset
dataset = FileDataset(config_path="config/datasets/adult.yaml")

# 2. Train classifier
classifier = MLPClassifier(input_dim=14, hidden_dims=[64, 32], output_dim=2)
classifier.fit(train_loader, test_loader, epochs=50)

# 3. Train generative model
flow = MaskedAutoregressiveFlow(input_dim=14, hidden_dims=[64, 64], n_layers=5)
flow.fit(train_loader, test_loader, epochs=100)

# 4. Generate counterfactual
method = PPCEF(gen_model=flow, disc_model=classifier, ...)
result = method.explain(X=instance, y_origin=0, y_target=1, ...)

print(f"Original: {instance}")
print(f"Counterfactual: {result.x_cfs}")
```

[Full Quick Start Tutorial :material-arrow-right:](getting-started/quickstart.md){ .md-button .md-button--primary }

## Method Categories

The library organizes counterfactual methods into three categories:

```mermaid
flowchart TD
    A[Counterfactual Methods] --> B[Local Methods]
    A --> C[Global Methods]
    A --> D[Group Methods]

    B --> B1[PPCEF]
    B --> B2[DICE]
    B --> B3[DiCoFlex]
    B --> B4[...]

    C --> C1[GLOBE-CE]
    C --> C2[AReS]

    D --> D1[ReViCE]
    D --> D2[GLANCE]
    D --> D3[Group GLOBE-CE]
```

| Category | Scope | Best For |
|----------|-------|----------|
| **Local** | Single instance | Individual recourse, debugging |
| **Global** | Entire dataset | Policy insights, systematic patterns |
| **Group** | Subpopulations | Semi-personalized recourse |

## Why Counterfactuals?

Counterfactual explanations answer: *"What would need to change for a different outcome?"*

!!! example "Example"
    **Loan Application Denied**

    *"If your income were $5,000 higher OR your debt-to-income ratio were below 30%, your loan would be approved."*

This provides **actionable recourse** - concrete steps users can take to achieve their desired outcome.

## Architecture

```mermaid
flowchart LR
    subgraph Data
        D1[Datasets]
        D2[Preprocessing]
    end

    subgraph Models
        M1[Classifiers]
        M2[Generative Models]
    end

    subgraph Methods
        CF1[Local CFs]
        CF2[Global CFs]
        CF3[Group CFs]
    end

    subgraph Evaluation
        E1[Metrics]
        E2[Benchmarks]
    end

    D1 --> D2
    D2 --> M1
    D2 --> M2
    M1 --> Methods
    M2 --> Methods
    Methods --> E1
    E1 --> E2
```

## Next Steps

<div class="feature-grid" markdown>

<div class="feature-item" markdown>
:material-download: **Get Started**

Install the library and run your first example.

[Installation Guide](getting-started/installation.md)
</div>

<div class="feature-item" markdown>
:material-book-open-variant: **Learn the Basics**

Understand core concepts and workflows.

[User Guide](user-guide/index.md)
</div>

<div class="feature-item" markdown>
:material-compare: **Compare Methods**

See benchmark results across methods.

[Benchmark Results](benchmarks/results.md)
</div>

<div class="feature-item" markdown>
:material-api: **API Reference**

Detailed documentation for all modules.

[API Docs](reference/index.md)
</div>

</div>

## Citation

If you use this library in your research, please cite:

```bibtex
@software{counterfactuals,
  author = {Furman, Oleksii, Łukasz Lenkiewicz, Marcel Musiałek},
  title = {Counterfactuals: A Python Library for Counterfactual Explanations},
  url = {https://github.com/ofurman/counterfactuals},
  year = {2026}
}
```

## License

This project is licensed under the MIT License.
