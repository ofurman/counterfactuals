# Installation

This guide covers installing the Counterfactuals library and its dependencies.

## Requirements

- **Python 3.10+** (3.11 recommended)
- **Git** (for some dependencies)
- **CUDA** (optional, for GPU acceleration)

## Quick Installation

### Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package manager that handles dependencies efficiently.

```bash
# Clone the repository
git clone https://github.com/ofurman/counterfactuals.git
cd counterfactuals

# Install all dependencies
uv sync
```

### Using pip

```bash
# Clone the repository
git clone https://github.com/ofurman/counterfactuals.git
cd counterfactuals

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in editable mode
pip install -e .
```

## Dependencies

The library has several categories of dependencies:

### Core Dependencies

| Package | Purpose |
|---------|---------|
| `torch` | Deep learning framework |
| `numpy`, `pandas` | Data manipulation |
| `scikit-learn` | ML utilities |
| `scipy` | Scientific computing |

### Deep Learning & Flows

| Package | Purpose |
|---------|---------|
| `tensorflow` | TensorFlow backend |
| `nflows` | Normalizing flows |
| `torchdiffeq` | Neural ODEs |
| `pytorch-tabnet` | TabNet models |

### Counterfactual Methods

| Package | Purpose |
|---------|---------|
| `dice-ml` | DICE method |
| `alibi` | Interpretability methods |
| `cvxpy` | Convex optimization |
| `pyomo`, `omlt` | Optimization modeling |

### Experiment Management

| Package | Purpose |
|---------|---------|
| `hydra-core` | Configuration management |
| `mlflow` | Experiment tracking |

## Development Installation

For development, install with dev dependencies:

```bash
# Using uv
uv sync

# Using pip
pip install -e ".[dev]"
```

This includes:
- `ruff` - Linting and formatting
- `pytest` - Testing framework
- `pre-commit` - Git hooks

### Setting Up Pre-commit Hooks

```bash
pre-commit install
```

## GPU Support

For GPU acceleration with PyTorch:

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

If CUDA is not available, install PyTorch with CUDA support:

```bash
# For CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Verification

Verify your installation:

```python
import counterfactuals

# Check version
print(counterfactuals.__version__)

# Test imports
from counterfactuals.datasets import FileDataset
from counterfactuals.models.classifiers import MLPClassifier
from counterfactuals.cf_methods.local_methods import PPCEF

print("Installation successful!")
```

## Troubleshooting

### Common Issues

!!! warning "nflows installation fails"
    The `nflows` package is installed from a Git repository. Ensure Git is installed:
    ```bash
    git --version
    ```

!!! warning "TensorFlow/PyTorch conflicts"
    If you encounter version conflicts, try installing in a fresh virtual environment:
    ```bash
    python -m venv fresh_env
    source fresh_env/bin/activate
    pip install -e .
    ```

!!! warning "CVXPY solver issues"
    Some CVXPY solvers require additional system dependencies. See [CVXPY installation guide](https://www.cvxpy.org/install/).

### Getting Help

- Check [GitHub Issues](https://github.com/ofurman/counterfactuals/issues)
- Open a new issue with your error message and environment details

## Next Steps

Once installed, continue to the [Quick Start](quickstart.md) guide to generate your first counterfactual explanation.
