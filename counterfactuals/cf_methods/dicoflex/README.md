# DiCoFlex

**Diverse Counterfactual Explanations via Normalizing Flows**

DiCoFlex generates diverse, sparsity-constrained counterfactual explanations using conditional normalizing flows. Unlike optimization-based methods (e.g., PPCEF), DiCoFlex trains a conditional flow model on nearest-neighbor counterfactual pairs, then samples counterfactuals in a single forward pass.

## How It Works

1. **Dataset construction** -- For each factual point, find the `n_nearest` points from each opposing class using p-norm distance with feature masking. Optionally filter by classifier confidence threshold.
2. **Flow training** -- Train a conditional Masked Autoregressive Flow (MAF) where the context is `[factual_point | target_class_onehot | mask | p_norm]`. The flow learns to map a base distribution to the counterfactual distribution conditioned on the factual point and target class.
3. **Generation** -- Sample from the trained flow given a factual point and desired target class. Each sample is a counterfactual explanation.

## Module Structure

```
dicoflex/
    __init__.py          # Exports DiCoFlex class
    dicoflex.py          # DiCoFlex facade (fit / generate / save / load)
    dataset.py           # MulticlassCounterfactualDataset, MulticlassCounterfactualWrapper
    training.py          # Training loop for conditional flow model
    generation.py        # Counterfactual sampling from trained model
    visualization.py     # 2D visualization with decision boundaries
    utils.py             # Batch transforms, inverse data transforms
```

## Quick Start

### Using the facade class

```python
from counterfactuals.cf_methods.dicoflex import DiCoFlex
from counterfactuals.generative_models.maf import MaskedAutoregressiveFlow

dicoflex = DiCoFlex(
    flow_model_class=MaskedAutoregressiveFlow,
    hidden_features=64,
    num_layers=5,
    num_blocks_per_layer=2,
)

# Train
dicoflex.fit(
    X=dataset.X_train,
    y=dataset.y_train,
    masks=masks,             # Feature importance masks (n_masks x n_features)
    p_values=[1e-2, 2.0],   # p-norms for distance computation
    n_nearest=32,
    num_epochs=10000,
    patience=50,
    save_dir="results/adult",
)

# Generate counterfactuals
cfs, log_probs = dicoflex.generate(
    factual_points=X_test,
    target_class=1,
    p_value=2.0,
    mask=np.array([1.0]),    # One-hot mask index
    n_samples=100,
    temperature=0.8,
)
# cfs.shape = (n_factual, n_samples, n_features)
```

### Using the CLI script

```bash
# Run on Adult dataset
python counterfactuals/examples/train_dicoflex.py --adult

# Run on all DCENF datasets
python counterfactuals/examples/train_dicoflex.py --adult --bank --gmc --lending --default
```

## Supported Datasets

Works with any dataset from `counterfactuals/datasets/DCENF/`:

| Dataset | Numerical | Categorical | Classes |
|---------|-----------|-------------|---------|
| Adult   | 4         | 8           | 2       |
| Bank    | 7         | 9           | 2       |
| GMC     | 7         | 3           | 2       |
| Default | 14        | 9           | 2       |
| LendingClub | 8    | 4           | 2       |

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_nearest` | 16 | Nearest counterfactual neighbors per factual point |
| `p_values` | `[2.0]` | p-norms for distance computation |
| `masks` | -- | Feature importance masks (values near 0 = immutable) |
| `prob_threshold` | 0.0 | Classifier confidence threshold for filtering neighbors |
| `noise_level` | 0.01 | Gaussian noise for training augmentation |
| `temperature` | 0.8 | Sampling temperature (higher = more diverse) |
| `patience` | 50 | Early stopping patience (epochs without improvement) |
