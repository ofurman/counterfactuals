# Running Pipelines

Execute end-to-end experiments using Hydra configuration system.

## Overview

Pipelines automate the complete workflow:
1. Load dataset
2. Train/load models
3. Generate counterfactuals
4. Compute metrics
5. Log results

## Running a Pipeline

```bash
# Run PPCEF pipeline
python -m cel.pipelines.run_ppcef_pipeline

# Override configuration
python -m cel.pipelines.run_ppcef_pipeline \
    dataset.config_path=config/datasets/compas.yaml \
    counterfactuals_params.epochs=200
```

## Configuration Structure

```yaml
# pipelines/conf/config.yaml
defaults:
  - gen_model: large_maf
  - disc_model: mlp
  - metrics: default

dataset:
  _target_: counterfactuals.datasets.FileDataset
  config_path: config/datasets/adult.yaml

gen_model:
  train_model: true
  epochs: 200
  lr: 0.0001

disc_model:
  train_model: true
  epochs: 100
  lr: 0.001

counterfactuals_params:
  epochs: 100
  lr: 0.01
  alpha: 1.0
  beta: 0.5
```

## Available Pipelines

| Pipeline | Method |
|----------|--------|
| `run_ppcef_pipeline` | PPCEF |
| `run_dice_pipeline` | DICE |
| `run_globe_ce_pipeline` | GLOBE-CE |
| `run_rppcef_pipeline` | ReViCE |
| ... | ... |

## MLflow Logging

Results are automatically logged to MLflow:

```python
import mlflow

# View logged runs
mlflow.search_runs()
```

## Creating Custom Pipelines

See existing pipelines in `counterfactuals/pipelines/` for examples.
