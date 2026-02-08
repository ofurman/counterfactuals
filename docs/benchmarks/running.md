# Running Benchmarks

How to reproduce and extend benchmark experiments.

## Quick Start

```bash
# Run single method benchmark
python -m cel.pipelines.run_ppcef_pipeline \
    dataset.config_path=config/datasets/adult.yaml

# Run with multiple seeds
for seed in 0 1 2 3 4; do
    python -m cel.pipelines.run_ppcef_pipeline \
        random_state=$seed
done
```

## Multi-Dataset Benchmark

```bash
# Run across all classification datasets
for dataset in adult compas german_credit heloc; do
    python -m cel.pipelines.run_ppcef_pipeline \
        dataset.config_path=config/datasets/${dataset}.yaml
done
```

## Comparing Methods

```bash
# Run multiple methods on same dataset
python -m cel.pipelines.run_ppcef_pipeline
python -m cel.pipelines.run_dice_pipeline
python -m cel.pipelines.run_globe_ce_pipeline
```

## Viewing Results

```python
import mlflow

# List all runs
runs = mlflow.search_runs()
print(runs[["run_id", "params.method", "metrics.validity"]])

# Compare methods
runs.groupby("params.method")["metrics.validity"].mean()
```

## Custom Benchmark Configuration

Create a custom config:

```yaml
# pipelines/conf/benchmark.yaml
defaults:
  - _self_
  - override hydra/sweeper: basic

hydra:
  sweeper:
    params:
      dataset.config_path:
        - config/datasets/adult.yaml
        - config/datasets/compas.yaml
      random_state: range(0, 5)
```

Run sweep:

```bash
python -m cel.pipelines.run_ppcef_pipeline \
    --multirun \
    --config-name=benchmark
```

## Adding New Methods to Benchmarks

1. Create pipeline in `counterfactuals/pipelines/`
2. Add configuration in `pipelines/conf/`
3. Run benchmark with same datasets
