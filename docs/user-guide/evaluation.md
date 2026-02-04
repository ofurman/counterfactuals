# Evaluating Results

Assess the quality of generated counterfactuals using built-in metrics.

## Using MetricsOrchestrator

```python
from counterfactuals.metrics import MetricsOrchestrator

# Initialize with desired metrics
orchestrator = MetricsOrchestrator(
    metrics=[
        "validity",
        "proximity_l2",
        "sparsity",
        "plausibility"
    ],
    gen_model=flow
)

# Compute metrics
scores = orchestrator.compute(
    x_cfs=result.x_cfs,
    x_origs=result.x_origs,
    y_targets=result.y_cf_targets,
    classifier=classifier
)

for metric, value in scores.items():
    print(f"{metric}: {value:.4f}")
```

## Available Metrics

### Validity Metrics
- `validity` - Proportion achieving target class
- `coverage` - Proportion of successful generations

### Distance Metrics
- `proximity_l2` - Euclidean distance
- `proximity_l1` - Manhattan distance
- `proximity_mad` - Mean Absolute Deviation

### Sparsity Metrics
- `sparsity` - Average features changed

### Plausibility Metrics
- `plausibility` - Log-likelihood under flow
- `lof_score` - Local Outlier Factor
- `isolation_score` - Isolation Forest score

### Diversity Metrics
- `diversity` - Pairwise distance between CFs

## Custom Metrics

```python
from counterfactuals.metrics import Metric, register_metric

@register_metric
class MyMetric(Metric):
    name = "my_metric"

    def required_inputs(self):
        return {"x_cfs", "x_origs"}

    def __call__(self, x_cfs, x_origs, **kwargs):
        # Your computation
        return score
```

## Next Steps

- [Benchmark Results](../benchmarks/results.md) - Compare methods
