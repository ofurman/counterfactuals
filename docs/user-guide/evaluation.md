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

@register_metric("my_metric")
class MyMetric(Metric):
    name = "my_metric"

    def required_inputs(self):
        return {"X_cf", "X_test"}

    def __call__(self, X_cf, X_test, **kwargs):
        # Your computation
        return score
```

Metrics are registered on import. Place new metrics in `counterfactuals/metrics/` and
ensure the module is imported in `counterfactuals/metrics/__init__.py`, then add the
metric name to a metrics config file.

## Configuring Which Metrics Run

The default list of computed metrics lives in
`counterfactuals/pipelines/conf/metrics/default.yaml` under `metrics_to_compute`.
To use a different set, create another YAML file with the same structure and pass its
path to `MetricsOrchestrator(metrics_conf_path=...)` or `evaluate_cf(metrics_conf_path=...)`.

Example:

```yaml
metrics_to_compute:
  - "validity"
  - "sparsity"
  - "my_metric"
```

## Custom Metrics in Pipelines

Some pipelines add extra, method-specific metrics on top of the registered metrics.
For example, `run_pumal_pipeline.py` defines a custom `calculate_metrics` helper that
calls `evaluate_cf_for_pumal(...)`. That function first delegates to `evaluate_cf`
(which runs the standard MetricsOrchestrator on the configured list), then augments
the result with PUMAL-specific statistics. Conceptually it looks like this:

```python
def evaluate_cf_for_pumal(
    disc_model,
    gen_model,
    X_cf,
    model_returned,
    continuous_features,
    categorical_features,
    X_train,
    y_train,
    X_test,
    y_test,
    y_target,
    median_log_prob,
    S_matrix=None,
    D_matrix=None,
    metrics_conf_path="counterfactuals/pipelines/conf/metrics/group_metrics.yaml",
):
    cf_group_ids = None
    if S_matrix is not None:
        cf_group_ids = np.argmax(S_matrix, axis=1)

    metrics = evaluate_cf(
        disc_model=disc_model,
        gen_model=gen_model,
        X_cf=X_cf,
        model_returned=model_returned,
        continuous_features=continuous_features,
        categorical_features=categorical_features,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        y_target=y_target,
        median_log_prob=median_log_prob,
        cf_group_ids=cf_group_ids,
        metrics_conf_path=metrics_conf_path,
    )

    if S_matrix is not None:
        metrics["cf_belongs_to_group"] = np.mean(np.any(S_matrix == 1.0, axis=1))
        metrics["K_vectors"] = (S_matrix.sum(axis=0) != 0).sum()

    if D_matrix is not None and D_matrix.shape[0] > 1:
        metrics["distance_to_centroid_mean"] = np.linalg.norm(
            D_matrix - np.mean(D_matrix, axis=0), axis=1
        ).mean()

    return metrics
```

```python
import logging

logger = logging.getLogger(__name__)

def calculate_metrics(
    gen_model,
    disc_model,
    Xs_cfs,
    model_returned,
    categorical_features,
    continuous_features,
    X_train,
    y_train,
    X_test,
    y_test,
    median_log_prob,
    y_target=None,
    S_matrix=None,
    D_matrix=None,
):
    metrics = evaluate_cf_for_pumal(
        gen_model=gen_model,
        disc_model=disc_model,
        X_cf=Xs_cfs,
        model_returned=model_returned,
        categorical_features=categorical_features,
        continuous_features=continuous_features,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        median_log_prob=median_log_prob,
        y_target=y_target,
        S_matrix=S_matrix,
        D_matrix=D_matrix,
        metrics_conf_path="counterfactuals/pipelines/conf/metrics/group_metrics.yaml",
    )

    logger.info("Metrics:\n%s", metrics)
    return metrics
```

If you need additional custom values, add them after the base metrics call:

```python
def calculate_metrics_with_custom_values(
    gen_model,
    disc_model,
    Xs_cfs,
    model_returned,
    categorical_features,
    continuous_features,
    X_train,
    y_train,
    X_test,
    y_test,
    median_log_prob,
    y_target=None,
    extras=None,
):
    metrics = evaluate_cf(
        gen_model=gen_model,
        disc_model=disc_model,
        X_cf=Xs_cfs,
        model_returned=model_returned,
        categorical_features=categorical_features,
        continuous_features=continuous_features,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        median_log_prob=median_log_prob,
        y_target=y_target,
        metrics_conf_path="counterfactuals/pipelines/conf/metrics/default.yaml",
    )

    # Your custom code here
    if extras is not None:
        metrics["my_custom_metric"] = extras["my_custom_metric"]

    logger.info("Metrics:\n%s", metrics)
    return metrics
```

## Next Steps

- [Benchmark Results](../benchmarks/results.md) - Compare methods
