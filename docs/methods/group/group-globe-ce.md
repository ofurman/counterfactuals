# Group GLOBE-CE

Group GLOBE-CE is a pipeline-level extension of [GLOBE-CE](../global/globe-ce.md) that
partitions the test set into clusters and fits an independent GLOBE-CE search per
cluster. It is not a standalone class under `cel.cf_methods`; it is implemented by
the `GroupGLOBECEPipelineRunner` in `cel/pipelines/runners/group_globe_ce_runner.py`,
which delegates the actual counterfactual search to `cel.cf_methods.GLOBE_CE`.

## How it works

1. Filter the test set to instances whose predicted class is not the target class.
2. Compute per-feature bin widths from the unscaled test data (10 bins by default).
3. Run KMeans on the scaled inputs with `counterfactuals_params.n_clusters` clusters.
4. For each cluster, instantiate a fresh `GLOBE_CE(predict_fn, dataset, X=cluster_X,
   bin_widths=bin_widths, target_class=target_class)` and call `.explain(y_origin,
   y_target)`.
5. Re-scale the per-cluster counterfactuals back to model space and align them with
   the original factual ordering.

Each cluster therefore receives its own global translation vector(s), which trades
some global parsimony for tighter coverage of locally homogeneous subgroups.

## Running the pipeline

```bash
uv run python -m cel.pipelines.run_group_globe_ce_pipeline \
  --config-path ./conf --config-name group_globe_ce_config
```

The Hydra config lives at
`cel/pipelines/conf/group_globe_ce_config.yaml`. The relevant section:

```yaml
counterfactuals_params:
  target_class: 0
  batch_size: 4096
  log_prob_quantile: 0.25
  n_clusters: 4
```

## Parameters

| Parameter | Source | Description |
|-----------|--------|-------------|
| `n_clusters` | `counterfactuals_params.n_clusters` | Number of KMeans clusters to fit on the filtered test set. |
| `target_class` | `counterfactuals_params.target_class` | Desired prediction class for all counterfactuals. |
| `bin_widths` | computed from data | Per-feature bin widths (10 bins) reused across clusters; see `cel.pipelines.runners.globe_ce_runner.compute_bin_widths`. |
| `dataset` | `dataset._target_` | `MethodDataset` wrapping a `FileDataset`. |

GLOBE-CE-specific knobs (`p`, `monotonicity`, `delta_init`, etc.) are not exposed by
the current Group GLOBE-CE config; the runner instantiates `GLOBE_CE` with defaults
plus the bin widths and target class.

## When to use

- The test population has clearly separable subgroups and a single global translation
  underfits.
- You still want the interpretability of a small number of fixed translations rather
  than per-instance counterfactuals.
- Cluster count is low (defaults to 4) — for higher granularity, prefer a per-instance
  local method or [TCREx](tcrex.md).

## API Reference

The runner that implements Group GLOBE-CE:

::: cel.pipelines.runners.group_globe_ce_runner.GroupGLOBECEPipelineRunner

The underlying counterfactual search is GLOBE-CE — see its
[reference page](../global/globe-ce.md).
