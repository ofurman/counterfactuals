# Metrics CSV schema

One `cf_metrics_SimpleMLPClassifier.csv` per (tag, seed, dataset, method), single data row.
Written by `run_pipeline` in each `run_*_traintest_pipeline.py`.

## Timing columns

| Column | Written at | DiCE | CCHVAE | DiCoFlex |
|---|---|---|---|---|
| `disc_train_time` | around `create_disc_model` | SimpleMLP fit | SimpleMLP fit | SimpleMLP fit |
| `gen_train_time` | around `create_gen_model` / `train_dicoflex_generator` | MAF density model (metrics only) | MAF density model (metrics only) | **the DiCoFlex generator** |
| `cf_model_train_time` | **added in Stage 1** | `0.0` (training-free) | VAE fit inside `CCHVAE.__init__` | `0.0` (counted in `gen_train_time`) |
| `cf_search_time` | around CF generation | `exp.generate_counterfactuals` | `get_counterfactuals_without_check` x `num_counterfactuals` | flow sampling |
| `seed` | end of `run_pipeline` | `cfg.experiment.get("seed", 42)` | `cfg.experiment.seed` | `cfg.experiment.seed` |

Line references as of commit 6a1126e:

- DiCE: `df_metrics` block at `run_dice_traintest_pipeline.py:339-342`;
  `gen_train_start` at 303; `cf_search_time` at 174.
- CCHVAE: `df_metrics` at `run_cchvae_traintest_pipeline.py:258-261`; `disc_train_start` 216,
  `gen_train_start` 225; `CCHVAE(...)` construction 108; `time_start` 119.
- DiCoFlex: `df_metrics` at `run_dicoflex_traintest_pipeline.py:509-512`; `disc_train_time`
  236, `gen_train_time` 293.

## The CCHVAE gap (why Stage 1 exists)

`CCHVAE.__init__` → `_load_vae` trains the VAE when `vae_params.train: true`
(`cchvae_traintest_config.yaml:64`, 10 epochs, batch 32, layers `[input, 64, 32, 16]`).
Construction happens at line 108, **before** `time_start` at line 119, so the cost is in
neither `gen_train_time` nor `cf_search_time`. It is currently unmeasured.

## Quality metrics

From `counterfactuals/pipelines/conf/metrics/default.yaml`:

```
coverage, validity, actionability, sparsity,
proximity_euclidean_hamming, proximity_euclidean_jaccard, proximity_l1_jaccard,
proximity_mad_jaccard, proximity_l2_jaccard, proximity_mad_hamming,
prob_plausibility, log_density_cf, log_density_test,
lof_scores_cf, lof_scores_test,
isolation_forest_scores_cf, isolation_forest_scores_test,
number_of_instances
```

`number_of_instances` is the factual count for that cell and doubles as the weight for
weighted means and the denominator for `inference_time_per_factual`.

## Target class and factual counts

Factuals are selected as `y_test != target_class` in all three pipelines
(`run_dice_traintest_pipeline.py:122-124`, `run_cchvae_traintest_pipeline.py:91-93`,
`run_dicoflex_traintest_pipeline.py:332-333`). Config defaults:

| Config | `target_class` |
|---|---|
| `dice_traintest_config.yaml:45` | 0 |
| `cchvae_traintest_config.yaml:45` | 0 |
| `dicoflex_traintest_config.yaml:48` | **1** |

So the `seeds` tag has DiCoFlex on the complement of DiCE/CCHVAE. Observed complementary
counts: adult 3674 vs 6326, bank 2746 vs 7254. The `seeds-tc0` tag overrides DiCoFlex with
`++counterfactuals_params.target_class=0` so all three share one query set.

Consequence for the report: absolute `cf_search_time` is comparable only within a target
class; `inference_time_per_factual` is comparable across all rows.

## Aggregation conventions inherited from `scripts/calculate_metrics.py`

- Rows with `validity == 0` are excluded (lines 76-91) so a degenerate seed cannot pull a cell
  to 0.00.
- Quality metrics use `number_of_instances`-weighted means (lines 112-135).
- Columns whose mean or std is non-finite are dropped with a log line (lines 137-155).
- Presentation is `f"{mean:.2f} ± {std:.2f}"` (`format_mean_std`, line 32).

**Do not** weight timing columns by `number_of_instances`. A duration is per run, not per
instance; times aggregate as an unweighted mean over seeds.
