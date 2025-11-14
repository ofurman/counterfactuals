# Counterfactuals Library Overview

This document surveys the implementation that lives under `counterfactuals/` and explains how
datasets, preprocessing, models, counterfactual generators, metrics, and pipelines fit together. Use it
as the central reference when you need to navigate the codebase, add a new component, or wire a new
experiment.

## Package Map

| Path | Role | Highlights |
| --- | --- | --- |
| `counterfactuals/datasets` | Data ingestion & wrappers | YAML-driven configs, `FileDataset`, `MethodDataset`, regression variants. |
| `counterfactuals/preprocessing` | Feature engineering | Context-based pipeline, MinMax/Standard scalers, OneHot encoder, Torch dtype conversion. |
| `counterfactuals/dequantization` | Categorical noise & inversion | `GroupDequantizer`, noise registry, wrappers for normalizing flows. |
| `counterfactuals/models` | Generative & discriminative models | `PytorchBase`, mixins, MLP/LogReg/NODE classifiers, RealNVP/NICE/MAF flows. |
| `counterfactuals/cf_methods` | Counterfactual algorithms | Local/global/group methods (PPCEF, DiCE, CEM, AReS, Globe-CE, etc.), shared base classes. |
| `counterfactuals/losses` | Training objectives | Binary/multiclass discriminative losses and regression loss wrappers. |
| `counterfactuals/metrics` | Evaluation suite | Distance functions, plausibility checks, `CFMetrics`, Hydra-configured orchestrator. |
| `counterfactuals/pipelines` | Hydra entry points | `run_*` scripts, reusable nodes for dataset/model/counterfactual orchestration. |
| `counterfactuals/plotting` & `counterfactuals/utils.py` | Visualization & helpers | Matplotlib plots for flows, dict flattening, reporting utilities. |

Related resources:

- Dataset metadata YAMLs live in `config/datasets/`.
- Hydra presets for models/methods sit in `counterfactuals/pipelines/conf/`.
- Run scripts are designed to be executed with `uv run python counterfactuals/pipelines/<script>.py`.

## Data Layer

### Dataset configurations
- Each dataset is described with a YAML file under `config/datasets/`. The schema matches
  `datasets.base.DatasetParameters` (feature list, categorical vs. continuous split, actionability flags,
  target column, optional value mapping, and an optional `samples_keep` cap).
- Target columns are remapped according to `target_mapping` before splitting, enabling string labels in raw CSVs.

### Dataset classes
- `DatasetBase`: loads YAML, resolves CSV paths relative to the repo root, extracts features/labels, and provides
  `split_data` and CV utilities. The base class also exposes feature metadata (categorical indices, actionable
  fields) to downstream components.
- `FileDataset`: extends `DatasetBase` for classification tasks. It balances classes by down-sampling the majority
  class to the minority count, applies optional `samples_keep`, performs the train/test split, and stores
  `X_train`, `X_test`, `y_train`, and `y_test`.
- `RegressionFileDataset`: same concept for regression datasets (no stratification and no balancing).

### MethodDataset wrapper
- Encapsulates a `FileDataset` and an optional preprocessing pipeline. When a pipeline is supplied, it:
  1. Builds a `PreprocessingContext` with raw arrays and feature indices.
  2. Fits/transforms the pipeline to produce processed tensors (and updates categorical/continuous index tracking).
  3. Retains both raw and transformed views so you can invert transformations later.
- Provides PyTorch `train_dataloader`/`test_dataloader` helpers. The train loader can inject Gaussian noise into
  numerical features (`noise_lvl`) to regularize discriminative models.
- `get_cv_splits(n_splits=5)` refits the preprocessing pipeline per fold and updates `X_train`, `X_test`, etc., so
  cross-validation loops (like the PPCEF pipeline) can reuse the same wrapper instance.
- `categorical_features_lists` inspects the fitted `OneHotEncodingStep` to map original categorical features to the
  one-hot column groups needed by the dequantizer and categorical projection steps.

## Preprocessing & Dequantization

### Preprocessing pipeline
- `PreprocessingPipeline` composes named `PreprocessingStep`s. Each step operates on a `PreprocessingContext`,
  making it easy to chain scalers, encoders, and dtype conversions while keeping metadata in sync.
- Stock steps:
  - `MinMaxScalingStep` / `StandardScalingStep` (`preprocessing/scalers.py`): only touch continuous features.
  - `OneHotEncodingStep` (`preprocessing/encoders.py`): encodes categorical columns (train+test) and updates
    index bookkeeping so downstream components know where categorical blocks live after expansion.
  - `TorchDataTypeStep` (`preprocessing/torch_dtype.py`): finalizes arrays as `float32`/`int64` tensors.

### Group dequantization
- Normalizing-flow models expect continuous inputs, so categorical one-hot blocks are smoothed via
  `GroupDequantizer`, which builds one `Dequantizer` per categorical group using the shared
  `processing.GroupTransformer`. Key behaviors:
  - Adds noise sampled from `dequantization.noise.NOISE_REGISTRY` (uniform, Gaussian via sigmoid, or logistic).
  - Applies a logit transform with an `ALPHA` guard to move values to ℝ.
  - Stores per-group `dividers` to undo the transformation via `inverse_transform`.
- `dequantization.utils.DequantizationWrapper` wraps a trained generator so plausibility checks can run on
  original-scale samples without re-training the flow.

## Modeling Stack

### Discriminative models
- All discriminators subclass `models.pytorch_base.PytorchBase`, ensuring a consistent `.fit`, `.predict`, `.save`,
  and `.load` interface. They also mix in classifier/regression helpers where needed.
- Available classifiers live in `counterfactuals/models/classifier/`:
  - `logistic_regression.LogisticRegression`
  - `multilayer_perceptron.MultilayerPerceptron` (configurable hidden sizes/dropout)
  - `node` directory (Neural Oblivious Decision Ensembles).
- Regression variants (`models/regression`) mirror the API.
- `pipelines/nodes/disc_model_nodes.py` provides `create_disc_model`, which instantiates the Hydra-configured model,
  trains it (with checkpoints/patience), evaluates via `classification_report` or `r2_score`, and persists metrics to
  CSV for traceability.

### Generative models
- Models implement `models.generative_mixin.GenerativePytorchMixin`, which defines `predict_log_proba` and
  `sample_and_log_proba`. The main implementations cover:
  - Masked Autoregressive Flow (MAF) variations (`models/generative/maf`).
  - RealNVP, NICE, and CNF-based flows.
  - Kernel density estimators (`kde.py`) for quick baselines.
- Training logic lives in `pipelines/nodes/gen_model_nodes.py`. It instantiates the Hydra-selected architecture,
  trains with configured `batch_size`/`epochs`/`patience`, logs average log-likelihood on train/test, and persists the
  checkpoint to `models/<dataset>/<cf_method>/...`.

### Losses
- Located in `counterfactuals/losses/` and instantiated through Hydra:
  - `BinaryDiscLoss` and `MulticlassDiscLoss` wrap standard PyTorch criteria with convenience defaults.
  - `RegressionLoss` provides MSE-style variants.

## Counterfactual Methods

### Base contracts
- `cf_methods/counterfactual_base.py` defines `BaseCounterfactualMethod`, the shared `.explain` and
  `.explain_dataloader` API, plus the `ExplanationResult` dataclass that bundles optimized deltas, targets, originals,
  and optional logs.
- Mixins (`local_counterfactual_mixin`, `global_counterfactual_mixin`, `group_counterfactual_mixin`) capture method
  families’ shared helpers (e.g., target selection or group constraints).

### Local methods (`cf_methods/local/`)
- Pointwise optimizers such as:
  - `ppcef` and `regression_ppcef`
  - `dice`, `casebased_sace`, `sace`
  - `cem`, `cet`, `cegp`, `c_chvae`, `wach`, `artelt`, `lice`
  - Research additions like `DiCoFlex`
- Most take a trained generator/discriminator pair plus a discriminator loss. They implement `_search_step` style
  loops (see `ppcef.ppcef.PPCEF`) that optimize deltas per instance with penalties for distance, classifier loss, and
  plausibility thresholds.

### Group & global methods
- `cf_methods/global/` contains algorithms that explain model behavior over regions, e.g., `globe_ce` and `ares`.
- `cf_methods/group/` targets cohort-level counterfactuals (e.g., `group_globe_ce`), enforcing shared interventions.

### Configuration
- Every method has a Hydra config under `counterfactuals/pipelines/conf/` (either at the root or under
  `conf/other_methods/`). Set `_target_` to the Python class and add method-specific hyperparameters so pipelines
  can instantiate them generically.

## Evaluation & Metrics

- `metrics/metrics.py` exposes `CFMetrics`, which directly computes coverage, validity, sparsity, actionability,
  plausibility thresholds, LOF scores, isolation forest scores, and several distance metrics (leveraging helper
  functions from `metrics/distances.py` and `metrics/distance.py`).
- `metrics/orchestrator.py` builds on top of `CFMetrics` using a registry defined in `metrics/utils._METRIC_REGISTRY`.
  A Hydra config (`pipelines/conf/metrics/default.yaml`) controls which metrics run — extend it to enable or disable
  metrics per experiment.
- Supporting modules (`metrics/basic_metrics.py`, `plausibility.py`, `validation.py`) provide reusable building
  blocks for checking inputs, computing Hamming/Jaccard/L2 distances, and gating plausibility with log-prob
  thresholds.

## Pipelines & Hydra Integration

- Each `counterfactuals/pipelines/run_*_pipeline.py` script is a Hydra entry point that combines dataset loading,
  preprocessing, model training/loading, counterfactual search, and metric computation for a specific method family
  (PPCEF, DICE, CET, WACH, AReS, Globe-CE, etc.).
- Shared orchestration nodes (`pipelines/nodes/`) encapsulate repeated steps:
  - `helper_nodes.set_model_paths` standardizes output folders (`models/<dataset>/<cf_method>/fold_<n>`).
  - `gen_model_nodes` / `disc_model_nodes` manage model instantiation, training, evaluation, and checkpointing.
  - `counterfactuals_nodes` contains generic helpers for building the CF method, computing log-prob thresholds, and
    streaming dataloaders into `explain_dataloader`.
- Hydra config tree (`counterfactuals/pipelines/conf/`) includes:
  - `gen_model/*.yaml` and `disc_model/*.yaml` presets describing architectures.
  - Method-level configs (`ppcef_config.yaml`, `dice_config.yaml`, etc.) that compose dataset settings, model presets,
    and `counterfactuals_params`.
  - `metrics/default.yaml` to toggle metrics orchestrator outputs.
- Run pipelines with `uv run python counterfactuals/pipelines/<script>.py <override>=<value>`. Example:

  ```bash
  uv run python counterfactuals/pipelines/run_ppcef_pipeline.py \
    dataset.config_path=config/datasets/heloc.yaml \
    disc_model.model=disc_model/mlp_large \
    counterfactuals_params.target_class=1
  ```

  Outputs land under `models/` (models, metrics, counterfactual CSVs) and standard Hydra logs under `outputs/`.

## Visualization & Utilities

- `counterfactuals/utils.py` bundles quick Matplotlib helpers for visualizing flows, optimization landscapes, and
  counterfactual trajectories (`plot_x_point`, `plot_model_distribution`, `plot_loss_space`). It also exposes dict
  flattening helpers used when exporting metrics or Hydra configs.
- `counterfactuals/plotting` hosts higher-level visualization scripts (e.g., `counterfactual_visualization.py`) for
  notebook/report usage.

## Extension Playbooks

1. **Add a new dataset**
   - Drop a YAML file under `config/datasets/` with features, metadata, and the target column.
   - Reference it in Hydra (`dataset.config_path=...`) or create a new method config that sets it as the default.
   - If special preprocessing is needed, subclass `PreprocessingStep` and add it to the `PreprocessingPipeline`.

2. **Add a discriminative or generative model preset**
   - Implement the PyTorch module under `counterfactuals/models/<classifier|generative>/`.
   - Create a Hydra config under `counterfactuals/pipelines/conf/disc_model/` or `conf/gen_model/`, setting
     `_target_` to the new class and exposing tunable hyperparameters.

3. **Introduce a counterfactual method**
   - Subclass `BaseCounterfactualMethod` (and the appropriate mixin), implement `.explain_dataloader`, and expose any
     knobs via the constructor.
   - Add a Hydra config (e.g., `conf/<method>_config.yaml`) setting `counterfactuals_params.cf_method._target_`.
   - Either wire it into an existing pipeline or add a dedicated `run_<method>_pipeline.py` following the existing
     scripts as templates.

4. **Register a metric**
   - Implement a metric class with a `required_inputs()` declaration and register it in
     `metrics/utils._METRIC_REGISTRY`.
   - Reference the metric name in `pipelines/conf/metrics/default.yaml` (or a method-specific override).

5. **Add a pipeline**
   - Use `run_ppcef_pipeline.py` as a blueprint: parse Hydra config, load dataset, fit models (optionally relabel the
     dataset), run counterfactual search, evaluate metrics, and save CSVs. Reuse the node helpers where possible.

## Development Workflow & Standards

The rules spelled out in `AGENTS.md` apply everywhere:

- Target Python 3.11, prefer modern typing features (`list[int]`, `typing.Self`, `TypedDict`, `Literal`, dataclasses,
  Enums).
- Use Ruff for linting/formatting (100-character lines, `uv run ruff check --fix` + `uv run ruff format`).
- All public functions/classes/modules need Google-style docstrings and complete type hints.
- Use the `logging` module (structured messages) — avoid `print` in library code.
- Manage dependencies with uv (`uv sync`, `uv add`, `uv run python ...`). Avoid invoking `python` directly in docs or
  scripts.
- Keep changes small and well-scoped; update or add tests under `tests/` when behavior changes.
- When contributing pipeline changes, describe which `uv run python counterfactuals/pipelines/...` command you used
  and where outputs were written.

Couple this guide with `docs/ppcef_pipeline.md` (method-specific deep dive) and `README.md` (project overview) for a
complete mental model of the repository.
