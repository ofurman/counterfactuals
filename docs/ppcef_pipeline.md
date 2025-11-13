# PPCEF Pipeline Guide

This document explains how the PPCEF workflow is wired together inside
`counterfactuals/pipelines/run_ppcef_pipeline.py` and describes the contribution
expectations for future development. The goal is to provide enough context to
follow the end-to-end data flow, tune experiments through Hydra, and extend the
library without reverse engineering the code base.

## End-to-End Flow (run_ppcef_pipeline.py)

The `main` function in `run_ppcef_pipeline.py` is the orchestration entry point.
The high-level lifecycle for each cross-validation fold is:

1. **Configuration (Hydra)**  
   - `ppcef_config.yaml` pulls in defaults for the PPCEF method, MAF generator,
     MLP discriminator, dataset, and training hyperparameters.  
   - CLI overrides such as `dataset.config_path=config/datasets/heloc.yaml` or
     `gen_model=train_model=false` can be passed when launching the pipeline.

2. **Dataset loading and preprocessing**  
   - `FileDataset` reads the YAML dataset config (paths, feature metadata, target,
     actionable fields).  
   - `MethodDataset` wraps the file dataset and plugs in the preprocessing
     pipeline:
     - `MinMaxScalingStep` rescales continuous features to `[0, 1]`.  
     - `OneHotEncodingStep` expands categorical features while tracking the index
       groups required later for categorical intervals.  
     - `TorchDataTypeStep` ensures tensors land on the expected dtype for model
       training/inference.  
   - `MethodDataset.get_cv_splits(5)` refits the preprocessing pipeline for every
     fold and updates the in-memory arrays (note that after the loop finishes the
     dataset contains the last fold’s data).

3. **Model preparation**  
   - `set_model_paths` (helper_nodes) creates an experiment folder under
     `models/<dataset>/<cf-method>/[fold_X]` and returns deterministic file names
     for discriminator, generator, and results. When `experiment.relabel_with_disc_model`
     is true, the generator checkpoints are tagged with the discriminator name so
     you do not mix incompatible models.  
   - `create_disc_model` builds the discriminator defined in
     `conf/disc_model/*.yaml`, trains it if `disc_model.train_model` is true, and
     saves both the model and an evaluation report
     (`eval_disc_model_<name>.csv`). When relabeling is enabled the dataset labels
     are replaced with discriminator predictions before training the generator.  
   - `GroupDequantizer` is fitted on `dataset.X_train` to add controlled noise and
     logit-transform categorical one-hot blocks. The generator is built and
     trained through `create_gen_model`, which internally logs train/test
     likelihoods and persists the checkpoint (`gen_model_<name>.pt`). After
     training, both `X_train`/`X_test` are transformed with the fitted
     dequantizer so likelihood thresholds and counterfactual search operate in
     the same dequantized space.

4. **Counterfactual search**  
   - `search_counterfactuals` filters out the target class (e.g., label `0`) so
     we only attempt flips for examples that do **not** already belong to the
     desired class.  
   - A `PPCEF` instance is created with the generator, discriminator, and
     discriminator loss from the config (`MulticlassDiscLoss` by default).  
   - Log-probability thresholds are computed through
     `gen_model.predict_log_prob(train_loader)` and selecting the configured
     quantile (default `0.25`). This bounds the plausibility of proposed
     counterfactuals.  
   - A TensorDataset supplies the filtered records to `PPCEF.explain_dataloader`;
     runtime parameters (`epochs`, `lr`, `alpha`, `alpha_s`, `alpha_k`, `patience`,
     categorical intervals) are drawn from `counterfactuals_params`. The helper
     functions `get_categorical_intervals` and `apply_categorical_discretization`
     keep categorical blocks consistent after optimization.  
   - Outputs include:
     - `counterfactuals_<cf_method>_<disc_model>.csv`: dequantized counterfactuals
       stored in `save_folder`.  
     - `cf_search_time`: averaged runtime for transparency and reporting.

5. **Evaluation & reporting**  
   - `DequantizationWrapper` wraps the generator so plausibility checks can be
     run in the original data space.  
   - `calculate_metrics` delegates to `counterfactuals.metrics.evaluate_cf`,
     providing categorical/continuous feature indices, the original data, and the
     computed log-probability median. The resulting dictionary is dumped to
     `cf_metrics_<disc_model>.csv` alongside the search time. Metrics commonly
     include validity, plausibility, proximity, diversity, and coverage.

## Configuration Reference

Key knobs from `counterfactuals/pipelines/conf/ppcef_config.yaml`:

| Section | Purpose | Common overrides |
| ------ | ------- | ---------------- |
| `experiment` | Sets the root folder for artifacts and whether labels are relabeled with the discriminator before generator training. | `experiment.output_folder=/tmp/experiments` |
| `dataset` | Points to a YAML config under `config/datasets`. Defines feature names, categorical vs. continuous split, actionable flags, and target column. | `dataset.config_path=config/datasets/heloc.yaml` |
| `disc_model` | Controls discriminator training (epochs, learning rate, patience). The underlying architecture and hidden sizes live under `conf/disc_model/*.yaml`. | `disc_model.model=disc_model/mlp_large` |
| `gen_model` | Same pattern for the normalizing flow (MAF, RealNVP, etc.) under `conf/gen_model`. Includes training settings and optional noise injection. | `gen_model=model=gen_model/small_maf` |
| `counterfactuals_params` | PPCEF-specific search hyperparameters plus utility objects (loss, target class, categorical handling). | `counterfactuals_params.target_class=1` |

Hydra lets you compose these pieces from the CLI. Example:

```bash
uv run python counterfactuals/pipelines/run_ppcef_pipeline.py \
  dataset.config_path=config/datasets/heloc.yaml \
  disc_model.model=disc_model/mlp_large \
  counterfactuals_params.target_class=1
```

Hydra will create a timestamped working directory under `.hydra`, but
`set_model_paths` keeps experiment results under `models/<dataset>/ppcef/`.

## Running the Pipeline Locally

1. **Setup**  
   - Install uv if you have not already (`pip install uv`).  
   - Sync dependencies: `uv sync`.  
   - (Optional) Activate the venv created by uv (`source .venv/bin/activate`).

2. **Execute the PPCEF run**  
   - Use `uv run python counterfactuals/pipelines/run_ppcef_pipeline.py`.  
   - Add Hydra overrides to switch datasets, architecture presets, or
     hyperparameters as needed.  
   - The script forces CPU execution (`CUDA_VISIBLE_DEVICES=-1`). To use a GPU,
     modify `main` accordingly or run a custom entry point.

3. **Inspect artifacts**  
   - `models/<dataset>/<cf_method>/fold_<n>/counterfactuals_*.csv` – generated
     counterfactuals per discriminator/gen pair.  
   - `cf_metrics_*.csv` – evaluation metrics per fold.  
   - `eval_disc_model_*.csv` – discriminator performance snapshot.  
   - Hydra logs (under `outputs/<timestamp>` by default) capture the structured
     logging stream.

4. **Common variations**  
   - Skip training when you already have checkpoints:
     `disc_model.train_model=false gen_model.train_model=false`. Place your
     `.pt` files where `set_model_paths` expects them.  
   - Restrict to numeric features by disabling categorical handling:
     `counterfactuals_params.use_categorical=false` (skips discretization).  
   - Adjust plausibility strictness through `counterfactuals_params.log_prob_quantile`.

## Extending the Library

When introducing a new dataset, model, or counterfactual method:

- **Datasets**: Add a YAML file under `config/datasets/` describing feature sets,
  actionability, and target column. Hook it up via
  `dataset.config_path=config/datasets/<your_file>.yaml`.
- **Models**: Drop a new config under `conf/disc_model/` or `conf/gen_model/` that
  points to the fully qualified class path and exposes tunable parameters. The
  Hydra config system makes the new preset immediately selectable from the CLI.
- **Methods**: Implement the method under `counterfactuals/cf_methods/`, ensure it
  matches the PPCEF interface (`explain_dataloader`, etc.), and register it with a
  new Hydra config (e.g., `conf/<method>_config.yaml`). You can then clone
  `run_ppcef_pipeline.py` or add branching inside the existing script if the
  pipeline is similar.

## Contribution Guidelines

The project follows the rules codified in `AGENTS.md`. Highlights for day-to-day
changes:

- **Environment & tooling**  
  - Target Python 3.11; prefer modern typing features (`list[int]`, `typing.Self`,
    `TypedDict`, `Literal`, dataclasses, and enums where appropriate).  
  - Manage dependencies with uv. Run code as `uv run <command>` instead of calling
    `python` directly.  
  - Keep Ruff as the source of truth for linting/formatting
    (`uv run ruff check --fix` and `uv run ruff format`).

- **Style & quality**  
  - Follow PEP 8 with a 100-character limit (Ruff enforces this).  
  - Public modules, classes, and functions need Google-style docstrings and full
    type hints.  
  - Use the `logging` module for diagnostics; avoid `print` in library code.  
  - Keep changesets focused and well-scoped so they are easy to review.

- **Testing & verification**  
  - Add or update tests under `tests/` when adding features or fixing bugs.  
  - Run `uv run pytest` plus relevant integration scripts before sending patches.  
  - For pipeline work, capture a short note in PRs describing which command you
    executed (including Hydra overrides) and where the artifacts landed.

- **Documentation**  
  - Update `docs/` or in-line docstrings whenever you change behavior,
    configuration names, or outputs.  
  - Prefer markdown tables or concise code samples over large prose blocks when
    explaining new options.  
  - Keep diagrams or plots (if any) under `docs/` or `notebooks/` and reference
    them from the README.

- **Process**  
  - Branch per feature/bugfix, keep commits small, and state the intent clearly in
    commit messages.  
  - When adding dependencies, update both `pyproject.toml` and `uv.lock` via
    `uv add`.  
  - Treat Ruff warnings as CI failures; do not silence linters unless there is a
    strong justification documented in-code.

Following these steps ensures the PPCEF pipeline remains reproducible and that
new contributors can confidently extend the library.
