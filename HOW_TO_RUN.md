# How to reproduce the experiments on this branch

This branch (`tabdce`, on top of `develop`) adds:

- new pipelines: TabDCE, CCHVAE/DiCE/DiCoFlex train-test, DiCoFlex/DiCE/CCHVAE pairwise, CADEX, CE-Flow, etc.
- shell drivers in the repo root that loop pipelines over datasets
- a `scripts/` toolbox for computing metrics and rendering result tables (markdown + LaTeX)

The commands below reproduce the same outputs that I generate locally. Run everything from the repo root and use `uv` (plain `python` is not available on macOS in this project).

---

## 0. Prerequisites

```bash
# environment
uv sync             # installs everything pinned in uv.lock / pyproject.toml

# data must already be split per dataset
ls data_train_test_val
# adult/  bank/  default/  gmc/  lending-club/   each has train.csv + test.csv + config.json
```

Dataset configs live in `config/datasets/`. Train/test runs use the `*_split.yaml` files (e.g. `adult_split.yaml`, `lending_club_split.yaml`); pairwise (CV) runs use the non-split yamls (`adult_census.yaml`, `lending_club.yaml`, …).

---

## 1. Train + generate counterfactuals

There are two execution flavours: **train-test** (single train.csv / test.csv split, fast) and **pairwise** (5-fold CV, used for the metric tables).

### 1a. Train-test pipelines (CCHVAE, DiCE, DiCoFlex)

```bash
# CCHVAE (currently configured for adult; edit DATASETS/PIPELINES inside the script for others)
./run_traintest_experiments.sh

# CCHVAE + DiCE on gmc and lending-club
./run_traintest_experiments2.sh

# DiCoFlex with monotonic constraints (age may only INCREASE, etc.)
./run_dicoflex_traintest_experiments.sh    # bank, default
./run_dicoflex_traintest_experiments2.sh   # gmc, lending-club
```

What they do under the hood:

```bash
uv run python -m counterfactuals.pipelines.run_cchvae_traintest_pipeline \
    disc_model=simple_mlp \
    disc_model.train_model=true \
    gen_model.train_model=true \
    dataset.config_path=config/datasets/adult_split.yaml \
    dataset.train_data_path=data_train_test_val/adult/train.csv \
    dataset.test_data_path=data_train_test_val/adult/test.csv
```

DiCoFlex adds the directional constraint via Hydra:

```bash
"++counterfactuals_params.monotonic_overrides={age: INCREASE}"
```

Outputs land under `models/<dataset>_split/<method>/fold_*/`.

### 1b. Pairwise (5-fold CV) pipelines

```bash
./run_dice_pairwise_experiments.sh        # DiCE      × {mlp, mlr} × 5 datasets
./run_dicoflex_pairwise_experiments.sh    # DiCoFlex  × {mlp, mlr} × 5 datasets

# convenience: both at once
./all.sh
```

The same datasets are exercised: `lending_club, adult_census, credit_default, give_me_some_credit, bank_marketing`. Outputs go under `models/<dataset>/<method>/fold_{0..4}/`.

### 1c. Constraint-setup sweep

For each dataset there are 4–5 constraint setups (combinations of immutable + monotonic-increase features). The sweep runs DiCE / DiCoFlex / CCHVAE across the matrix:

```bash
# everything
./run_constraint_setup_experiments.sh

# subset
./run_constraint_setup_experiments.sh --datasets adult --setups 1 2 --methods dicoflex
```

Results land under `models/constraint_setups/<dataset>_setup<i>/...`. Generated per-setup yamls go to a temp dir; pass `--keep-configs` to inspect them.

---

## 2. Aggregate metrics → markdown tables

`scripts/calculate_metrics.py` reads `models/<dataset>/<method>/fold_*/cf_metrics_<model>.csv`, drops folds with 0 % validity, and writes a `mean ± std` markdown table.

Single run:

```bash
uv run python scripts/calculate_metrics.py \
    --dataset adult_census \
    --method DiceExplainerWrapper \
    --model-name MLPClassifier \
    --output-dir metrics_tables_dice \
    --models-root models \
    --metrics-conf-path counterfactuals/pipelines/conf/metrics/default.yaml
```

Loop over the full grid (datasets × discriminative models) for one method:

```bash
# args: METHOD  OUTPUT_DIR  MODELS_ROOT  DISC_MODELS_CSV  METRICS_CONF
./scripts/run_calculate_metrics.sh DiceExplainerWrapper metrics_tables_dice models "" \
    counterfactuals/pipelines/conf/metrics/default.yaml

./scripts/run_calculate_metrics.sh CADEX metrics_tables models \
    "MLPClassifier,MultinomialLogisticRegression" \
    counterfactuals/pipelines/conf/metrics/default.yaml
```

The script auto-picks the dataset tier (FULL vs SMALL) based on whether the method is in `FULL_METHODS`. Output: one `<dataset>_<method>_<model>_metrics.md` per (dataset, model) under `OUTPUT_DIR`.

Concatenate per-method tables into a single file:

```bash
./scripts/concat_md.sh metrics_tables_dice metrics_tables_dice/all.md
```

---

## 3. LaTeX result tables

### 3a. Multi-dataset table

```bash
uv run python scripts/generate_latex_tables.py \
    --results-dir metrics_tables_dice \
    --file-glob '*_metrics.md' \
    --metrics-conf-path counterfactuals/pipelines/conf/metrics/default.yaml \
    --combine-models \
    --drop-empty-rows \
    --output dice_tables.tex
```

Useful filters: `--include-methods`, `--exclude-methods`, `--include-datasets`, `--exclude-datasets`, `--include-metrics`, `--exclude-metrics` (each takes comma-separated values, repeatable). Use `--config-yaml` to supply `metric_meta` / `model_aliases` / `dataset_aliases` / `method_aliases` for nicer labels.

### 3b. Single-dataset table

`scripts/generate_latex_tables_single_dataset.py` has the same interface but emits one table per dataset.

### 3c. Per-criterion selection table (the "best-10" view)

```bash
uv run python scripts/compute_models_perselect_table.py \
    --models-root models \
    --datasets adult bank default gmc lending-club \
    --output models_perselect_table.tex
```

For each factual it picks the 10 valid CFs that best minimise each of (Prox.-Cont, Spars.-Cat, eps-Sparsity), and reports the mean + mixed-feature pairwise diversity. Continuous features are inverse-MinMaxed and re-scaled with a StandardScaler fit on train.

### 3d. Actionability table (single set of constraints)

```bash
uv run python scripts/compute_actionability_metrics.py \
    --datasets bank default adult gmc lending-club \
    --models-root models \
    --output scripts/actionability_table.tex
```

### 3e. Actionability across the constraint sweep

```bash
uv run python scripts/compute_sweep_actionability_metrics.py \
    --sweep-root models/constraint_setups \
    --datasets adult default \
    --scale standard
```

Writes `metrics.tex` per setup folder and a combined `<dataset>_metrics.tex` at the sweep root.

### 3f. "Best counterfactual" per (dataset, method)

```bash
uv run python -m scripts.find_best_counterfactuals \
    --datasets bank default adult gmc lending-club \
    --output best_counterfactuals.md
```

Picks the single CF that minimises Prox.-Cont + Spars.-Cat + eps-Sparsity (validity-filtered), and dumps the factual + CF in raw feature space.

---

## 4. End-to-end recipe (what I run to refresh everything)

```bash
# 1. train + generate CFs on the 5 main datasets
./all.sh                                       # DiCE + DiCoFlex pairwise
./run_traintest_experiments.sh                 # CCHVAE adult
./run_traintest_experiments2.sh                # CCHVAE + DiCE gmc, lending-club
./run_dicoflex_traintest_experiments.sh        # DiCoFlex bank, default
./run_dicoflex_traintest_experiments2.sh       # DiCoFlex gmc, lending-club
./run_constraint_setup_experiments.sh          # full constraint sweep

# 2. metric markdown per method (full sweep across datasets × models)
for m in DiceExplainerWrapper CADEX GLOBE_CE PPCEF AReS CCHVAE CaseBasedSACE; do
    ./scripts/run_calculate_metrics.sh "$m" metrics_tables models "" \
        counterfactuals/pipelines/conf/metrics/default.yaml
done

# 3. LaTeX tables
uv run python scripts/generate_latex_tables.py \
    --results-dir metrics_tables --combine-models --drop-empty-rows \
    --output results_main.tex

uv run python scripts/compute_models_perselect_table.py \
    --output models_perselect_table.tex

uv run python scripts/compute_actionability_metrics.py \
    --output scripts/actionability_table.tex

uv run python scripts/compute_sweep_actionability_metrics.py \
    --sweep-root models/constraint_setups --datasets adult default --scale standard

uv run python -m scripts.find_best_counterfactuals \
    --datasets bank default adult gmc lending-club \
    --output best_counterfactuals.md

# 4. format
uv run ruff format .
```

---

## 5. Notes / gotchas

- The shared `config/datasets/<dataset>_split.yaml` is method-agnostic; per-method things (e.g. monotonic direction) belong in pipeline configs or the CLI override (`++counterfactuals_params.monotonic_overrides=...`).
- DiCE / CCHVAE cannot enforce direction, so monotonic features are folded into immutable for them — this is the safe default. Only DiCoFlex re-enables them with a direction.
- `calculate_metrics.py` ignores folds where `validity == 0` so a single failed fold doesn't drag aggregates to zero.
- Validity always uses MinMax space (the disc_model lives there), even when proximity/sparsity are reported in StandardScaler space.
- Lending-club's split-config is named `lending_club_split.yaml` (underscore) while its data folder is `lending-club/` — the shell scripts already handle the mapping.
