# Pipeline Refactoring Plan: Complete Runner Migration

## Context
The codebase is mid-refactor: 16 runner classes exist in `runners/` but many pipeline files either use `SimplePipelineRunner` (delegating to a standalone function) or the old `full_pipeline()` pattern. Goal: every `run_*_pipeline.py` should use a dedicated `PipelineRunner` subclass directly.

**Out of scope** (per user decision): DiCoFlex variants (3 files), traintest variants (3 files). These 6 files are untouched.

## Phase 0: Infrastructure

### 0.1 Extract `compute_pairwise_mean_distance` to shared utility
- **File**: `counterfactuals/pipelines/metrics_utils.py` (new)
- Currently duplicated in: `run_dice_pairwise_pipeline.py`, `run_cchvae_pairwise_pipeline.py`, `run_tabdce_pairwise_pipeline.py`, `run_dicoflex_pairwise_pipeline.py`
- Extract the standard version: `compute_pairwise_mean_distance(cfs_3d: np.ndarray) -> float`

### 0.2 Create `PairwiseMixin` for pairwise runners
- **File**: `counterfactuals/pipelines/runners/pairwise_mixin.py` (new)
- Overrides `calculate_metrics()` to:
  1. Call base `calculate_metrics()` using `result.X_cf` (first CF per instance)
  2. Extract `result.extras["Xs_cfs_all"]` (3D array)
  3. Append `metrics["pairwise_mean_distance"]`
- Convention: pairwise runners store `Xs_cfs_all` in `SearchResult.extras` and set `X_cf = Xs_cfs_first`

## Phase 1: Convert SimplePipelineRunner files to dedicated runners (7 files)

Each file already has a working runner in `runners/`. Replace the `SimplePipelineRunner + standalone function` pattern with the dedicated runner class. Delete the standalone `search_counterfactuals` function from each pipeline file.

| Pipeline file | Runner class | Runner file |
|---|---|---|
| `run_artelt_pipeline.py` | `ArteltPipelineRunner` | `runners/artelt_runner.py` |
| `run_casebased_sace_pipeline.py` | `CaseBasedSACEPipelineRunner` | `runners/casebased_sace_runner.py` |
| `run_cchvae_pipeline.py` | `CCHVAEPipelineRunner` | `runners/cchvae_runner.py` |
| `run_cegp_pipeline.py` | `CEGPPipelineRunner` | `runners/cegp_runner.py` |
| `run_cem_pipeline.py` | `CEMPipelineRunner` | `runners/cem_runner.py` |
| `run_ppcef_pipeline.py` | `PPCEFPipelineRunner` | `runners/ppcef_runner.py` |
| `run_wach_ours_pipeline.py` | `WACHOURSPipelineRunner` | `runners/wach_ours_runner.py` |

Each becomes ~15 lines: imports + `@hydra.main` + runner instantiation.

## Phase 2: Convert `full_pipeline()`-based files (2 files)

### 2.1 `run_dice_pipeline.py`
- Runner `DiCEPipelineRunner` already exists in `runners/dice_runner.py`
- Delete: standalone `search_counterfactuals`, `calculate_metrics`, `get_log_prob_threshold`, `get_categorical_intervals`, `apply_categorical_discretization`, `DiscWrapper` (all duplicated in runner)
- Replace `full_pipeline()` call with `DiCEPipelineRunner(cfg, logger, preprocessing_pipeline).run()`

### 2.2 `run_tabdce_pairwise_pipeline.py`
- Uses `full_pipeline()` — needs a new pairwise runner (Phase 3.3)

## Phase 3: Create new pairwise runners (3 files)

### 3.1 `runners/dice_pairwise_runner.py` (new)
- `DiCEPairwisePipelineRunner(PairwiseMixin, PipelineRunner)`
- `search_counterfactuals`: DiCE with `total_CFs > 1`, stores `Xs_cfs_all` in extras
- Used by: `run_dice_pairwise_pipeline.py`

### 3.2 `runners/cchvae_pairwise_runner.py` (new)
- `CCHVAEPairwisePipelineRunner(PairwiseMixin, PipelineRunner)`
- `search_counterfactuals`: runs CCHVAE N times, stacks results
- Used by: `run_cchvae_pairwise_pipeline.py`

### 3.3 `runners/tabdce_pairwise_runner.py` (new)
- `TabDCEPairwisePipelineRunner(PairwiseMixin, PipelineRunner)`
- `search_counterfactuals`: runs TabDCE N times, stacks results
- Used by: `run_tabdce_pairwise_pipeline.py`

## Phase 4: Create new runners for remaining manual-loop pipelines (5 files)

### 4.1 `runners/wach_runner.py` (new) — for `run_wach_pipeline.py`
- `WACHPipelineRunner(PipelineRunner)`
- Override `create_gen_model()`: calls without dequantizer
- `search_counterfactuals`: WACH's `explain_dataloader` returns 5 values directly
- Note: WACH computes log_prob_threshold inside search — override `compute_log_prob_threshold` or handle in search

### 4.2 `runners/tcrex_runner.py` (new) — for `run_tcrex_pipeline.py`
- `TCRExPipelineRunner(PipelineRunner)`
- `search_counterfactuals`: filters by `origin_class`, uses `align_counterfactuals_with_factuals`, stores `n_groups` in extras
- Override `save_results` to add `n_groups` column
- Override `create_gen_model()`: calls without dequantizer

### 4.3 `runners/lice_runner.py` (new) — for `run_lice_pipeline.py`
- `LiCEPipelineRunner(PipelineRunner)`
- Override `run()`: uses raw dataset (no MethodDataset), no dequantizer
- `search_counterfactuals`: per-sample generation with try/except, SPN, ONNX export
- Fix: replace `print()` with `logger.info()`, fix config name from `globe_ce_config` to `lice_config`

### 4.4 `runners/group_globe_ce_runner.py` (new) — for `run_group_globe_ce_pipeline.py`
- `GroupGLOBECEPipelineRunner(PipelineRunner)`
- Override `run()`: uses raw dataset, no preprocessing pipeline, no dequantizer
- `search_counterfactuals`: KMeans clustering + per-cluster AReS + GLOBE_CE
- Extract shared `one_hot()` to `pipelines/utils.py` (duplicated in 3 files)

### 4.5 `runners/regional_globe_ce_runner.py` (new) — for `run_regional_globe_ce_pipeline.py`
- `RegionalGLOBECEPipelineRunner(PipelineRunner)`
- Override `run()`: raw dataset, no dequantizer
- Very similar to group_globe_ce but with per-cluster generation pattern
- Shares `one_hot()` from `pipelines/utils.py`

### 4.6 `runners/ppcefr_runner.py` (new) — for `run_ppcefr_pipeline.py`
- `PPCEFRPipelineRunner(PipelineRunner)`
- Override `run()`: raw dataset, single fold (no CV), no dequantizer
- Override `calculate_metrics()`: calls `evaluate_cf_regression` instead of `evaluate_cf`
- `search_counterfactuals`: PPCEFR with `target_change`, `delta`

## Phase 5: Cleanup

### 5.1 Delete `full_pipeline/` directory
- `counterfactuals/pipelines/full_pipeline/full_pipeline.py`
- `counterfactuals/pipelines/full_pipeline/__init__.py`
- Verify no remaining imports

### 5.2 Delete `runners/simple_runner.py`
- Remove `SimplePipelineRunner` once all callers are gone

### 5.3 Delete `run_ppcef_simple_runner.py`
- Leftover from refactor experimentation

### 5.4 Update `runners/__init__.py`
- Export all new runners

### 5.5 Run linting
- `ruff check --fix` and `ruff format` on all modified files

## Files Modified (summary)

**New files (11):**
- `counterfactuals/pipelines/metrics_utils.py`
- `counterfactuals/pipelines/runners/pairwise_mixin.py`
- `counterfactuals/pipelines/runners/dice_pairwise_runner.py`
- `counterfactuals/pipelines/runners/cchvae_pairwise_runner.py`
- `counterfactuals/pipelines/runners/tabdce_pairwise_runner.py`
- `counterfactuals/pipelines/runners/wach_runner.py`
- `counterfactuals/pipelines/runners/tcrex_runner.py`
- `counterfactuals/pipelines/runners/lice_runner.py`
- `counterfactuals/pipelines/runners/group_globe_ce_runner.py`
- `counterfactuals/pipelines/runners/regional_globe_ce_runner.py`
- `counterfactuals/pipelines/runners/ppcefr_runner.py`

**Simplified pipeline files (14):**
- `run_artelt_pipeline.py`, `run_casebased_sace_pipeline.py`, `run_cchvae_pipeline.py`
- `run_cegp_pipeline.py`, `run_cem_pipeline.py`, `run_ppcef_pipeline.py`
- `run_wach_ours_pipeline.py`, `run_dice_pipeline.py`
- `run_dice_pairwise_pipeline.py`, `run_cchvae_pairwise_pipeline.py`
- `run_tabdce_pairwise_pipeline.py`
- `run_wach_pipeline.py`, `run_tcrex_pipeline.py`, `run_lice_pipeline.py`
- `run_group_globe_ce_pipeline.py`, `run_regional_globe_ce_pipeline.py`
- `run_ppcefr_pipeline.py`

**Deleted files (4):**
- `counterfactuals/pipelines/full_pipeline/full_pipeline.py`
- `counterfactuals/pipelines/full_pipeline/__init__.py`
- `counterfactuals/pipelines/runners/simple_runner.py`
- `counterfactuals/pipelines/run_ppcef_simple_runner.py`

**Modified existing files (2):**
- `counterfactuals/pipelines/utils.py` — add `one_hot()` helper
- `counterfactuals/pipelines/runners/__init__.py` — export new runners

**Untouched (out of scope, 6 files):**
- `run_dicoflex_pipeline.py`, `run_dicoflex_pairwise_pipeline.py`, `run_dicoflex_traintest_pipeline.py`
- `run_dice_traintest_pipeline.py`, `run_cchvae_traintest_pipeline.py`, `run_dicoflex_traintest_pipeline.py`

## Verification
1. `ruff check counterfactuals/pipelines/` — no lint errors
2. `ruff format --check counterfactuals/pipelines/` — properly formatted
3. `grep -r "full_pipeline" counterfactuals/` — no remaining references (except dicoflex/traintest out-of-scope files)
4. `grep -r "SimplePipelineRunner" counterfactuals/` — no remaining references
5. Run existing tests: `uv run pytest tests/`
