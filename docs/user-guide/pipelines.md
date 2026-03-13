# Running Pipelines

Execute end-to-end experiments using the Hydra configuration system.

## Overview

Pipelines automate the complete workflow:

1. Load dataset (with 5-fold cross-validation)
2. Train / load discriminative and generative models
3. Generate counterfactuals
4. Compute evaluation metrics
5. Save results to CSV

## Architecture

Each pipeline is built around two layers:

- **Runner class** — lives in `counterfactuals/pipelines/runners/` and contains all the logic
  for a specific CF method. Inherits from the abstract `PipelineRunner` base class.
- **Entry-point script** — a thin `run_*_pipeline.py` wrapper that configures preprocessing
  and instantiates the runner. This is what you invoke on the command line.

### PipelineRunner base class

`PipelineRunner` implements the *template method* pattern. The `run()` method orchestrates
the full CV loop, calling overridable hooks in order:

```
run()
 ├── load_dataset()
 ├── for each fold:
 │    ├── create_disc_model()
 │    ├── create_gen_model()
 │    ├── compute_log_prob_threshold()
 │    ├── search_counterfactuals()   ← abstract, must be implemented
 │    ├── calculate_metrics()
 │    └── save_results()
```

Each runner subclass declares `cf_method_name` as a class-level constant:

```python
class PPCEFPipelineRunner(PipelineRunner):
    cf_method_name = "PPCEF"

    def search_counterfactuals(self, dataset, gen_model, disc_model, save_folder, log_prob_threshold):
        ...
```

### SearchResult

`search_counterfactuals()` must return a `SearchResult` dataclass:

| Field | Type | Description |
|-------|------|-------------|
| `X_cf` | array | Generated counterfactual examples |
| `X_test` | array | Original test examples used for generation |
| `y_orig` | array | Original labels |
| `y_target` | array | Target labels |
| `model_returned` | bool array | Per-sample success flag |
| `cf_search_time` | float | Wall-clock search time in seconds |
| `extras` | dict | Method-specific outputs (e.g. group IDs, S/D matrices) |

## Running a Pipeline

```bash
# Run PPCEF pipeline
uv run python -m counterfactuals.pipelines.run_ppcef_pipeline

# Override configuration
uv run python -m counterfactuals.pipelines.run_ppcef_pipeline \
    dataset.config_path=config/datasets/compas.yaml \
    counterfactuals_params.epochs=200
```

## Configuration Structure

```yaml
# pipelines/conf/ppcef_config.yaml
defaults:
  - gen_model: large_maf
  - disc_model: mlp

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
  target_class: 1
  epochs: 100
  lr: 0.01
  alpha: 1.0
  log_prob_quantile: 0.5
```

## Available Pipelines

### Local methods

| Entry-point script | Runner class | CF method |
|---|---|---|
| `run_ppcef_pipeline` | `PPCEFPipelineRunner` | PPCEF |
| `run_wach_pipeline` | `WACHPipelineRunner` | WACH / RPPCEF |
| `run_wach_ours_pipeline` | `WACHOURSPipelineRunner` | WACH-OURS |
| `run_dice_pipeline` | `DiceExplainerRunner` | DiCE |
| `run_dice_pairwise_pipeline` | `DicePairwisePipelineRunner` | DiCE (pairwise) |
| `run_cchvae_pipeline` | `CCHVAEPipelineRunner` | CCHVAE |
| `run_cchvae_pairwise_pipeline` | `CCHVAEPairwisePipelineRunner` | CCHVAE (pairwise) |
| `run_cegp_pipeline` | `CEGPPipelineRunner` | CEGP |
| `run_cem_pipeline` | `CEMPipelineRunner` | CEM |
| `run_cet_pipeline` | `CETPipelineRunner` | CET |
| `run_artelt_pipeline` | `ArteltPipelineRunner` | Artelt |
| `run_cadex_pipeline` | `CADEXPipelineRunner` | CADEX |
| `run_casebased_sace_pipeline` | `CaseBasedSACEPipelineRunner` | Case-Based SACE |
| `run_tabdce_pipeline` | `TabDCEPipelineRunner` | TabDCE |
| `run_tabdce_pairwise_pipeline` | `TabDCEPairwisePipelineRunner` | TabDCE (pairwise) |
| `run_ceflow_pipeline` | `CeFlowPipelineRunner` | CeFlow |
| `run_lice_pipeline` | `LiCEPipelineRunner` | LiCE |
| `run_ppcefr_pipeline` | `PPCEFRPipelineRunner` | PPCEF-R (regression) |

### Group / global methods

| Entry-point script | Runner class | CF method |
|---|---|---|
| `run_globe_ce_pipeline` | `GLOBECEPipelineRunner` | GLOBE-CE |
| `run_group_globe_ce_pipeline` | `GroupGLOBECEPipelineRunner` | Group GLOBE-CE |
| `run_regional_globe_ce_pipeline` | `RegionalGLOBECEPipelineRunner` | Regional GLOBE-CE |
| `run_ares_pipeline` | `AReSPipelineRunner` | AReS |
| `run_glance_pipeline` | `GLANCEPipelineRunner` | GLANCE |
| `run_tcrex_pipeline` | `TCRExPipelineRunner` | TCREx |
| `run_pumal_pipeline` | `PUMALPipelineRunner` | PUMAL |

## Creating a Custom Pipeline

1. Create a runner class in `counterfactuals/pipelines/runners/my_method_runner.py`:

```python
from counterfactuals.pipelines.base_runner import PipelineRunner, SearchResult

class MyMethodPipelineRunner(PipelineRunner):
    cf_method_name = "MyMethod"

    def search_counterfactuals(
        self, dataset, gen_model, disc_model, save_folder, log_prob_threshold
    ):
        disc_model_name = self.cfg.disc_model.model._target_.split(".")[-1]

        X_test, y_test = self._filter_test_data(dataset, self.cfg.counterfactuals_params.target_class)
        cf_method = MyMethod(disc_model=disc_model)

        Xs_cfs = cf_method.explain(X_test)
        self._save_counterfactuals(Xs_cfs, save_folder, self.cf_method_name, disc_model_name)

        return SearchResult(
            X_cf=Xs_cfs,
            X_test=X_test,
            y_orig=y_test,
            y_target=...,
            model_returned=...,
            cf_search_time=...,
        )
```

2. Create a thin entry-point script `counterfactuals/pipelines/run_my_method_pipeline.py`:

```python
import logging
import hydra
from omegaconf import DictConfig
from counterfactuals.pipelines.runners.my_method_runner import MyMethodPipelineRunner
from counterfactuals.preprocessing import MinMaxScalingStep, PreprocessingPipeline, TorchDataTypeStep

logger = logging.getLogger(__name__)

@hydra.main(config_path="./conf", config_name="my_method_config", version_base="1.2")
def main(cfg: DictConfig):
    preprocessing_pipeline = PreprocessingPipeline([
        ("minmax", MinMaxScalingStep()),
        ("torch_dtype", TorchDataTypeStep()),
    ])
    runner = MyMethodPipelineRunner(cfg, logger, preprocessing_pipeline)
    runner.run()

if __name__ == "__main__":
    main()
```

3. Add a Hydra config file at `counterfactuals/pipelines/conf/my_method_config.yaml`.

### Overridable hooks

Override these methods on the runner class for non-standard behaviour:

| Method | When to override |
|--------|-----------------|
| `load_dataset()` | Custom dataset loading (e.g. wrap with `MethodDataset`) |
| `create_disc_model()` | Custom discriminative model setup |
| `create_gen_model()` | Custom generative model (e.g. CeFlow uses two models) |
| `compute_log_prob_threshold()` | Different plausibility threshold computation |
| `calculate_metrics()` | Method-specific metrics (e.g. group metrics for GLANCE) |
| `save_results()` | Extra columns in the output CSV (e.g. `n_groups` for TCREx) |
| `run()` | Completely custom pipeline loop (e.g. no CV, regression task) |
