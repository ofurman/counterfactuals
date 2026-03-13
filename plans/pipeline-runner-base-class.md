# Plan: Pipeline Runner Base Class

## Problem

30 pipeline runners share massive code duplication:

- **14 pipelines** delegate to `full_pipeline()` but still each define their own identical `calculate_metrics()` and `search_counterfactuals()`.
- **16 pipelines** are fully self-contained with copy-pasted `main()` loops (dataset loading, CV fold loop, disc model creation, gen model creation, dequantization, CF search, metrics, CSV export).
- `calculate_metrics()` is copy-pasted ~20 times with identical body (just calls `evaluate_cf` and logs).
- Helper functions like `one_hot()` are duplicated across AReS and GLOBE-CE.
- Return types are inconsistent: some return 6-tuples, group methods return 7-tuples with extras dict.

## Current Architecture

```
full_pipeline(cfg, preprocessing, logger, search_counterfactuals, calculate_metrics)
    └── callback-based: injects search_counterfactuals + calculate_metrics
    └── handles: dataset, CV loop, disc model, gen model, dequantization, log_prob threshold, CSV export

Self-contained pipelines (AReS, GLOBE-CE, CeFlow, GLANCE, PUMAL, WACH, etc.):
    └── each has its own main() duplicating the full_pipeline loop
    └── each defines its own calculate_metrics() and search_counterfactuals()
```

## Proposed Architecture

Replace `full_pipeline()` and all self-contained `main()` loops with a **template method base class**.

```
PipelineRunner (base class)
├── run()                          — template method, orchestrates everything
├── load_dataset()                 — shared, override for custom dataset handling
├── create_disc_model()            — shared
├── relabel_with_disc_model()      — shared, optional
├── create_gen_model()             — shared, override for custom gen model setup
├── compute_log_prob_threshold()   — shared
├── search_counterfactuals()       — ABSTRACT, each pipeline implements this
├── calculate_metrics()            — shared default (calls evaluate_cf), override for group methods
├── save_results()                 — shared
└── get_evaluate_fn()              — returns evaluate_cf by default, override for group methods
```

### Concrete Subclasses

```
PipelineRunner
├── PPCEFPipelineRunner           — only implements search_counterfactuals()
├── DiCEPipelineRunner            — only implements search_counterfactuals()
├── CCHVAEPipelineRunner          — only implements search_counterfactuals()
├── AReSPipelineRunner            — overrides create_gen_model() + search_counterfactuals()
├── CeFlowPipelineRunner         — overrides create_gen_model() (needs flow + density model)
├── GLANCEPipelineRunner         — overrides calculate_metrics() + search_counterfactuals()
├── PUMALPipelineRunner          — overrides calculate_metrics() + search_counterfactuals()
├── GLOBECEPipelineRunner        — overrides search_counterfactuals() (unscaled space)
└── ... (one per method)
```

## Detailed Design

### 1. `SearchResult` dataclass

Unify the inconsistent return types (6-tuple vs 7-tuple):

```python
@dataclass
class SearchResult:
    X_cf: np.ndarray
    X_test: np.ndarray
    y_orig: np.ndarray
    y_target: np.ndarray
    model_returned: np.ndarray
    cf_search_time: float
    extras: dict[str, Any] = field(default_factory=dict)
```

All `search_counterfactuals()` methods return `SearchResult`. Group methods put `cf_group_ids`, `S_matrix`, `D_matrix` etc. in `extras`.

### 2. `PipelineRunner` base class

```python
# counterfactuals/pipelines/base_runner.py

class PipelineRunner(ABC):
    def __init__(self, cfg: DictConfig, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger

    def run(self) -> None:
        """Template method — the shared pipeline skeleton."""
        dataset = self.load_dataset()
        dequantizer = GroupDequantizer(dataset.categorical_features_lists)

        for fold_n, _ in enumerate(dataset.get_cv_splits(self.cfg.experiment.n_folds)):
            disc_model_path, gen_model_path, save_folder = set_model_paths(self.cfg, fold=fold_n)

            # Step 1: Discriminative model
            disc_model = self.create_disc_model(dataset, disc_model_path, save_folder)

            # Step 2: Optional relabeling
            if self.cfg.experiment.relabel_with_disc_model:
                self.relabel_with_disc_model(dataset, disc_model)

            # Step 3: Generative model + dequantization
            dequantizer.fit(dataset.X_train)
            gen_model = self.create_gen_model(dataset, gen_model_path, dequantizer)

            # Step 4: Log-prob threshold
            log_prob_threshold = self.compute_log_prob_threshold(
                gen_model, dataset, dequantizer
            )

            # Step 5: Search counterfactuals
            result = self.search_counterfactuals(dataset, gen_model, disc_model, save_folder)

            # Step 6: Wrap gen model for metrics
            wrapped_gen_model = DequantizationWrapper(gen_model, dequantizer)

            # Step 7: Calculate metrics
            metrics = self.calculate_metrics(
                wrapped_gen_model, disc_model, dataset, result, log_prob_threshold
            )

            # Step 8: Save
            self.save_results(metrics, result.cf_search_time, save_folder)

    def load_dataset(self) -> MethodDataset:
        """Load and preprocess dataset. Override for custom dataset handling."""
        file_dataset = instantiate(self.cfg.dataset)
        preprocessing_pipeline = instantiate(self.cfg.preprocessing)
        return MethodDataset(file_dataset, preprocessing_pipeline)

    def create_disc_model(self, dataset, path, save_folder):
        """Shared discriminative model creation."""
        return create_disc_model(self.cfg, dataset, path, save_folder)

    def relabel_with_disc_model(self, dataset, disc_model):
        dataset.y_train = disc_model.predict(dataset.X_train)
        dataset.y_test = disc_model.predict(dataset.X_test)

    def create_gen_model(self, dataset, path, dequantizer):
        """Shared generative model creation. Override for custom setup (e.g., CeFlow)."""
        return create_gen_model(self.cfg, dataset, path, dequantizer)

    def compute_log_prob_threshold(self, gen_model, dataset, dequantizer):
        """Compute median log-prob threshold with temporary dequantization."""
        X_train_dq = dequantizer.transform(dataset.X_train)
        threshold = get_log_prob_threshold(
            gen_model, dataset, self.cfg.counterfactuals_params.batch_size,
            self.cfg.counterfactuals_params.log_prob_quantile, self.logger
        )
        dataset.X_train = dequantizer.inverse_transform(X_train_dq)
        return threshold

    @abstractmethod
    def search_counterfactuals(
        self, dataset, gen_model, disc_model, save_folder
    ) -> SearchResult:
        """Each pipeline method implements its own CF search logic."""
        ...

    def calculate_metrics(self, gen_model, disc_model, dataset, result, log_prob_threshold):
        """Default metrics calculation. Override for group methods."""
        return evaluate_cf(
            gen_model=gen_model,
            disc_model=disc_model,
            X_cf=result.X_cf,
            model_returned=result.model_returned,
            categorical_features=dataset.categorical_features_indices,
            continuous_features=dataset.numerical_features_indices,
            X_train=dataset.X_train,
            y_train=dataset.y_train.reshape(-1),
            X_test=result.X_test,
            y_test=result.y_orig,
            y_target=result.y_target,
            median_log_prob=log_prob_threshold,
        )

    def save_results(self, metrics, cf_search_time, save_folder):
        """Save metrics to CSV."""
        df = pd.DataFrame(metrics, index=[0])
        df["cf_search_time"] = cf_search_time
        disc_model_name = self.cfg.disc_model.model._target_.split(".")[-1]
        df.to_csv(os.path.join(save_folder, f"cf_metrics_{disc_model_name}.csv"), index=False)
```

### 3. Example subclass — simple method (PPCEF)

```python
class PPCEFPipelineRunner(PipelineRunner):
    @abstractmethod
    def search_counterfactuals(self, dataset, gen_model, disc_model, save_folder) -> SearchResult:
        # ... PPCEF-specific CF search logic (moved from current search_counterfactuals function)
        return SearchResult(X_cf=..., X_test=..., y_orig=..., y_target=...,
                           model_returned=..., cf_search_time=...)
```

### 4. Example subclass — group method (GLANCE)

```python
class GLANCEPipelineRunner(PipelineRunner):
    def search_counterfactuals(self, dataset, gen_model, disc_model, save_folder) -> SearchResult:
        # ... GLANCE-specific CF search logic
        return SearchResult(
            X_cf=..., X_test=..., y_orig=..., y_target=...,
            model_returned=..., cf_search_time=...,
            extras={"cf_group_ids": cf_group_ids}
        )

    def calculate_metrics(self, gen_model, disc_model, dataset, result, log_prob_threshold):
        return evaluate_cf_for_glance(
            gen_model=gen_model, disc_model=disc_model,
            X_cf=result.X_cf, model_returned=result.model_returned,
            ...,
            cf_group_ids=result.extras.get("cf_group_ids"),
        )
```

### 5. Example subclass — custom gen model (CeFlow)

```python
class CeFlowPipelineRunner(PipelineRunner):
    def create_gen_model(self, dataset, path, dequantizer):
        """CeFlow needs both a flow model and a density model."""
        flow_model = self._create_flow_model(dataset, path)
        density_model = self._create_density_model(dataset)
        return CeFlowComposite(flow_model, density_model)

    def search_counterfactuals(self, dataset, gen_model, disc_model, save_folder) -> SearchResult:
        # CeFlow-specific search using gen_model.flow_model and gen_model.density_model
        ...
```

### 6. Hydra entry point

Each pipeline's `main()` becomes a thin wrapper:

```python
@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    runner = PPCEFPipelineRunner(cfg, logger)
    runner.run()
```

## Special Cases & How They Map

| Pipeline | Overrides | Notes |
|----------|-----------|-------|
| PPCEF, DiCE, CCHVAE, Artelt, CEM, CEGP, CET, TabDCE, CADEX, SACE, Wach | `search_counterfactuals` only | Simplest case — use base `calculate_metrics` |
| GLANCE | `search_counterfactuals` + `calculate_metrics` | Group method, returns `cf_group_ids` in extras |
| PUMAL | `search_counterfactuals` + `calculate_metrics` | Group method, returns `S_matrix`/`D_matrix` in extras |
| AReS | `load_dataset` + `search_counterfactuals` | Custom dataset handling (one-hot encoding, unscaled space) |
| GLOBE-CE | `search_counterfactuals` | Unscaled space with predict_fn closure |
| CeFlow | `create_gen_model` + `search_counterfactuals` | Dual model (flow + density) |
| *_pairwise | `search_counterfactuals` + `calculate_metrics` | Pairwise comparison variants |
| *_traintest | `search_counterfactuals` | Train/test split variants |

## Implementation Steps

### Phase 1: Foundation (no pipeline changes yet)
1. Create `counterfactuals/pipelines/base_runner.py` with `PipelineRunner` and `SearchResult`.
2. Extract shared helpers from `full_pipeline.py` into the base class methods.
3. Add unit tests for `PipelineRunner` with a mock subclass.

### Phase 2: Migrate `full_pipeline` users (14 pipelines)
4. Convert PPCEF pipeline to `PPCEFPipelineRunner` (proof of concept).
5. Run existing tests + manual smoke test to verify equivalence.
6. Convert remaining `full_pipeline` users one-by-one (DiCE, CCHVAE, Artelt, CEM, CEGP, CET, TabDCE, CADEX, SACE, etc.).
7. Delete `full_pipeline.py` once all users are migrated.

### Phase 3: Migrate self-contained pipelines (16 pipelines)
8. Convert AReS (needs `load_dataset` override).
9. Convert CeFlow (needs `create_gen_model` override).
10. Convert GLANCE, PUMAL (need `calculate_metrics` override).
11. Convert GLOBE-CE (unscaled space handling).
12. Convert remaining self-contained pipelines.

### Phase 4: Cleanup
13. Delete all standalone `calculate_metrics()` functions (replaced by base class).
14. Move shared utils (`one_hot()`, `_build_features_tree_from_one_hot()`) to `counterfactuals/utils/`.
15. Standardize type hints to Python 3.10+ style across all pipeline files.

## Estimated Impact

- **~2500-3000 lines deleted** (duplicated `calculate_metrics`, duplicated CV loops, duplicated model creation).
- **~300 lines added** (base class + SearchResult).
- **Net reduction: ~2200-2700 lines**.
- Each new pipeline method only needs to implement `search_counterfactuals()` (~30-80 lines) instead of a full `main()` (~150-300 lines).

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Some pipelines have subtle differences in the main loop | Identify all differences before migrating; use hook methods for edge cases |
| CeFlow dual-model pattern doesn't fit cleanly | Allow `create_gen_model()` to return any object; CeFlow uses a composite wrapper |
| AReS/GLOBE-CE unscaled space handling | Override `load_dataset()` or add a `preprocess_for_search()` hook |
| Breaking existing Hydra configs | Keep `@hydra.main` entry point unchanged; only refactor internal structure |
| Pairwise pipelines have different metric calculation | Override `calculate_metrics()` in pairwise variants |
