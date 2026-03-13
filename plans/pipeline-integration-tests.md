# Plan: Pipeline Integration Tests

## Goal

Add smoke-test-level integration tests that verify every `PipelineRunner` subclass
produces a structurally valid `SearchResult` when given a tiny synthetic dataset.
No test should require file I/O, model checkpoints, or real training epochs.

---

## Design Principles

1. **Test `search_counterfactuals()` directly**, not `run()`.
   `run()` is orchestration + 5-fold CV + file I/O — already tested implicitly by
   the runners themselves. What we need to verify is that the CF-generation logic
   is wired correctly end-to-end.

2. **In-memory fixtures only** — no disk reads, no Hydra runtime, no checkpoint files.
   Use `OmegaConf.create({...})` for configs and a `SimpleNamespace`-style mock
   dataset that matches the `MethodDataset` interface.

3. **Real but tiny models where possible** — a 1-epoch MLP is fine and avoids the
   complexity of mocking every method (`predict`, `predict_proba`, `predict_log_prob`,
   `eval`, …). Use `train_model=False`-style construction (random weights, no
   training) where the runner doesn't use the model's predictions for anything
   meaningful in a smoke test.

4. **One parametrized smoke test covers all runners** where they share the same
   interface. Runners with special dataset requirements get dedicated tests.

5. **Assertions are structural** — shape, dtype, timing ≥ 0.
   Do not assert on metric *values* in smoke tests; that belongs to metrics unit tests.

---

## File Structure

```
tests/
└── test_pipelines/
    ├── __init__.py
    ├── conftest.py               ← shared fixtures (dataset, models, cfg builders, tmp dir)
    ├── test_smoke_local.py       ← all standard local-method runners
    ├── test_smoke_group.py       ← group / global runners (GLANCE, GLOBE-CE, TCREx, …)
    ├── test_smoke_pairwise.py    ← pairwise runners (DiCE, CCHVAE, TabDCE)
    └── test_smoke_special.py     ← CeFlow, LiCE, PPCEFR (each has special requirements)
```

---

## Shared Fixtures (`conftest.py`)

### `synthetic_dataset` fixture

A plain Python object (not a `MethodDataset`) with the minimal attribute set that
all runners read. Constructed from `numpy` arrays with deterministic `np.random.default_rng(0)`.

```python
# Shape: 80 train rows, 20 test rows, 6 features (4 continuous + 2 categorical one-hot group)
# Binary classification: y ∈ {0, 1}, balanced

@pytest.fixture(scope="session")
def synthetic_dataset():
    rng = np.random.default_rng(0)
    n_train, n_test, n_feat = 80, 20, 6

    X_train = rng.uniform(0, 1, (n_train, n_feat)).astype(np.float32)
    X_test  = rng.uniform(0, 1, (n_test,  n_feat)).astype(np.float32)
    y_train = rng.integers(0, 2, n_train).astype(np.float32)
    y_test  = rng.integers(0, 2, n_test ).astype(np.float32)
    # Guarantee at least one sample per class in test set
    y_test[:10] = 0; y_test[10:] = 1

    ds = SimpleNamespace(
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        features=[f"f{i}" for i in range(n_feat)],
        numerical_features_indices=[0, 1, 2, 3],
        categorical_features_indices=[4, 5],
        categorical_features=["cat_a"],
        categorical_features_lists=[[4, 5]],   # one one-hot group of size 2
        numerical_features=[f"f{i}" for i in range(4)],
        actionable_features=None,
    )
    # Provide a minimal train_dataloader()
    def train_dataloader(batch_size=32, shuffle=False, noise_lvl=0):
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.tensor(X_train), torch.tensor(y_train)
            ),
            batch_size=batch_size, shuffle=shuffle,
        )
    ds.train_dataloader = train_dataloader
    return ds
```

Why 6 features with one two-column one-hot group: enough to exercise categorical
handling in PPCEF, TabDCE, DiCE without making the model large.

---

### `tiny_disc_model` fixture

A two-layer MLP instantiated with random weights (no training). Exposed interface:
`predict(X)`, `predict_proba(X)`, `eval()`.

```python
@pytest.fixture(scope="session")
def tiny_disc_model():
    """MLPClassifier with random weights — fast, no training needed."""
    model = MLPClassifier(num_inputs=6, num_targets=1,
                          hidden_layer_sizes=[8], dropout=0.0)
    model.eval()
    return model
```

For runners that call `disc_model.predict_proba` (CET wrapper, DiCE) the existing
`MLPClassifier.predict_proba` is sufficient.

---

### `tiny_gen_model` fixture

A tiny MAF (or KDE as fallback) with random weights. Exposed interface:
`predict_log_prob(dataloader)`, `eval()`.

```python
@pytest.fixture(scope="session")
def tiny_gen_model():
    """Small MAF with random weights — provides predict_log_prob interface."""
    model = MaskedAutoregressiveFlow(
        features=6, context_features=1,
        hidden_features=4, num_blocks_per_layer=1, num_layers=2,
    )
    model.eval()
    return model
```

---

### `base_cfg` fixture

A function fixture that returns an `OmegaConf.DictConfig` pre-populated with the
keys every runner accesses:

```python
@pytest.fixture
def base_cfg():
    return OmegaConf.create({
        "disc_model": {
            "model": {"_target_": "counterfactuals.models.classifier.MLPClassifier"},
        },
        "counterfactuals_params": {
            "target_class": 1,
            "batch_size": 16,
            "epochs": 1,
            "lr": 0.01,
            "log_prob_quantile": 0.5,
        },
        "experiment": {"relabel_with_disc_model": False},
    })
```

Each test that needs extra keys merges with `OmegaConf.merge(base_cfg, {...})`.

---

### `make_runner` helper

```python
def make_runner(runner_cls, cfg, logger=logging.getLogger("test")):
    return runner_cls(cfg, logger, preprocessing_pipeline=None)
```

---

## Test: `test_smoke_local.py`

### Runners covered

| Runner class | Extra cfg keys needed | Special notes |
|---|---|---|
| `PPCEFPipelineRunner` | `disc_model_criterion`, `alpha`, `alpha_s`, `alpha_k`, `patience`, `use_categorical`, `plausibility_weight`, `plausibility_bias` | Standard |
| `WACHOURSPipelineRunner` | `disc_model_criterion`, `alpha` | No gen model used |
| `WACHPipelineRunner` | `cf_method._target_`, `log_prob_quantile` | Has `create_gen_model` override |
| `ArteltPipelineRunner` | none beyond base | No gen model used |
| `CEGPPipelineRunner` | `beta`, `c_init`, `c_steps`, `max_iterations`, `feature_range`, `fit_d_type`, `fit_disc_perc` | Passes X_train to explain |
| `CEMPipelineRunner` | `mode`, `kappa`, `beta`, `c_init`, `c_steps`, `max_iterations`, `learning_rate_init`, `fit_no_info_type`, `feature_range`, `clip_range` | No gen model used |
| `CETPipelineRunner` | none beyond base | Calls `dataset.inverse_transform` → add to fixture |
| `CADEXPipelineRunner` | CADEX-specific params | — |
| `CaseBasedSACEPipelineRunner` | SACE-specific params | — |
| `CCHVAEPipelineRunner` | `latent_dim`, `hidden_sizes`, `vae_epochs`, `vae_lr` | Trains VAE internally |
| `DiCEPipelineRunner` | `n_cfs` | Needs pandas-compatible disc_model |
| `TabDCEPipelineRunner` | `tabdce.*` sub-config | Trains diffusion model internally |

### Parametrized smoke test pattern

```python
LOCAL_RUNNERS = [
    (PPCEFPipelineRunner,    ppcef_cfg_factory),
    (WACHOURSPipelineRunner, wach_ours_cfg_factory),
    (ArteltPipelineRunner,   artelt_cfg_factory),
    (CEGPPipelineRunner,     cegp_cfg_factory),
    (CEMPipelineRunner,      cem_cfg_factory),
    # …
]

@pytest.mark.parametrize("runner_cls,cfg_factory", LOCAL_RUNNERS,
                          ids=[cls.cf_method_name for cls, _ in LOCAL_RUNNERS])
def test_search_counterfactuals_returns_valid_result(
    runner_cls, cfg_factory, synthetic_dataset, tiny_disc_model, tiny_gen_model, tmp_path
):
    cfg = cfg_factory()
    runner = make_runner(runner_cls, cfg)

    result = runner.search_counterfactuals(
        dataset=synthetic_dataset,
        gen_model=tiny_gen_model,
        disc_model=tiny_disc_model,
        save_folder=str(tmp_path),
        log_prob_threshold=0.0,
    )

    # --- structural assertions ---
    assert isinstance(result, SearchResult)
    n = result.X_test.shape[0]
    assert result.X_cf.shape     == (n, synthetic_dataset.X_test.shape[1])
    assert result.y_orig.shape   == (n,)
    assert result.y_target.shape == (n,)
    assert result.model_returned.shape == (n,)
    assert result.cf_search_time >= 0.0
```

### CET-specific dataset augmentation

`CETPipelineRunner` calls `dataset.inverse_transform(dataset.X_train)`.
Add a helper to the fixture:

```python
ds.inverse_transform = lambda X: X  # identity — data is already unscaled in tests
```

---

## Test: `test_smoke_group.py`

### Runners covered

| Runner class | Dataset additions needed | Special notes |
|---|---|---|
| `GLANCEPipelineRunner` | none | Overrides `calculate_metrics`; test only `search_counterfactuals` |
| `TCRExPipelineRunner` | `origin_class` in cfg | Calls `cf_method.fit()` then `explain()` |
| `PUMALPipelineRunner` | `origin_class`, one-hot y handling | Complex extras (S_matrix, D_matrix) |
| `GLOBECEPipelineRunner` | `preprocessing_pipeline.get_step("minmax")` on dataset | Needs unscaled data + minmax scaler ref |
| `GroupGLOBECEPipelineRunner` | `feature_transformer` on dataset | Overrides `run()`, test `search_counterfactuals` directly |
| `RegionalGLOBECEPipelineRunner` | `feature_transformer`, `log_prob_quantile` | Same as above |
| `AReSPipelineRunner` | `feature_transformer` or `preprocessing_pipeline`, `one_hot_feature_groups` | Most complex dataset requirements |

### Globe-CE / AReS dataset fixture extension

These runners call `dataset.preprocessing_pipeline.get_step("minmax")` and/or
`dataset.feature_transformer.inverse_transform(X)` / `.transform(X)`.

Create a `globe_ce_dataset` fixture (extends `synthetic_dataset`) that adds:

```python
from sklearn.preprocessing import MinMaxScaler

@pytest.fixture(scope="session")
def globe_ce_dataset(synthetic_dataset):
    import copy
    ds = copy.copy(synthetic_dataset)
    scaler = MinMaxScaler()
    scaler.fit(ds.X_train)

    class _MinMaxStep:
        """Minimal interface matching MinMaxScalingStep."""
        def _transform_array(self, X): return scaler.transform(X)
        def _inverse_transform_array(self, X): return scaler.inverse_transform(X)

    class _PreprocessingPipeline:
        def get_step(self, name):
            return _MinMaxStep() if name == "minmax" else None

    class _FeatureTransformer:
        def transform(self, X): return scaler.transform(X)
        def inverse_transform(self, X): return scaler.inverse_transform(X)

    ds.preprocessing_pipeline = _PreprocessingPipeline()
    ds.feature_transformer = _FeatureTransformer()
    ds.one_hot_feature_groups = None  # no one-hot groups in synthetic data
    return ds
```

### PUMAL extras assertion

```python
def test_pumal_extras(pumal_result):
    assert "S_matrix" in pumal_result.extras
    assert "D_matrix" in pumal_result.extras
```

### TCREx n_groups assertion

```python
def test_tcrex_n_groups_stored(tcrex_runner, ...):
    result = tcrex_runner.search_counterfactuals(...)
    assert "n_groups" in result.extras
    assert result.extras["n_groups"] >= 1
```

---

## Test: `test_smoke_pairwise.py`

### Runners covered

| Runner class | Extra assertions |
|---|---|
| `DiCEPairwisePipelineRunner` | `Xs_cfs_all` in extras, shape `(n, cf_per_instance, n_features)` |
| `CCHVAEPairwisePipelineRunner` | `Xs_cfs_all` in extras, same shape |
| `TabDCEPairwisePipelineRunner` | `Xs_cfs_all` in extras, same shape |

```python
PAIRWISE_RUNNERS = [...]

@pytest.mark.parametrize("runner_cls,cfg_factory", PAIRWISE_RUNNERS)
def test_pairwise_extras_shape(runner_cls, cfg_factory, ...):
    result = runner.search_counterfactuals(...)
    cfs_all = result.extras["Xs_cfs_all"]
    n = result.X_test.shape[0]
    k = cfg["counterfactuals_params"]["cf_samples_per_factual"]
    assert cfs_all.shape == (n, k, synthetic_dataset.X_test.shape[1])
```

---

## Test: `test_smoke_special.py`

### CeFlowPipelineRunner

CeFlow has a custom `create_gen_model` that stores `self.flow_model` separately.
In the test, set `runner.flow_model` directly (the same `tiny_gen_model`):

```python
def test_ceflow_search_counterfactuals(ceflow_cfg, synthetic_dataset, tiny_disc_model,
                                       tiny_gen_model, tmp_path):
    runner = CeFlowPipelineRunner(ceflow_cfg, logger, preprocessing_pipeline=None)
    runner.flow_model = tiny_gen_model          # inject flow model directly
    result = runner.search_counterfactuals(
        dataset=synthetic_dataset,
        gen_model=tiny_gen_model,               # density model
        disc_model=tiny_disc_model,
        save_folder=str(tmp_path),
        log_prob_threshold=0.0,
    )
    _assert_valid_result(result, synthetic_dataset)
```

### LiCEPipelineRunner

LiCE requires the optional `spn` package. Gate with `pytest.importorskip`:

```python
def test_lice_search_counterfactuals(lice_cfg, synthetic_dataset, tiny_disc_model, tmp_path):
    spn = pytest.importorskip("spn", reason="LiCE requires optional 'spn' package")
    # LiCE also needs dataset.features to include target as last element
    ds = copy.copy(synthetic_dataset)
    ds.features = [f"f{i}" for i in range(6)] + ["target"]
    runner = LiCEPipelineRunner(lice_cfg, logger, preprocessing_pipeline=None)
    result = runner.search_counterfactuals(
        dataset=ds, gen_model=None, disc_model=tiny_disc_model,
        save_folder=str(tmp_path), log_prob_threshold=0.0,
    )
    _assert_valid_result(result, ds)
```

### PPCEFRPipelineRunner (regression)

PPCEFR overrides `run()` and uses `evaluate_cf_regression`. Test only
`search_counterfactuals`:

```python
def test_ppcefr_search_counterfactuals(ppcefr_cfg, regression_dataset, tiny_disc_model,
                                        tiny_gen_model, tmp_path):
    """Regression runner: y_orig/y_target are continuous floats."""
    runner = PPCEFRPipelineRunner(ppcefr_cfg, logger, preprocessing_pipeline=None)
    runner._delta = torch.tensor(-10.0)   # inject pre-computed delta
    result = runner.search_counterfactuals(
        dataset=regression_dataset, gen_model=tiny_gen_model, disc_model=tiny_disc_model,
        save_folder=str(tmp_path), log_prob_threshold=None,
    )
    assert result.X_cf.shape[0] == regression_dataset.X_test.shape[0]
    assert result.cf_search_time >= 0.0
```

---

## Helper: `_assert_valid_result`

Shared assertion function used by all tests:

```python
def _assert_valid_result(result: SearchResult, dataset) -> None:
    n = result.X_test.shape[0]
    n_feat = dataset.X_test.shape[1]
    assert isinstance(result, SearchResult)
    assert result.X_cf.shape     == (n, n_feat), f"X_cf shape mismatch: {result.X_cf.shape}"
    assert result.X_test.shape   == (n, n_feat)
    assert result.y_orig.shape   == (n,)
    assert result.y_target.shape == (n,)
    assert result.model_returned.shape == (n,)
    assert result.model_returned.dtype == bool or result.model_returned.dtype == np.bool_
    assert float(result.cf_search_time) >= 0.0
    assert isinstance(result.extras, dict)
```

---

## Config Factories

Each runner needs a unique `cfg` with its method-specific parameters.
These are thin `OmegaConf` dicts, merged on top of `base_cfg`.

| Runner | Key extra params |
|---|---|
| PPCEF | `disc_model_criterion._target_`, `alpha`, `alpha_s`, `alpha_k`, `patience`, `use_categorical=False`, `plausibility_weight=1.0`, `plausibility_bias=0.0` |
| WACH_OURS | `disc_model_criterion._target_`, `alpha` |
| WACH | `cf_method._target_`, `log_prob_quantile` |
| Artelt | none |
| CEGP | `beta=0.01`, `c_init=10.0`, `c_steps=5`, `max_iterations=50`, `feature_range=[-0.5, 1.5]`, `fit_d_type="mean"`, `fit_disc_perc=[25,50,75]` |
| CEM | `mode="PN"`, `kappa=0.0`, `beta=0.01`, `c_init=10.0`, `c_steps=5`, `max_iterations=50`, `learning_rate_init=0.01`, `fit_no_info_type="median"`, `feature_range=[-0.5,1.5]`, `clip_range=[-0.5,1.5]` |
| CET | none extra (uses `disc_model.predict_proba`) |
| TCREx | `origin_class=0`, `tau=0.5`, `rho=0.1`, `surrogate_tree_params={}` |
| PUMAL | `origin_class=0`, `disc_model_criterion._target_`, `cf_method.cf_method_type`, `cf_method.K`, `alpha_dist`, `alpha_plaus`, `alpha_class`, `alpha_s`, `alpha_k`, `alpha_d`, `decrease_loss_patience` |
| GLANCE | `cf_method.k=-1`, `cf_method.s=2`, `cf_method.m=1` |
| GLOBE_CE | none extra (uses dataset preprocessing_pipeline) |
| TabDCE | `tabdce.k_neighbors=3`, `tabdce.search_method="knn"`, `tabdce.hidden_dim=16`, `tabdce.T=10`, `tabdce.epochs=1`, `tabdce.lr=0.001`, `tabdce.batch_size=8`, `tabdce.use_gpu=False` |
| DiCE | `n_cfs=2` |
| CCHVAE | `latent_dim=4`, `hidden_sizes=[8]`, `vae_epochs=1`, `vae_lr=0.01` |
| CeFlow | `flow_model.*`, `use_categorical=False`, `alpha_min`, `alpha_max`, … |
| PPCEFR | `disc_loss._target_`, `target_change=1.0`, `alpha`, `log_prob_quantile=0.5` |

---

## Pytest Marks & Speed

Add a `pytest.ini` / `pyproject.toml` section:

```toml
[tool.pytest.ini_options]
markers = [
    "smoke: lightweight regression guards (all runners, fast)",
    "integration: heavier end-to-end tests",
]
```

All tests in `test_pipelines/` are marked `@pytest.mark.smoke`.
Default CI runs: `pytest -m smoke`.
Expensive runners (TabDCE diffusion training, CCHVAE VAE training) are marked
`@pytest.mark.slow` and skipped by default.

---

## Estimated Test Count

| File | Tests |
|---|---|
| `test_smoke_local.py` | ~12 parametrized |
| `test_smoke_group.py` | ~7 parametrized + 2 extras assertions |
| `test_smoke_pairwise.py` | ~3 parametrized + 3 shape assertions |
| `test_smoke_special.py` | 3 dedicated |
| **Total** | **~30 tests** |

---

## Implementation Order

1. `tests/test_pipelines/conftest.py` — fixtures (`synthetic_dataset`, `tiny_disc_model`, `tiny_gen_model`, `base_cfg`, `globe_ce_dataset`, `_assert_valid_result`)
2. `tests/test_pipelines/test_smoke_local.py` — start with `PPCEF` (simplest), then expand
3. `tests/test_pipelines/test_smoke_group.py`
4. `tests/test_pipelines/test_smoke_pairwise.py`
5. `tests/test_pipelines/test_smoke_special.py`
6. Update `pyproject.toml` with pytest marks

---

## Open Questions / Risks

| Issue | Mitigation |
|---|---|
| `TabDCEPipelineRunner` trains a diffusion model (slow even for 1 epoch) | Mark `@pytest.mark.slow`; skip in default CI |
| `CCHVAEPipelineRunner` trains a VAE internally during `search_counterfactuals` | Same — `@pytest.mark.slow` |
| `LiCEPipelineRunner` needs optional `spn` package | `pytest.importorskip("spn")` |
| `DiCEPipelineRunner` expects pandas-compatible wrapper around disc_model | Verify `DiscWrapper` class in `dice_runner.py` works with mock model |
| `GLOBECEPipelineRunner` uses `minmax_scaler._transform_array` (custom method) | Implement the `_MinMaxStep` adapter in `globe_ce_dataset` fixture |
| `CeFlowPipelineRunner` needs a flow model with specific transform API | Inject `runner.flow_model = tiny_gen_model`; add `transform_to_latent`/`transform_to_data` stubs if needed |
| `PUMALPipelineRunner` builds S/D matrices from `delta.get_matrices()` | Synthetic data with tiny K and short epochs; accept that matrices may be trivial |
