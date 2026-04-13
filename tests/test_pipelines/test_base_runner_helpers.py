"""Unit tests for PipelineRunner helper methods and template-method hooks."""

from unittest.mock import MagicMock

import numpy as np
import pytest
from omegaconf import OmegaConf

from counterfactuals.pipelines.base_runner import CfMethodOutput, PipelineRunner, SearchResult

# ---------------------------------------------------------------------------
# Minimal concrete runner for testing
# ---------------------------------------------------------------------------


class _MockRunner(PipelineRunner):
    """Minimal concrete runner that records hook call order."""

    cf_method_name = "MockMethod"

    # Track which hooks were called and in what order
    call_log: list[str]

    def __init__(self, cfg, logger):
        super().__init__(cfg, logger)
        self.call_log = []

    def search_counterfactuals(
        self, dataset, gen_model, disc_model, save_folder, log_prob_threshold
    ):
        return self._default_search_counterfactuals(
            dataset, gen_model, disc_model, save_folder, log_prob_threshold
        )

    def create_cf_method(self, dataset, gen_model, disc_model):
        self.call_log.append("create_cf_method")
        return MagicMock(name="cf_method")

    def pre_cf_generation(self, cf_method, dataset):
        self.call_log.append("pre_cf_generation")

    def run_cf_method(self, cf_method, cf_dataloader, dataset, log_prob_threshold):
        self.call_log.append("run_cf_method")
        n = 5
        return CfMethodOutput(
            x_cfs=np.zeros((n, 4), dtype=np.float32),
            x_origs=np.zeros((n, 4), dtype=np.float32),
            y_origs=np.zeros(n, dtype=np.float32),
            y_targets=np.ones(n, dtype=np.float32),
        )

    def postprocess_cf_output(self, output, dataset):
        self.call_log.append("postprocess_cf_output")
        return output


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_cfg():
    return OmegaConf.create(
        {
            "disc_model": {
                "model": {"_target_": "counterfactuals.models.classifier.MLPClassifier"},
            },
            "counterfactuals_params": {
                "target_class": 1,
                "batch_size": 4,
            },
        }
    )


@pytest.fixture
def mock_dataset():
    """Dataset with 10 test samples split evenly between classes 0 and 1."""
    rng = np.random.default_rng(42)
    n_train, n_test, n_feat = 20, 10, 4
    ds = MagicMock()
    ds.X_train = rng.uniform(0, 1, (n_train, n_feat)).astype(np.float32)
    ds.X_test = rng.uniform(0, 1, (n_test, n_feat)).astype(np.float32)
    ds.y_train = np.zeros(n_train, dtype=np.float32)
    ds.y_test = np.array([0] * 5 + [1] * 5, dtype=np.float32)
    return ds


# ---------------------------------------------------------------------------
# Helper method tests
# ---------------------------------------------------------------------------


def test_get_disc_model_name(minimal_cfg):
    runner = _MockRunner(minimal_cfg, MagicMock())
    assert runner._get_disc_model_name() == "MLPClassifier"


def test_get_target_class(minimal_cfg):
    runner = _MockRunner(minimal_cfg, MagicMock())
    assert runner._get_target_class() == 1


def test_timed_search_sets_elapsed(minimal_cfg):
    runner = _MockRunner(minimal_cfg, MagicMock())
    with runner._timed_search() as timer:
        pass
    assert timer["elapsed"] >= 0.0


def test_save_results_uses_disc_model_name(minimal_cfg, tmp_path):
    runner = _MockRunner(minimal_cfg, MagicMock())
    metrics = {"accuracy": 0.9}
    runner.save_results(metrics, 1.23, str(tmp_path))
    saved = list(tmp_path.glob("cf_metrics_*.csv"))
    assert len(saved) == 1
    assert "MLPClassifier" in saved[0].name


# ---------------------------------------------------------------------------
# Template-method hook order tests
# ---------------------------------------------------------------------------


def test_hook_call_order(minimal_cfg, mock_dataset, tmp_path):
    """Hooks must be called in the documented order."""
    runner = _MockRunner(minimal_cfg, MagicMock())

    result = runner.search_counterfactuals(
        dataset=mock_dataset,
        gen_model=MagicMock(),
        disc_model=MagicMock(),
        save_folder=str(tmp_path),
        log_prob_threshold=0.0,
    )

    assert runner.call_log == [
        "create_cf_method",
        "pre_cf_generation",
        "run_cf_method",
        "postprocess_cf_output",
    ]
    assert isinstance(result, SearchResult)
    assert result.X_cf.shape == (5, 4)
    assert result.cf_search_time >= 0.0


def test_default_model_returned_when_none(minimal_cfg, mock_dataset, tmp_path):
    """When run_cf_method returns model_returned=None, all-True mask is used."""

    class _NoneModelReturnedRunner(_MockRunner):
        def run_cf_method(self, cf_method, cf_dataloader, dataset, log_prob_threshold):
            n = 5
            return CfMethodOutput(
                x_cfs=np.zeros((n, 4), dtype=np.float32),
                x_origs=np.zeros((n, 4), dtype=np.float32),
                y_origs=np.zeros(n, dtype=np.float32),
                y_targets=np.ones(n, dtype=np.float32),
                model_returned=None,
            )

    runner = _NoneModelReturnedRunner(minimal_cfg, MagicMock())
    result = runner.search_counterfactuals(
        dataset=mock_dataset,
        gen_model=MagicMock(),
        disc_model=MagicMock(),
        save_folder=str(tmp_path),
        log_prob_threshold=0.0,
    )
    assert result.model_returned.all()


def test_postprocess_can_modify_output(minimal_cfg, mock_dataset, tmp_path):
    """postprocess_cf_output modifications are reflected in SearchResult."""

    class _PostprocessRunner(_MockRunner):
        def postprocess_cf_output(self, output, dataset):
            output.x_cfs = output.x_cfs + 99.0
            return output

    runner = _PostprocessRunner(minimal_cfg, MagicMock())
    result = runner.search_counterfactuals(
        dataset=mock_dataset,
        gen_model=MagicMock(),
        disc_model=MagicMock(),
        save_folder=str(tmp_path),
        log_prob_threshold=0.0,
    )
    assert (result.X_cf == 99.0).all()


def test_create_cf_method_not_implemented(minimal_cfg):
    """Base PipelineRunner.create_cf_method raises NotImplementedError."""

    class _NoHookRunner(PipelineRunner):
        cf_method_name = "NoHook"

        def search_counterfactuals(self, *args, **kwargs):
            return self._default_search_counterfactuals(*args, **kwargs)

    runner = _NoHookRunner(minimal_cfg, MagicMock())
    with pytest.raises(NotImplementedError):
        runner.create_cf_method(MagicMock(), MagicMock(), MagicMock())


def test_run_cf_method_not_implemented(minimal_cfg):
    """Base PipelineRunner.run_cf_method raises NotImplementedError."""

    class _NoRunHookRunner(PipelineRunner):
        cf_method_name = "NoRunHook"

        def search_counterfactuals(self, *args, **kwargs):
            return self._default_search_counterfactuals(*args, **kwargs)

        def create_cf_method(self, *args, **kwargs):
            return MagicMock()

    runner = _NoRunHookRunner(minimal_cfg, MagicMock())
    with pytest.raises(NotImplementedError):
        runner.run_cf_method(MagicMock(), MagicMock(), MagicMock(), 0.0)
