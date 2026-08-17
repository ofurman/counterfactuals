"""Tests for SpaceAdapterClassifier."""

import numpy as np
import torch

from counterfactuals.models.space_adapter import SpaceAdapterClassifier


class _AffineSpace:
    """Stand-in for MethodDataset: an invertible per-feature affine map."""

    def __init__(self, scale: float, shift: float):
        self.scale = scale
        self.shift = shift

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.shift) / self.scale

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return X * self.scale + self.shift


class _RecordingClassifier(torch.nn.Module):
    """Records what it was asked to predict on."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)
        self.seen: list[np.ndarray] = []

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.linear(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.seen.append(np.asarray(X))
        return np.zeros(len(X), dtype=np.int64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.seen.append(np.asarray(X))
        return np.full((len(X), 2), 0.5)


def _make_adapter(clf):
    # Caller works in a [0, 1]-ish space, classifier in a z-scored-ish one.
    caller = _AffineSpace(scale=10.0, shift=5.0)
    model = _AffineSpace(scale=2.0, shift=-1.0)
    return SpaceAdapterClassifier(clf, caller_dataset=caller, model_dataset=model), caller, model


def test_predict_converts_through_original_units():
    clf = _RecordingClassifier()
    adapter, caller, model = _make_adapter(clf)
    X_caller = np.random.default_rng(0).normal(size=(4, 3)).astype(np.float32)

    adapter.predict(X_caller)

    expected = model.transform(caller.inverse_transform(X_caller))
    np.testing.assert_allclose(clf.seen[-1], expected, rtol=1e-5)


def test_predict_proba_accepts_tensors():
    clf = _RecordingClassifier()
    adapter, caller, model = _make_adapter(clf)
    X_caller = torch.randn(4, 3)

    probs = adapter.predict_proba(X_caller)

    expected = model.transform(caller.inverse_transform(X_caller.numpy()))
    np.testing.assert_allclose(clf.seen[-1], expected, rtol=1e-5)
    assert probs.shape == (4, 2)


def test_forward_matches_base_model_on_converted_input():
    clf = _RecordingClassifier()
    adapter, caller, model = _make_adapter(clf)
    X_caller = np.random.default_rng(1).normal(size=(4, 3)).astype(np.float32)

    out = adapter(X_caller)

    expected_in = torch.from_numpy(
        model.transform(caller.inverse_transform(X_caller)).astype(np.float32)
    )
    with torch.no_grad():
        expected = clf(expected_in)
    torch.testing.assert_close(out, expected)


def test_delegates_other_attributes():
    clf = _RecordingClassifier()
    adapter, _, _ = _make_adapter(clf)
    assert list(adapter.parameters()) == list(clf.parameters())
    assert adapter.to("cpu") is adapter
