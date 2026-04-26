import numpy as np
import pytest

from cel.preprocessing.base import PreprocessingContext
from cel.preprocessing.scalers import (
    MinMaxScalingStep,
    ScalingStep,
    StandardScalingStep,
)


@pytest.fixture()
def mixed_context() -> PreprocessingContext:
    """Context with 2 continuous (cols 0,1) and 1 categorical (col 2) feature."""
    rng = np.random.default_rng(0)
    X_train = rng.uniform(1, 10, size=(50, 3))
    X_train[:, 2] = rng.integers(0, 3, size=50).astype(float)  # categorical
    X_test = rng.uniform(1, 10, size=(20, 3))
    X_test[:, 2] = rng.integers(0, 3, size=20).astype(float)
    return PreprocessingContext(
        X_train=X_train,
        X_test=X_test,
        categorical_indices=[2],
        continuous_indices=[0, 1],
    )


@pytest.fixture()
def continuous_only_context() -> PreprocessingContext:
    rng = np.random.default_rng(1)
    X = rng.uniform(-5, 5, size=(40, 3))
    return PreprocessingContext(X_train=X, categorical_indices=[], continuous_indices=[0, 1, 2])


@pytest.fixture()
def no_continuous_context() -> PreprocessingContext:
    X = np.array([[0, 1], [1, 2], [2, 0]], dtype=float)
    return PreprocessingContext(X_train=X, categorical_indices=[0, 1], continuous_indices=[])


class TestMinMaxScalingStep:
    def test_default_range(self, mixed_context: PreprocessingContext):
        step = MinMaxScalingStep()
        step.fit(mixed_context)
        result = step.transform(mixed_context)
        cont = result.X_train[:, [0, 1]]
        assert cont.min() >= 0.0 - 1e-9
        assert cont.max() <= 1.0 + 1e-9

    def test_custom_range(self, mixed_context: PreprocessingContext):
        step = MinMaxScalingStep(feature_range=(-1.0, 1.0))
        step.fit(mixed_context)
        result = step.transform(mixed_context)
        cont = result.X_train[:, [0, 1]]
        assert cont.min() >= -1.0 - 1e-9
        assert cont.max() <= 1.0 + 1e-9

    def test_categorical_unchanged(self, mixed_context: PreprocessingContext):
        step = MinMaxScalingStep()
        step.fit(mixed_context)
        result = step.transform(mixed_context)
        np.testing.assert_array_equal(result.X_train[:, 2], mixed_context.X_train[:, 2])

    def test_round_trip(self, mixed_context: PreprocessingContext):
        step = MinMaxScalingStep()
        step.fit(mixed_context)
        transformed = step.transform(mixed_context)
        recovered = step.inverse_transform(transformed)
        np.testing.assert_allclose(recovered.X_train, mixed_context.X_train, atol=1e-10)

    def test_x_test_transformed(self, mixed_context: PreprocessingContext):
        step = MinMaxScalingStep()
        step.fit(mixed_context)
        result = step.transform(mixed_context)
        assert result.X_test is not None

    def test_x_test_none(self, continuous_only_context: PreprocessingContext):
        ctx = PreprocessingContext(
            X_train=continuous_only_context.X_train,
            X_test=None,
            categorical_indices=[],
            continuous_indices=[0, 1, 2],
        )
        step = MinMaxScalingStep()
        step.fit(ctx)
        result = step.transform(ctx)
        assert result.X_test is None

    def test_is_scaling_step(self):
        assert isinstance(MinMaxScalingStep(), ScalingStep)


class TestStandardScalingStep:
    def test_zero_mean_unit_var(self, continuous_only_context: PreprocessingContext):
        step = StandardScalingStep()
        step.fit(continuous_only_context)
        result = step.transform(continuous_only_context)
        np.testing.assert_allclose(result.X_train.mean(axis=0), 0.0, atol=1e-10)
        np.testing.assert_allclose(result.X_train.std(axis=0), 1.0, atol=1e-10)

    def test_categorical_unchanged(self, mixed_context: PreprocessingContext):
        step = StandardScalingStep()
        step.fit(mixed_context)
        result = step.transform(mixed_context)
        np.testing.assert_array_equal(result.X_train[:, 2], mixed_context.X_train[:, 2])

    def test_round_trip(self, mixed_context: PreprocessingContext):
        step = StandardScalingStep()
        step.fit(mixed_context)
        transformed = step.transform(mixed_context)
        recovered = step.inverse_transform(transformed)
        np.testing.assert_allclose(recovered.X_train, mixed_context.X_train, atol=1e-10)

    def test_is_scaling_step(self):
        assert isinstance(StandardScalingStep(), ScalingStep)


class TestNoContinuousFeatures:
    def test_transform_returns_unchanged(self, no_continuous_context: PreprocessingContext):
        step = MinMaxScalingStep()
        step.fit(no_continuous_context)
        result = step.transform(no_continuous_context)
        np.testing.assert_array_equal(result.X_train, no_continuous_context.X_train)

    def test_inverse_transform_returns_unchanged(self, no_continuous_context: PreprocessingContext):
        step = MinMaxScalingStep()
        step.fit(no_continuous_context)
        result = step.inverse_transform(no_continuous_context)
        np.testing.assert_array_equal(result.X_train, no_continuous_context.X_train)
