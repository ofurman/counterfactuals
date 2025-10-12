"""Minimal tests for AdultCensusDataset functionality."""

import numpy as np
import pytest

from counterfactuals.datasets.regression_file_dataset import RegressionFileDataset

CONFIG_PATHs = [
    "config/datasets/toy_regression.yaml",
    "config/datasets/concrete.yaml",
    "config/datasets/diabetes.yaml",
    "config/datasets/wine_quality_regression.yaml",
    "config/datasets/yacht.yaml",
]


@pytest.mark.parametrize("config_path", CONFIG_PATHs)
def test_dataset_initialization(config_path: str):
    """Test that dataset initializes correctly."""
    dataset = RegressionFileDataset(config_path=config_path)

    assert dataset.X is not None
    assert dataset.y is not None
    assert dataset.X.shape[0] == dataset.y.shape[0]

    assert dataset.X.shape[1] == len(dataset.config.features)
    assert len(np.unique(dataset.y)) > 0


@pytest.mark.parametrize("config_path", CONFIG_PATHs)
def test_dataset_train_test_split(config_path: str):
    """Test that train/test split is created automatically."""
    dataset = RegressionFileDataset(config_path=config_path)

    assert hasattr(dataset, "X_train")
    assert hasattr(dataset, "X_test")
    assert hasattr(dataset, "y_train")
    assert hasattr(dataset, "y_test")

    total_samples = dataset.X.shape[0]
    train_samples = dataset.X_train.shape[0]
    test_samples = dataset.X_test.shape[0]

    assert train_samples + test_samples == total_samples
    assert train_samples > 0 and test_samples > 0


@pytest.mark.parametrize("config_path", CONFIG_PATHs)
def test_dataset_config(config_path: str):
    """Test that config is loaded correctly."""
    dataset = RegressionFileDataset(config_path=config_path)

    assert dataset.config is not None
    assert hasattr(dataset.config, "features")
    assert hasattr(dataset.config, "continuous_features")
    assert hasattr(dataset.config, "categorical_features")
    assert hasattr(dataset.config, "target")


@pytest.mark.parametrize("config_path", CONFIG_PATHs)
def test_dataset_samples_keep(config_path: str):
    """Test that `samples_keep` parameter works."""
    if hasattr(RegressionFileDataset, "samples_keep"):
        dataset_full = RegressionFileDataset(config_path=config_path, samples_keep=1000)
        dataset_sampled = RegressionFileDataset(
            config_path=config_path, samples_keep=100
        )

        assert dataset_sampled.X.shape[0] < dataset_full.X.shape[0]
        assert dataset_sampled.X.shape[0] == 100
