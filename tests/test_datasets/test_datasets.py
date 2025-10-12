"""Minimal tests for AdultCensusDataset functionality."""
import pytest
import numpy as np

from counterfactuals.datasets.base import DatasetBase
from counterfactuals.datasets.adult_census import AdultCensusDataset
from counterfactuals.datasets.adult import AdultDataset
from counterfactuals.datasets.audit import AuditDataset

TEST_DATASETS = [
    AdultCensusDataset,
    AdultDataset,
    AuditDataset,
]


@pytest.mark.parametrize("dataset", TEST_DATASETS)
def test_dataset_initialization(dataset: type[DatasetBase]):
    """Test that dataset initializes correctly."""
    dataset = dataset()

    assert dataset.X is not None
    assert dataset.y is not None
    assert dataset.X.shape[0] == dataset.y.shape[0]

    assert dataset.X.shape[1] == len(dataset.config.features)
    assert len(np.unique(dataset.y)) > 0


@pytest.mark.parametrize("dataset", TEST_DATASETS)
def test_dataset_train_test_split(dataset: type[DatasetBase]):
    """Test that train/test split is created automatically."""
    dataset = dataset()

    assert hasattr(dataset, "X_train")
    assert hasattr(dataset, "X_test")
    assert hasattr(dataset, "y_train")
    assert hasattr(dataset, "y_test")

    total_samples = dataset.X.shape[0]
    train_samples = dataset.X_train.shape[0]
    test_samples = dataset.X_test.shape[0]

    assert train_samples + test_samples == total_samples
    assert train_samples > 0 and test_samples > 0


@pytest.mark.parametrize("dataset", TEST_DATASETS)
def test_dataset_config(dataset: type[DatasetBase]):
    """Test that config is loaded correctly."""
    dataset = dataset()

    assert dataset.config is not None
    assert hasattr(dataset.config, "features")
    assert hasattr(dataset.config, "continuous_features")
    assert hasattr(dataset.config, "categorical_features")
    assert hasattr(dataset.config, "target")


@pytest.mark.parametrize("dataset", TEST_DATASETS)
def test_dataset_samples_keep(dataset: type[DatasetBase]):
    """Test that `samples_keep` parameter works."""
    if hasattr(dataset, "samples_keep"):
        dataset_full = dataset(samples_keep=1000)
        dataset_sampled = dataset(samples_keep=100)

        assert dataset_sampled.X.shape[0] < dataset_full.X.shape[0]
        assert dataset_sampled.X.shape[0] == 100
