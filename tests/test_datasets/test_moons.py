"""Minimal tests for MoonsDataset functionality."""

import numpy as np

from counterfactuals.datasets.moons import MoonsDataset


def test_moons_dataset_initialization():
    """Test that MoonsDataset initializes correctly with automatic config loading."""
    dataset = MoonsDataset()

    # Check that data is loaded
    assert dataset.X is not None
    assert dataset.y is not None
    assert dataset.X.shape[0] == dataset.y.shape[0]

    # Check expected shapes for moons dataset
    assert dataset.X.shape[1] == 2  # 2 features
    assert len(np.unique(dataset.y)) == 2  # Binary classification


def test_moons_dataset_train_test_split():
    """Test that train/test split is created automatically."""
    dataset = MoonsDataset()

    # Check that splits exist
    assert hasattr(dataset, "X_train")
    assert hasattr(dataset, "X_test")
    assert hasattr(dataset, "y_train")
    assert hasattr(dataset, "y_test")

    # Check split dimensions
    total_samples = dataset.X.shape[0]
    train_samples = dataset.X_train.shape[0]
    test_samples = dataset.X_test.shape[0]

    assert train_samples + test_samples == total_samples
    assert train_samples > test_samples  # Default 80/20 split


def test_moons_dataset_config():
    """Test that config is loaded correctly."""
    dataset = MoonsDataset()

    # Check config exists
    assert dataset.config is not None
    assert hasattr(dataset.config, "features")
    assert hasattr(dataset.config, "continuous_features")
    assert hasattr(dataset.config, "target")

    # Check moons-specific config
    assert dataset.config.features == [0, 1]
    assert dataset.config.continuous_features == [0, 1]


def test_moons_dataset_cv_splits():
    """Test cross-validation splits work."""
    dataset = MoonsDataset()

    splits = list(dataset.get_cv_splits(n_splits=3))
    assert len(splits) == 3

    for X_train, X_test, y_train, y_test in splits:
        assert X_train.shape[1] == 2
        assert X_test.shape[1] == 2
        assert X_train.shape[0] == y_train.shape[0]
        assert X_test.shape[0] == y_test.shape[0]


if __name__ == "__main__":
    # Simple test runner
    print("Testing MoonsDataset...")
    test_moons_dataset_initialization()
    print("✅ Initialization test passed")

    test_moons_dataset_train_test_split()
    print("✅ Train/test split test passed")

    test_moons_dataset_config()
    print("✅ Config test passed")

    test_moons_dataset_cv_splits()
    print("✅ CV splits test passed")

    print("🎉 All MoonsDataset tests passed!")
