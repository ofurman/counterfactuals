"""Minimal tests for AdultDataset functionality."""

import numpy as np

from counterfactuals.datasets.adult import AdultDataset


def test_adult_dataset_initialization():
    """Test that AdultDataset initializes correctly with automatic config loading."""
    dataset = AdultDataset()

    # Check that data is loaded
    assert dataset.X is not None
    assert dataset.y is not None
    assert dataset.X.shape[0] == dataset.y.shape[0]

    # Check expected shapes for Adult dataset
    assert dataset.X.shape[1] == 8  # 8 features (2 continuous + 6 categorical)
    assert len(np.unique(dataset.y)) == 2  # Binary classification


def test_adult_dataset_train_test_split():
    """Test that train/test split is created automatically."""
    dataset = AdultDataset()

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
    assert train_samples > 0 and test_samples > 0


def test_adult_dataset_config():
    """Test that config is loaded correctly."""
    dataset = AdultDataset()

    # Check config exists
    assert dataset.config is not None
    assert hasattr(dataset.config, "features")
    assert hasattr(dataset.config, "continuous_features")
    assert hasattr(dataset.config, "categorical_features")
    assert hasattr(dataset.config, "target")

    # Check Adult-specific config
    assert len(dataset.config.features) == 8
    assert len(dataset.config.continuous_features) == 2
    assert len(dataset.config.categorical_features) == 6
    assert dataset.config.target == "income"

    # Check feature names
    expected_continuous = ["age", "hours_per_week"]
    expected_categorical = ["workclass", "education", "marital_status", 
                           "occupation", "race", "gender"]
    assert dataset.config.continuous_features == expected_continuous
    assert dataset.config.categorical_features == expected_categorical


def test_adult_dataset_feature_config():
    """Test that feature configuration is loaded correctly."""
    dataset = AdultDataset()

    # Check feature config exists
    assert hasattr(dataset.config, "feature_config")
    assert "age" in dataset.config.feature_config
    assert "race" in dataset.config.feature_config
    assert "gender" in dataset.config.feature_config

    # Check actionable features (race and gender should not be actionable)
    assert dataset.config.feature_config["age"].actionable is True
    assert dataset.config.feature_config["race"].actionable is False
    assert dataset.config.feature_config["gender"].actionable is False


def test_adult_dataset_data_types():
    """Test that data types are correct."""
    dataset = AdultDataset()

    # Check data types
    assert dataset.X.dtype == np.float32
    assert dataset.y.dtype == np.int64
    assert dataset.X_train.dtype == np.float32
    assert dataset.X_test.dtype == np.float32
    assert dataset.y_train.dtype == np.int64
    assert dataset.y_test.dtype == np.int64


def test_adult_dataset_cv_splits():
    """Test cross-validation splits work."""
    dataset = AdultDataset()

    splits = list(dataset.get_cv_splits(n_splits=3))
    assert len(splits) == 3

    for X_train, X_test, y_train, y_test in splits:
        assert X_train.shape[1] == 8
        assert X_test.shape[1] == 8
        assert X_train.shape[0] == y_train.shape[0]
        assert X_test.shape[0] == y_test.shape[0]


def test_adult_dataset_no_missing_values():
    """Test that no missing values exist after preprocessing."""
    dataset = AdultDataset()

    # Check for NaN values
    assert not np.isnan(dataset.X).any()
    assert not np.isnan(dataset.y).any()


if __name__ == "__main__":
    # Simple test runner
    print("Testing AdultDataset...")
    test_adult_dataset_initialization()
    print("✅ Initialization test passed")

    test_adult_dataset_train_test_split()
    print("✅ Train/test split test passed")

    test_adult_dataset_config()
    print("✅ Config test passed")

    test_adult_dataset_feature_config()
    print("✅ Feature config test passed")

    test_adult_dataset_data_types()
    print("✅ Data types test passed")

    test_adult_dataset_cv_splits()
    print("✅ CV splits test passed")

    test_adult_dataset_no_missing_values()
    print("✅ No missing values test passed")

    print("🎉 All AdultDataset tests passed!")