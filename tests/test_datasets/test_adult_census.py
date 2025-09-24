"""Minimal tests for AdultCensusDataset functionality."""

import numpy as np

from counterfactuals.datasets.adult_census import AdultCensusDataset


def test_adult_census_dataset_initialization():
    """Test that AdultCensusDataset initializes correctly with automatic config loading."""  
    dataset = AdultCensusDataset()

    # Check that data is loaded
    assert dataset.X is not None
    assert dataset.y is not None
    assert dataset.X.shape[0] == dataset.y.shape[0]

    # Check expected shapes for Adult Census dataset
    assert dataset.X.shape[1] == 12  # 12 features (4 continuous + 8 categorical)
    assert len(np.unique(dataset.y)) == 2  # Binary classification
    
    # Check sample limit
    assert dataset.X.shape[0] <= 32000  # Should be limited by SAMPLES_KEEP


def test_adult_census_dataset_train_test_split():
    """Test that train/test split is created automatically."""
    dataset = AdultCensusDataset()

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


def test_adult_census_dataset_config():
    """Test that config is loaded correctly."""
    dataset = AdultCensusDataset()

    # Check config exists
    assert dataset.config is not None
    assert hasattr(dataset.config, "features")
    assert hasattr(dataset.config, "continuous_features")
    assert hasattr(dataset.config, "categorical_features")
    assert hasattr(dataset.config, "target")

    # Check Adult Census-specific config
    assert len(dataset.config.features) == 12
    assert len(dataset.config.continuous_features) == 4
    assert len(dataset.config.categorical_features) == 8
    assert dataset.config.target == "class"

    # Check feature names
    expected_continuous = ["age", "capital-gain", "capital-loss", "hours-per-week"]
    expected_categorical = ["workclass", "education", "marital-status", 
                           "occupation", "relationship", "race", "sex", "native-country"]
    assert dataset.config.continuous_features == expected_continuous
    assert dataset.config.categorical_features == expected_categorical


def test_adult_census_dataset_feature_config():
    """Test that feature configuration is loaded correctly."""
    dataset = AdultCensusDataset()

    # Check feature config exists
    assert hasattr(dataset.config, "feature_config")
    assert "age" in dataset.config.feature_config
    assert "race" in dataset.config.feature_config
    assert "sex" in dataset.config.feature_config
    assert "native-country" in dataset.config.feature_config

    # Check actionable features (race, sex, native-country should not be actionable)
    assert dataset.config.feature_config["age"].actionable is True
    assert dataset.config.feature_config["workclass"].actionable is True
    assert dataset.config.feature_config["race"].actionable is False
    assert dataset.config.feature_config["sex"].actionable is False
    assert dataset.config.feature_config["native-country"].actionable is False


def test_adult_census_dataset_data_types():
    """Test that data types are correct."""
    dataset = AdultCensusDataset()

    # Check data types
    assert dataset.X.dtype == np.float32
    assert dataset.y.dtype == np.int64
    assert dataset.X_train.dtype == np.float32
    assert dataset.X_test.dtype == np.float32
    assert dataset.y_train.dtype == np.int64
    assert dataset.y_test.dtype == np.int64


def test_adult_census_dataset_target_values():
    """Test that target values are properly processed."""
    dataset = AdultCensusDataset()

    # Check target values (should be 0 for <=50K and 1 for >50K)
    unique_targets = np.unique(dataset.y)
    assert len(unique_targets) == 2
    assert 0 in unique_targets
    assert 1 in unique_targets


def test_adult_census_dataset_cv_splits():
    """Test cross-validation splits work."""
    dataset = AdultCensusDataset()

    splits = list(dataset.get_cv_splits(n_splits=3))
    assert len(splits) == 3

    for X_train, X_test, y_train, y_test in splits:
        assert X_train.shape[1] == 12
        assert X_test.shape[1] == 12
        assert X_train.shape[0] == y_train.shape[0]
        assert X_test.shape[0] == y_test.shape[0]


def test_adult_census_dataset_no_missing_values():
    """Test that no missing values exist after preprocessing."""
    dataset = AdultCensusDataset()

    # Check for NaN values
    assert not np.isnan(dataset.X).any()
    assert not np.isnan(dataset.y).any()


def test_adult_census_dataset_samples_limit():
    """Test that samples_keep parameter works."""
    dataset_default = AdultCensusDataset()
    dataset_limited = AdultCensusDataset(samples_keep=1000)

    # Limited dataset should be smaller
    assert dataset_limited.X.shape[0] <= 1000
    assert dataset_limited.X.shape[0] <= dataset_default.X.shape[0]


if __name__ == "__main__":
    # Simple test runner
    print("Testing AdultCensusDataset...")
    test_adult_census_dataset_initialization()
    print("✅ Initialization test passed")

    test_adult_census_dataset_train_test_split()
    print("✅ Train/test split test passed")

    test_adult_census_dataset_config()
    print("✅ Config test passed")

    test_adult_census_dataset_feature_config()
    print("✅ Feature config test passed")

    test_adult_census_dataset_data_types()
    print("✅ Data types test passed")

    test_adult_census_dataset_target_values()
    print("✅ Target values test passed")

    test_adult_census_dataset_cv_splits()
    print("✅ CV splits test passed")

    test_adult_census_dataset_no_missing_values()
    print("✅ No missing values test passed")

    test_adult_census_dataset_samples_limit()
    print("✅ Samples limit test passed")

    print("🎉 All AdultCensusDataset tests passed!")