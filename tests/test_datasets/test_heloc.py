"""Minimal tests for HelocDataset functionality."""

import pytest
import numpy as np
from counterfactuals.datasets.heloc import HelocDataset


def test_heloc_dataset_initialization():
    """Test that HelocDataset initializes correctly with automatic config loading."""
    dataset = HelocDataset()
    
    # Check that data is loaded
    assert dataset.X is not None
    assert dataset.y is not None
    assert dataset.X.shape[0] == dataset.y.shape[0]
    
    # Check expected shapes for HELOC dataset
    assert dataset.X.shape[1] == 23  # 23 features
    assert len(np.unique(dataset.y)) == 2  # Binary classification


def test_heloc_dataset_train_test_split():
    """Test that train/test split is created automatically."""
    dataset = HelocDataset()
    
    # Check that splits exist
    assert hasattr(dataset, 'X_train')
    assert hasattr(dataset, 'X_test')
    assert hasattr(dataset, 'y_train')
    assert hasattr(dataset, 'y_test')
    
    # Check split dimensions
    total_samples = dataset.X.shape[0]
    train_samples = dataset.X_train.shape[0]
    test_samples = dataset.X_test.shape[0]
    
    assert train_samples + test_samples == total_samples
    assert train_samples > 0 and test_samples > 0


def test_heloc_dataset_config():
    """Test that config is loaded correctly."""
    dataset = HelocDataset()
    
    # Check config exists
    assert dataset.config is not None
    assert hasattr(dataset.config, 'features')
    assert hasattr(dataset.config, 'continuous_features')
    assert hasattr(dataset.config, 'target')
    
    # Check HELOC-specific config
    assert len(dataset.config.features) == 23
    assert dataset.config.target == "RiskPerformance"


def test_heloc_dataset_transform():
    """Test data transformation (MinMaxScaler)."""
    dataset = HelocDataset(transform=True)
    
    # Check that data is scaled (should be approximately in [0, 1] range)
    assert np.all(dataset.X >= 0)
    assert np.all(dataset.X <= 1.0001)  # Allow for small floating point precision
    
    # Check that transformer is stored
    assert hasattr(dataset, 'feature_transformer')


def test_heloc_dataset_sample_frac():
    """Test that sampling fraction works."""
    dataset_full = HelocDataset(sample_frac=1.0)
    dataset_sampled = HelocDataset(sample_frac=0.1)
    
    # Sampled dataset should be smaller
    assert dataset_sampled.X.shape[0] < dataset_full.X.shape[0]


if __name__ == "__main__":
    # Simple test runner
    print("Testing HelocDataset...")
    test_heloc_dataset_initialization()
    print("✅ Initialization test passed")
    
    test_heloc_dataset_train_test_split()
    print("✅ Train/test split test passed")
    
    test_heloc_dataset_config()
    print("✅ Config test passed")
    
    test_heloc_dataset_transform()
    print("✅ Transform test passed")
    
    test_heloc_dataset_sample_frac()
    print("✅ Sample fraction test passed")
    
    print("🎉 All HelocDataset tests passed!")