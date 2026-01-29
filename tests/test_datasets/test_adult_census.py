import os
import shutil
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch

from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

from counterfactuals.datasets.file_dataset import FileDataset
from counterfactuals.pipelines.run_ppcef_pipeline import (
    search_counterfactuals,
    calculate_metrics,
)
from counterfactuals.pipelines.run_globe_ce_pipeline import (
    search_counterfactuals as search_counterfactuals_globe_ce,
    calculate_metrics as calculate_metrics_globe_ce,
)
from counterfactuals.pipelines.full_pipeline.full_pipeline import full_pipeline
from counterfactuals.preprocessing import (
    PreprocessingPipeline,
    MinMaxScalingStep,
    TorchDataTypeStep,
)


def test_adult_census_initialization():
    # Construct absolute path to the config file
    # This file is located at tests/test_datasets/test_adult_census.py
    project_root = Path(__file__).parent.parent.parent
    config_path = project_root / "config/datasets/adult_census.yaml"
    
    # Initialize the dataset
    dataset = FileDataset(config_path=config_path)
    
    # Assertions to verify initialization
    assert dataset.X is not None
    assert dataset.y is not None
    assert len(dataset.X) > 0
    assert len(dataset.y) > 0
    assert dataset.X.shape[0] == dataset.y.shape[0]
    
    # Verify features are loaded
    assert len(dataset.features) > 0
    assert dataset.X.shape[1] == len(dataset.features)


def test_adult_census_ppcef_execution():
    """
    Integration test to verify that PPCEF pipeline runs on Adult Census dataset.
    This runs for 1 epoch on a tiny subset to ensure end-to-end execution works.
    """
    # Create a temporary directory for output
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Clear any existing Hydra instance
        GlobalHydra.instance().clear()
        
        # Calculate relative path to config directory
        config_dir = "../../counterfactuals/pipelines/conf"
        
        try:
            with initialize(version_base="1.2", config_path=config_dir):
                cfg = compose(
                    config_name="ppcef_config",
                    overrides=[
                        "dataset.config_path=config/datasets/adult_census.yaml",
                        "dataset.samples_keep=50",  # Tiny dataset for speed
                        "disc_model.train_model=true",
                        "disc_model.epochs=1",
                        "disc_model.batch_size=16",
                        "disc_model.patience=1",
                        "gen_model.train_model=true",
                        "gen_model.epochs=1",
                        "gen_model.batch_size=16",
                        "gen_model.patience=1",
                        "counterfactuals_params.epochs=1",
                        "counterfactuals_params.batch_size=16",
                        f"experiment.output_folder={tmp_dir}",
                    ]
                )
        except ValueError as e:
            # Handle the case where Hydra is already initialized
            # This shouldn't happen with GlobalHydra.clear() but good for safety
             pytest.skip(f"Hydra initialization failed: {e}")

        # Mock logging to clean up output
        with patch("logging.getLogger"):
             # Define preprocessing pipeline
            preprocessing_pipeline = PreprocessingPipeline(
                [
                    ("minmax", MinMaxScalingStep()),
                    ("torch_dtype", TorchDataTypeStep()),
                ]
            )
            
            # Run the full pipeline
            try:
                # We also need to mock logger passing if full_pipeline expects it
                from logging import getLogger
                test_logger = getLogger("test_logger")
                
                full_pipeline(
                    cfg,
                    preprocessing_pipeline,
                    test_logger,
                    search_counterfactuals,
                    calculate_metrics
                )
            except Exception as e:
                # Fail if it's not the environment issue
                if "tensorflow" in str(e).lower() or "dll" in str(e).lower():
                    # Likely the environment issue on windows
                    pytest.skip(f"Skipping due to environment issue: {e}")
                else:
                    pytest.fail(f"PPCEF pipeline execution failed with error: {e}")
            
            # Verify that output files were created in the temp directory
            output_path = Path(tmp_dir)
            files = list(output_path.glob("**/*.csv")) + list(output_path.glob("**/*.pt")) + list(output_path.glob("**/*.pth"))
            
            # We expect at least some model checkpoints or result csvs
            # If the pipeline ran successfully, files should exist.
            # However, if we skipped or failed, this assertion won't be reached or will fail.
            if len(files) == 0:
                 # Check if we should have failed earlier
                 pass
            assert len(files) > 0, "No output files were generated by the pipeline"


def test_adult_census_globe_ce_execution():
    """
    Integration test to verify that GLOBE-CE pipeline runs on Adult Census dataset.
    This runs for 1 epoch on a tiny subset to ensure end-to-end execution works.
    """
    # Create a temporary directory for output
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Clear any existing Hydra instance
        GlobalHydra.instance().clear()

        # Calculate relative path to config directory
        config_dir = "../../counterfactuals/pipelines/conf"

        try:
            with initialize(version_base="1.2", config_path=config_dir):
                cfg = compose(
                    config_name="globe_ce_config",
                    overrides=[
                        "dataset.config_path=config/datasets/adult_census.yaml",
                        "dataset.samples_keep=50",  # Tiny dataset for speed
                        "disc_model.train_model=true",
                        "disc_model.epochs=1",
                        "disc_model.batch_size=16",
                        "disc_model.patience=1",
                        "gen_model.train_model=true",
                        "gen_model.epochs=1",
                        "gen_model.batch_size=16",
                        "gen_model.patience=1",
                        "counterfactuals_params.limit_samples=2",                        
                        "counterfactuals_params.batch_size=16",
                        f"experiment.output_folder={tmp_dir}",
                    ]
                )
        except ValueError as e:
            pytest.skip(f"Hydra initialization failed: {e}")

        # GLOBE-CE usually uses a different preprocessing pipeline than PPCEF
        # It needs torch_dtype first, then minmax for some reason related to how data is handled?
        # Let's match run_globe_ce_pipeline.py which uses:
        # ("torch_dtype", TorchDataTypeStep()), ("minmax", MinMaxScalingStep())
        
        preprocessing_pipeline = PreprocessingPipeline(
            [
                ("torch_dtype", TorchDataTypeStep()),
                ("minmax", MinMaxScalingStep()),
            ]
        )

        with patch("logging.getLogger"):
            try:
                from logging import getLogger
                test_logger = getLogger("test_logger")

                full_pipeline(
                    cfg,
                    preprocessing_pipeline,
                    test_logger,
                    search_counterfactuals_globe_ce,
                    calculate_metrics_globe_ce
                )
            except Exception as e:
                 if "tensorflow" in str(e).lower() or "dll" in str(e).lower():
                    pytest.skip(f"Skipping due to environment issue: {e}")
                 else:
                    pytest.fail(f"GLOBE-CE pipeline execution failed with error: {e}")

            # Verify outputs
            output_path = Path(tmp_dir)
            files = list(output_path.glob("**/*.csv")) + list(output_path.glob("**/*.pt")) + list(output_path.glob("**/*.pth"))
            assert len(files) > 0, "No output files were generated by the pipeline"

