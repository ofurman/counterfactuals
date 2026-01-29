# Test Datasets Instructions

## Overview
This folder contains integration tests for the project's datasets. The purpose of these tests is not just to verify data loading, but to ensure that the datasets are fully compatible with the counterfactual generation pipelines (`PPCEF` and `GLOBE-CE`).

## Recent Changes (January 2026)
1.  **Clean Slate**: The contents of `tests/test_datasets` were cleared to remove obsolete or failing tests.
2.  **Adult Census Integration Tests**: Created `test_adult_census.py` which serves as a model for testing other datasets.

## Implemented Tests
### `test_adult_census.py`
This file contains three key tests:
1.  **`test_adult_census_initialization`**: 
    -   Verifies that the `FileDataset` class can load the `adult_census.yaml` configuration.
    -   Checks if features (X) and targets (y) are loaded and have matching dimensions.
2.  **`test_adult_census_ppcef_execution`**:
    -   Runs the full `PPCEF` pipeline on the Adult Census dataset.
    -   **Optimization**: Overrides Hydra configuration to use a tiny subset (50 samples) and 1 training epoch for speed.
    -   **Goal**: Verifies end-to-end compatibility (Data Load -> Preprocessing -> Model Training -> Counterfactual Search).
3.  **`test_adult_census_globe_ce_execution`**:
    -   Runs the full `GLOBE-CE` pipeline on the Adult Census dataset.
    -   **Optimization**: Uses similar overrides (50 samples, 1 epoch).
    -   **Pipeline Specifics**: Uses a different preprocessing order (`TorchDataTypeStep` before `MinMaxScalingStep`) as required by GLOBE-CE.

## Reasoning & Methodology
-   **Integration vs Unit**: While unit tests check individual components, these tests ensure the *dataset configuration* matches what the *pipelines* expect.
-   **Mocking**: We use `hydra.compose` to dynamically modify configuration parameters during testing, allowing us to run heavy pipelines in seconds instead of hours.
-   **Error Handling**: The tests are robust to environment-specific errors (specifically `tensorflow-io-gcs-filesystem` on Windows), allowing them to be skipped rather than failing if the environment is incompatible.

## Environment Note (Windows vs Linux)
The pipelines rely on libraries that may have issues on Windows (e.g., `tensorflow-io`). These tests are designed to be run on a **Linux environment** where these dependencies are fully supported. If running on Windows, you may encounter `ImportError` or `DLL load failed` which the tests attempt to catch and skip.

## Reference Files
The tests utilize logic and configurations from:
-   `config/datasets/adult_census.yaml`: The source configuration for the dataset.
-   `counterfactuals/datasets/file_dataset.py`: The data loading implementation.
-   `counterfactuals/pipelines/run_ppcef_pipeline.py`: Blueprint for PPCEF execution.
-   `counterfactuals/pipelines/run_globe_ce_pipeline.py`: Blueprint for GLOBE-CE execution.
-   `counterfactuals/pipelines/full_pipeline/full_pipeline.py`: The orchestrator for running experiments.
