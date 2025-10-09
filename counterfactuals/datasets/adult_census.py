from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from counterfactuals.datasets.base import DatasetBase

SAMPLES_KEEP = 32000


class AdultCensusDataset(DatasetBase):
    """Adult Census dataset loader compatible with DatasetBase."""

    # Path relative to this file: counterfactuals/datasets/adult_census.py
    # Go up 3 levels to project root, then to config
    CONFIG_PATH = (
        Path(__file__).resolve().parent.parent.parent
        / "config"
        / "datasets"
        / "adult_census.yaml"
    )

    def __init__(
        self,
        config_path: Path = CONFIG_PATH,
        samples_keep: int = SAMPLES_KEEP,
        shuffle: bool = True,
    ):
        """Initialize the Adult Census dataset with OmegaConf config.

        Args:
            config_path: Path to the dataset configuration file.
            samples_keep: Maximum number of samples to keep from the dataset.
            shuffle: Whether to shuffle data before splitting.
        """
        conf = OmegaConf.load(str(config_path))
        super().__init__(config=conf)
        self.samples_keep = samples_keep
        self.raw_data = self._load_csv(conf.raw_data_path)
        self.X, self.y = self.preprocess(self.raw_data)

        # Train/test split
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data(
            self.X, self.y
        )

    def _load_csv(self, file_path: str) -> pd.DataFrame:
        """Load dataset from CSV file.

        Args:
            file_path: Path to the CSV file (relative to project root).

        Returns:
            Loaded dataset as a pandas DataFrame.
        """
        # Resolve path relative to project root
        project_root = Path(__file__).resolve().parent.parent.parent
        path = project_root / file_path

        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        return pd.read_csv(path, index_col=False)

    def preprocess(self, raw_data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Preprocess raw data into feature and target arrays.

        Args:
            raw_data: Raw dataset as a pandas DataFrame.

        Returns:
            Tuple (X, y) as numpy arrays.
        """
        # Copy and clean data
        raw_data = raw_data.copy()
        raw_data = raw_data.dropna()
        raw_data = raw_data[: self.samples_keep]

        # Process target column (>50K vs <=50K)
        target_column = self.config.target
        raw_data[target_column] = raw_data[target_column].apply(
            lambda x: 1 if x.strip() == ">50K" else 0
        )

        # Process categorical features using label encoding for simplicity
        processed_data = raw_data.copy()
        for feature in self.config.categorical_features:
            if feature in processed_data.columns:
                # Convert categorical to numeric codes
                processed_data[feature] = pd.Categorical(processed_data[feature]).codes

        # Extract features and target
        X = processed_data[self.config.features].to_numpy().astype(np.float32)
        y = processed_data[target_column].to_numpy().astype(np.int64)

        return X, y
