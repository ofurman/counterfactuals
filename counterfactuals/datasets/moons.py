import pandas as pd
import numpy as np
from pathlib import Path

from counterfactuals.datasets.base import DatasetBase, DatasetParameters, FeatureParameters


class MoonsDataset(DatasetBase):
    """Moons dataset loader compatible with DatasetBase."""

    # Path relative to this file: counterfactuals/datasets/moons.py
    # Go up 3 levels to project root, then to config
    CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "datasets" / "moons.yaml"

    def __init__(self, config_path: Path = CONFIG_PATH):
        """Initializes the Moons dataset with YAML config.

        Args:
            config_path: Path to the dataset configuration file.
        """
        config = self._load_config(config_path)
        super().__init__(config=config)

        self.raw_data = self._load_csv(config.raw_data_path)
        self.X, self.y = self.preprocess(self.raw_data)

        # Train/test split
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data(
            self.X, self.y
        )

    def _load_csv(self, file_path: str) -> pd.DataFrame:
        """Loads dataset from CSV file.

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
        return pd.read_csv(path, header=None)
