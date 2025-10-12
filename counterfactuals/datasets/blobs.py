from pathlib import Path

import numpy as np
import pandas as pd

from counterfactuals.datasets.base import DatasetBase


class BlobsDataset(DatasetBase):
    """Blobs dataset loader compatible with DatasetBase."""

    CONFIG_PATH = (
        Path(__file__).resolve().parent.parent.parent
        / "config"
        / "datasets"
        / "blobs.yaml"
    )

    def __init__(self, config_path: Path = CONFIG_PATH, transform: bool = True):
        """Initializes the Blobs dataset with OmegaConf config.

        Args:
            config_path: Path to the dataset configuration file.
            transform: Whether to apply MinMax scaling transformation.
        """
        super().__init__(config_path=config_path)
        self.transform_data = transform

        self.raw_data = self._load_csv(self.config.raw_data_path, header=None)
        self.X, self.y = self.preprocess(self.raw_data)

        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data(
            self.X, self.y
        )

    def preprocess(self, raw_data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Preprocesses raw data into feature and target arrays.

        Args:
            raw_data: Raw dataset as a pandas DataFrame.

        Returns:
            Tuple (X, y) as numpy arrays.
        """
        return raw_data[self.config.features].to_numpy(), raw_data[self.config.target].to_numpy()
