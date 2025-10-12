from pathlib import Path

import numpy as np
import pandas as pd

from counterfactuals.datasets.base import DatasetBase


class AuditDataset(DatasetBase):
    """Audit dataset loader compatible with DatasetBase."""

    CONFIG_PATH = (
        Path(__file__).resolve().parent.parent.parent
        / "config"
        / "datasets"
        / "audit.yaml"
    )

    def __init__(self, config_path: Path = CONFIG_PATH, transform: bool = True):
        """Initializes the Audit dataset with OmegaConf config.

        Args:
            config_path: Path to the dataset configuration file.
            transform: Whether to apply MinMax scaling transformation.
        """
        super().__init__(config_path=config_path)
        self.transform_data = transform

        self.raw_data = self._load_csv(self.config.raw_data_path)
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
        features = self.config.features.copy()
        if "Detection_Risk" in features:
            features.remove("Detection_Risk")

        row_per_class = sum(raw_data[self.config.target] == 1)
        raw_data = pd.concat(
            [
                raw_data[raw_data[self.config.target] == 0].sample(
                    row_per_class, random_state=42
                ),
                raw_data[raw_data[self.config.target] == 1],
            ]
        )

        X = raw_data[features].to_numpy().astype(np.float32)
        y = raw_data[self.config.target].to_numpy().astype(np.int64)

        return X, y
