import pandas as pd
import numpy as np
from pathlib import Path

from counterfactuals.datasets.base import DatasetBase, DatasetParameters, FeatureParameters


class MoonsDataset(DatasetBase):
    """Moons dataset loader compatible with DatasetBase."""

    def __init__(self, config: DatasetParameters):
        """Initializes the Moons dataset.

        Args:
            config: Dataset configuration object.
        """
        super().__init__(config)
        self.raw_data = self._load_csv(config.raw_data_path)
        self.X, self.y = self.preprocess(self.raw_data)

    def _load_csv(self, file_path: str) -> pd.DataFrame:
        """Loads dataset from CSV file.

        Args:
            file_path: Path to the CSV file.

        Returns:
            Loaded dataset as a pandas DataFrame.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        return pd.read_csv(path, header=None)

    def preprocess(self, raw_data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Preprocess raw data into feature and target arrays.

        Args:
            raw_data: Raw dataset as a pandas DataFrame.

        Returns:
            Tuple (X, y) as numpy arrays.
        """
        X = raw_data.iloc[:, :-1].to_numpy()
        y = raw_data.iloc[:, -1].to_numpy()
        return X, y
