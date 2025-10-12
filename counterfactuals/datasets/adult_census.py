from pathlib import Path

import numpy as np
import pandas as pd

from counterfactuals.datasets.base import DatasetBase

SAMPLES_KEEP = 30000


class AdultCensusDataset(DatasetBase):
    """Adult Census dataset loader compatible with DatasetBase."""

    CONFIG_PATH = (
        Path(__file__).resolve().parent.parent.parent
        / "config"
        / "datasets"
        / "adult_census.yaml"
    )

    def __init__(
        self,
        config_path: Path = CONFIG_PATH,
        transform: bool = True,
        samples_keep: int = SAMPLES_KEEP,
    ):
        """Initializes the Adult Census dataset with OmegaConf config.
        Args:
            config_path: Path to the dataset configuration file.
            transform: Whether to apply transformation.
            samples_keep: Number of samples to keep.
        """
        super().__init__(config_path=config_path)
        self.transform_data = transform
        self.samples_keep = samples_keep

        self.raw_data = self._load_csv(self.config.raw_data_path)
        self.X, self.y = self.preprocess(self.raw_data)
        print(self.X, self.y)

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
        """Preprocesses raw data into feature and target arrays.
        Args:
            raw_data: Raw dataset as a pandas DataFrame.
        Returns:
            Tuple (X, y) as numpy arrays.
        """
        raw_data = raw_data.dropna()
        raw_data = raw_data.head(self.samples_keep)
        raw_data[self.config.target] = raw_data[self.config.target].apply(lambda x: x.strip())
        raw_data[self.config.target] = raw_data[self.config.target].replace(
            {"<=50K": 0, ">50K": 1}
        )

        return raw_data[self.config.features].to_numpy(), raw_data[self.config.target].to_numpy()
