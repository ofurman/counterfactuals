from pathlib import Path

import numpy as np
import pandas as pd

from counterfactuals.datasets.base import DatasetBase


class FileDataset(DatasetBase):
    """File dataset loader compatible with DatasetBase."""

    def __init__(
        self,
        config_path: Path,
    ):
        """Initializes the File dataset with OmegaConf config.
        Args:
            config_path: Path to the dataset configuration file.
            dataset_name: Optional name for the dataset (used for model paths).
        """
        super().__init__(config_path=config_path)
        self.samples_keep = self.config.samples_keep

        self.raw_data = self._load_csv(self.config.raw_data_path)
        self.X, self.y = self.preprocess(self.raw_data)

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
        raw_data = raw_data.dropna(subset=self.config.features)

        # Balance classes by downsampling majority class to match minority class
        target_col = self.config.target
        class_counts = raw_data[target_col].value_counts()
        min_class_count = class_counts.min()

        # Downsample each class to the minority class size
        balanced_dfs = []
        for class_value in class_counts.index:
            class_df = raw_data[raw_data[target_col] == class_value]
            if len(class_df) > min_class_count:
                class_df = class_df.sample(min_class_count, random_state=42)
            balanced_dfs.append(class_df)

        raw_data = pd.concat(balanced_dfs, ignore_index=True)

        if self.samples_keep > 0:
            raw_data = raw_data.sample(
                min(self.samples_keep, len(raw_data)), random_state=42
            )

        raw_data[self.config.target] = raw_data[self.config.target].replace(
            self.config.target_mapping
        )
        return raw_data[self.config.features].to_numpy(), raw_data[
            self.config.target
        ].to_numpy()
