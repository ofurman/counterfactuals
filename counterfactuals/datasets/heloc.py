import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

from counterfactuals.datasets.base import DatasetBase, DatasetParameters


class HelocDataset(DatasetBase):
    """HELOC dataset loader compatible with DatasetBase."""

    def __init__(self, config: DatasetParameters, transform: bool = True, sample_frac: float = 0.005):
        """Initialize the HELOC dataset.
        
        Args:
            config: Dataset configuration object.
            transform: Whether to apply MinMax scaling transformation.
            sample_frac: Fraction of data to sample (to reduce dataset size).
        """
        super().__init__(config)
        self.transform_data = transform
        self.sample_frac = sample_frac
        self.raw_data = self._load_csv(config.raw_data_path)
        self.X, self.y = self.preprocess(self.raw_data)
        
    def _load_csv(self, file_path: str) -> pd.DataFrame:
        """Load dataset from CSV file.
        
        Args:
            file_path: Path to the CSV file.
            
        Returns:
            Loaded dataset as a pandas DataFrame.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        # Load and sample the data
        data = pd.read_csv(path, index_col=False)
        if self.sample_frac < 1.0:
            data = data.sample(frac=self.sample_frac, random_state=42)
        return data

    def preprocess(self, raw_data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Preprocess raw data into feature and target arrays.
        
        Args:
            raw_data: Raw dataset as a pandas DataFrame.
            
        Returns:
            Tuple (X, y) as numpy arrays.
        """
        # Remove rows where all features are NaN
        raw_data = raw_data[(raw_data.iloc[:, 1:] >= 0).any(axis=1)].copy()
        
        # Process target column
        target_column = self.config.target
        raw_data[target_column] = (
            raw_data[target_column].replace({"Bad": "0", "Good": "1"}).astype(int)
        )
        
        # Handle missing values (negative values are treated as NaN)
        raw_data[raw_data < 0] = np.nan
        raw_data = raw_data.apply(lambda col: col.fillna(col.median()), axis=0)
        
        # Extract features and target
        X = raw_data[self.config.features].to_numpy().astype(np.float32)
        y = raw_data[target_column].to_numpy().astype(np.int64)
        
        # Apply transformation if requested
        if self.transform_data:
            scaler = MinMaxScaler()
            X = scaler.fit_transform(X)
            # Store scaler for potential future use
            self.feature_transformer = scaler
        
        return X, y
