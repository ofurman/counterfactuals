import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from counterfactuals.datasets.base import AbstractDataset

class DatasetHeloc(AbstractDataset):
    def __init__(self, data=None):
        super().__init__(data)
        self.scaler = MinMaxScaler()
    
    def load(self, file_path):
        """
        Load data from a CSV file and store it in the 'data' attribute.
        """
        try:
            self.data = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error loading data from {file_path}: {e}")

    def preprocess(self):
        """
        Preprocess the loaded data by applying Min-Max scaling to all columns except the last one (target column).
        """
        if self.data is not None:
            X = self.data.iloc[:, :-1]
            y = self.data.iloc[:, -1]
            
            X_scaled = self.scaler.fit_transform(X)
            
            self.data = pd.DataFrame(data=X_scaled, columns=X.columns)
            self.data = pd.concat([self.data, y], axis=1)

    def save(self, file_path):
        """
        Save the processed data (including scaled features) to a CSV file.
        """
        if self.data is not None:
            try:
                self.data.to_csv(file_path, index=False)
                print(f"Data saved to {file_path}")
            except Exception as e:
                print(f"Error saving data to {file_path}: {e}")
        else:
            print("No data to save.")
