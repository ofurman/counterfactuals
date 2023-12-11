from abc import ABC, abstractmethod

class AbstractDataset(ABC):
    def __init__(self, data=None):
        self.data = data

    @abstractmethod
    def load(self, file_path):
        """
        Load data from a file or source and store it in the 'data' attribute.
        """
        pass

    @abstractmethod
    def preprocess(self):
        """
        Preprocess the loaded data, if necessary.
        """
        pass

    @abstractmethod
    def save(self, file_path):
        """
        Save the processed data to a file or destination.
        """
        pass
