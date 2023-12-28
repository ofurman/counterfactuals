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
    
    @abstractmethod
    def get_split_data(self):
        """
        Return X_train, X_test, y_train, y_test.
        """

    @abstractmethod
    def train_dataloader(self):
        """
        Return train torch dataloader.
        """

    @abstractmethod
    def test_dataloader(self):
        """
        Return test torch dataloader.
        """