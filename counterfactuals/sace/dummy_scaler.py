
class DummyScaler:

    def __init__(self):
        pass

    def fit(self, X):
        pass

    def fit_transform(self, X):
        return X

    def transform(self, X):
        return X

    def inverse_transform(self, X):
        return X

