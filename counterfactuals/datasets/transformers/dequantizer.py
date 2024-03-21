import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import OrdinalEncoder


class DequantizationTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, alpha=1e-5):
        super().__init__()
        self.alpha = alpha
        self.ordinal_encoder = OrdinalEncoder()

    def fit(self, X, y=None):
        self.ordinal_encoder.fit(X)
        self.category_counts = [
            len(self.ordinal_encoder.categories_[i])
            for i in range(len(self.ordinal_encoder.categories_))
        ]
        return self

    def transform(self, X: np.ndarray, y: np.ndarray = None):
        X = self.ordinal_encoder.transform(X)
        inputs_all = np.array(X).copy()
        inputs_cat = inputs_all
        inputs_cat = self._dequant(inputs_cat)
        inputs_cat = self._sigmoid(inputs_cat, reverse=True)
        inputs_all = inputs_cat
        return inputs_all

    def inverse_transform(self, X: np.ndarray, y: np.ndarray = None):
        inputs_all = X.copy()
        inputs_cat = inputs_all

        inputs_cat = self._sigmoid(inputs_cat, reverse=False)
        for i, count in enumerate(self.category_counts):
            inputs_cat[:, i] = inputs_cat[:, i] * count
            inputs_cat[:, i] = np.floor(inputs_cat[:, i]).clip(min=0, max=count - 1)

        inputs_all = inputs_cat
        return inputs_all

    def softplus_np(self, x):
        return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

    def sigmoid_np(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid(self, z, reverse=False):
        if not reverse:
            z = self.sigmoid_np(z)
            z = (z - 0.5 * self.alpha) / (1 - self.alpha)
        else:
            z = z * (1 - self.alpha) + 0.5 * self.alpha
            z = np.log(z) - np.log(1 - z)
        return z

    def _dequant(self, z):
        for i, count in enumerate(self.category_counts):
            z[:, i] = z[:, i] + np.random.random(z[:, i].shape)
            z[:, i] = z[:, i] / count
        return z
