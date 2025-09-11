from typing import List

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class GroupTransformer(BaseEstimator, TransformerMixin):
    """Group transformer for categorical features."""

    def __init__(
        self,
        groups: List[List[int]],
        transformer_factory,
    ):
        self.groups = groups
        self._make_transformer = transformer_factory

    def fit(self, X):
        """Fit one transformer per categorical group.

        Args:
            X: np.ndarray of shape (n_samples, n_features)
        """
        X = self._validate_X(X, for_fit=True)
        self.transformers_ = []
        for g in self.groups:
            transformer = self._make_transformer()
            transformer.fit(X[:, g])
            self.transformers_.append(transformer)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: np.ndarray, *, inplace: bool = False) -> np.ndarray:
        """Dequantize (forward) categorical groups."""
        return self._apply(X, direction="forward", inplace=inplace)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform the data."""
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X: np.ndarray, *, inplace: bool = False) -> np.ndarray:
        """Quantize (inverse) categorical groups."""
        return self._apply(X, direction="inverse", inplace=inplace)

    def _apply(self, X: np.ndarray, *, direction: str, inplace: bool) -> np.ndarray:
        check_is_fitted(self, ["transformers_", "n_features_in_"])
        X = self._validate_X(X, for_fit=False)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but fitted with {self.n_features_in_}."
            )

        out = X if inplace else X.copy()

        for g, transformer in zip(self.groups, self.transformers_):
            cols = g
            block = out[:, cols]
            if direction == "forward":
                block_t = transformer.transform(block)
            elif direction == "inverse":
                block_t = transformer.inverse_transform(block)
            else:
                raise ValueError("direction must be 'forward' or 'inverse'.")

            if block_t.shape != block.shape:
                raise ValueError(
                    f"Transformer for group {g} changed shape from {block.shape} "
                    f"to {block_t.shape}. Expected same width per group."
                )
            out[:, cols] = block_t

        return out

    @staticmethod
    def _validate_X(X: np.ndarray, *, for_fit: bool) -> np.ndarray:
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy.ndarray.")
        if X.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_features).")
        if X.shape[1] == 0:
            raise ValueError("X must have at least one feature.")
        return X
