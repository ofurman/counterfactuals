from typing import List

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class GroupTransformer(BaseEstimator, TransformerMixin):
    """Applies a per-group transformer to categorical feature blocks.

    This meta-transformer manages multiple feature groups (column indices) such as one-hot encoded features, and
    instantiates one underlying transformer per group using the provided
    `transformer_factory`. Each group's transformer is fit on its respective
    slice of `X` and later used to (inverse-)transform only that slice.
    The overall shape of `X` is preserved.

    Attributes:
        groups (List[List[int]]): Column index groups, one list of indices per group.
        _make_transformer (Callable[[], TransformerMixin]): Factory creating a new
            (unfitted) transformer for a single group.
        transformers_ (List[TransformerMixin]): Fitted transformers, one per group.
        n_features_in_ (int): Number of features seen during fitting.
    """

    def __init__(
        self,
        groups: List[List[int]],
        transformer_factory,
    ):
        """Initialize the group transformer.

        Args:
            groups (List[List[int]]): Column index groups; each inner list contains
                indices belonging to one categorical group.
            transformer_factory (Callable[[], TransformerMixin]): Zero-argument factory
                returning a new transformer instance for a group (e.g., a dequantizer).

        Raises:
            ValueError: If `groups` is empty or contains an empty group.
        """
        if not groups:
            raise ValueError("`groups` must be a non-empty list of column index lists.")
        if any(len(g) == 0 for g in groups):
            raise ValueError("Each group in `groups` must contain at least one index.")
        self.groups = groups
        self.transformer_factory = transformer_factory

    def fit(self, X):
        """Fit one transformer per categorical group.

        Each group's transformer is created via `transformer_factory` and fit on
        `X[:, group]`.

        Args:
            X (np.ndarray): Array of shape (n_samples, n_features) with all features.

        Returns:
            GroupTransformer: This instance.

        Raises:
            TypeError: If `X` is not a NumPy array.
            ValueError: If `X` is not 2D or has zero columns.
        """
        X = self._validate_X(X, for_fit=True)
        self.transformers_ = []
        for g in self.groups:
            transformer = self.transformer_factory()
            transformer.fit(X[:, g])
            self.transformers_.append(transformer)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: np.ndarray, *, inplace: bool = False) -> np.ndarray:
        """Apply the forward transformation to each categorical group.

        Args:
            X (np.ndarray): Input array of shape (n_samples, n_features).
            inplace (bool, optional): If True, modify and return `X` in place;
                otherwise, work on and return a copy. Defaults to False.

        Returns:
            np.ndarray: Transformed array with the same shape as `X`.

        Raises:
            sklearn.exceptions.NotFittedError: If the transformer is not fitted.
            ValueError: If feature count differs from the fitted `n_features_in_`.
        """
        return self._apply(X, direction="forward", inplace=inplace)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit the transformers and return the forward-transformed data.

        Args:
            X (np.ndarray): Input array of shape (n_samples, n_features).

        Returns:
            np.ndarray: Transformed array with the same shape as `X`.
        """
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X: np.ndarray, *, inplace: bool = False) -> np.ndarray:
        """Apply the inverse transformation to each categorical group.

        Args:
            X (np.ndarray): Input array of shape (n_samples, n_features).
            inplace (bool, optional): If True, modify and return `X` in place;
                otherwise, work on and return a copy. Defaults to False.

        Returns:
            np.ndarray: Inverse-transformed array with the same shape as `X`.

        Raises:
            sklearn.exceptions.NotFittedError: If the transformer is not fitted.
            ValueError: If feature count differs from the fitted `n_features_in_`.
        """
        return self._apply(X, direction="inverse", inplace=inplace)

    def _apply(self, X: np.ndarray, *, direction: str, inplace: bool) -> np.ndarray:
        """Dispatch to forward or inverse transform on each group.

        Args:
            X (np.ndarray): Input array of shape (n_samples, n_features).
            direction (str): Either `"forward"` to call each group's `.transform`
                or `"inverse"` to call `.inverse_transform`.
            inplace (bool): If True, modify the provided `X`; otherwise operate on a copy.

        Returns:
            np.ndarray: Array with the same shape as `X`, transformed per group.

        Raises:
            sklearn.exceptions.NotFittedError: If the transformer is not fitted.
            ValueError: If `direction` is not `"forward"` or `"inverse"`.
            ValueError: If `X` feature width differs from `n_features_in_`.
            ValueError: If any group's transformer changes the group's width.
        """
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
        """Validate input array shape and type.

        Args:
            X (np.ndarray): Candidate array.
            for_fit (bool): Unused flag indicating whether validation is for fitting; kept
                for API completeness.

        Returns:
            np.ndarray: The validated input (unchanged).

        Raises:
            TypeError: If `X` is not a NumPy array.
            ValueError: If `X` is not 2D or has zero columns.
        """
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy.ndarray.")
        if X.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_features).")
        if X.shape[1] == 0:
            raise ValueError("X must have at least one feature.")
        return X
