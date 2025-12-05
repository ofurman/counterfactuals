from typing import Optional

import numpy as np
from sklearn.preprocessing import OneHotEncoder

from counterfactuals.preprocessing.base import PreprocessingContext, PreprocessingStep


class OneHotEncodingStep(PreprocessingStep):
    """One-hot encoding for categorical features only.

    This step uses sklearn's OneHotEncoder internally and only transforms
    categorical features specified in the context.

    Attributes:
        encoder: Internal sklearn OneHotEncoder instance.
        n_features_in: Number of input features.
        n_features_out: Number of output features after encoding.
    """

    def __init__(self):
        """Initialize OneHotEncoder."""
        self.encoder: Optional[OneHotEncoder] = None
        self.n_features_in: Optional[int] = None
        self.n_features_out: Optional[int] = None
        self._categorical_indices: Optional[list[int]] = None
        self._continuous_indices: Optional[list[int]] = None

    def fit(self, context: PreprocessingContext) -> "OneHotEncodingStep":
        """Fit the encoder on categorical features from training data.

        Args:
            context: Preprocessing context with training data and feature indices.

        Returns:
            Self for method chaining.
        """
        self.n_features_in = context.X_train.shape[1]
        self._categorical_indices = context.categorical_indices
        self._continuous_indices = context.continuous_indices

        if len(self._categorical_indices) > 0:
            # Extract and fit on categorical features only
            X_cat = np.concatenate(
                (
                    context.X_train[:, self._categorical_indices],
                    context.X_test[:, self._categorical_indices],
                ),
                axis=0,
            )
            self.encoder = OneHotEncoder(
                sparse_output=False, handle_unknown="ignore", dtype=np.float64
            )
            self.encoder.fit(X_cat)

            # Calculate output feature count
            n_encoded = self.encoder.transform(X_cat).shape[1]
            self.n_features_out = len(self._continuous_indices) + n_encoded
        else:
            # No categorical features
            self.n_features_out = self.n_features_in

        return self

    def transform(self, context: PreprocessingContext) -> PreprocessingContext:
        """Transform categorical features by one-hot encoding.

        Args:
            context: Preprocessing context with data to transform.

        Returns:
            New context with transformed data.
        """
        if len(self._categorical_indices) == 0:
            # No categorical features, return unchanged
            return context

        # Transform train data
        X_train_transformed = self._transform_array(context.X_train)

        # Transform test data if present
        X_test_transformed = None
        if context.X_test is not None:
            X_test_transformed = self._transform_array(context.X_test)

        categorical_indices = list(range(self.n_features_out))[self._continuous_indices[-1] + 1 :]

        return PreprocessingContext(
            X_train=X_train_transformed,
            X_test=X_test_transformed,
            y_train=context.y_train,
            y_test=context.y_test,
            categorical_indices=categorical_indices,
            continuous_indices=self._continuous_indices,
        )

    def inverse_transform(self, context: PreprocessingContext) -> PreprocessingContext:
        """Inverse transform from one-hot encoded features back to original format.

        Args:
            context: Preprocessing context with transformed data.

        Returns:
            New context with inverse transformed data.
        """
        if len(self._categorical_indices) == 0:
            # No categorical features, return unchanged
            return context

        # Inverse transform train data
        X_train_inv = self._inverse_transform_array(context.X_train)

        # Inverse transform test data if present
        X_test_inv = None
        if context.X_test is not None:
            X_test_inv = self._inverse_transform_array(context.X_test)

        return PreprocessingContext(
            X_train=X_train_inv,
            X_test=X_test_inv,
            y_train=context.y_train,
            y_test=context.y_test,
            categorical_indices=self._categorical_indices,
            continuous_indices=self._continuous_indices,
        )

    def _transform_array(self, X: np.ndarray) -> np.ndarray:
        """Transform a single array by one-hot encoding categorical features.

        Args:
            X: Input array with shape (n_samples, n_features).

        Returns:
            Transformed array with categorical features one-hot encoded.
        """
        X_cont = X[:, self._continuous_indices]
        X_cat = X[:, self._categorical_indices]
        X_cat_encoded = self.encoder.transform(X_cat)

        # Reconstruct array maintaining original feature order
        result = np.zeros((X.shape[0], self.n_features_out))
        cont_out_idx = 0

        for i in range(self.n_features_in):
            if i in self._continuous_indices:
                cont_in_idx = self._continuous_indices.index(i)
                result[:, cont_out_idx] = X_cont[:, cont_in_idx]
                cont_out_idx += 1
            else:
                cat_in_idx = self._categorical_indices.index(i)
                n_categories = len(self.encoder.categories_[cat_in_idx])
                start_idx = sum(len(self.encoder.categories_[j]) for j in range(cat_in_idx))
                result[:, cont_out_idx : cont_out_idx + n_categories] = X_cat_encoded[
                    :, start_idx : start_idx + n_categories
                ]
                cont_out_idx += n_categories

        return result

    def _inverse_transform_array(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform a single array from one-hot encoded to original format.

        Args:
            X: Transformed array with one-hot encoded categorical features.

        Returns:
            Original format array with categorical features decoded.
        """
        # Use object dtype to handle mixed numeric and string data
        result = np.empty((X.shape[0], self.n_features_in), dtype=object)
        cont_out_idx = 0
        cat_encoded_cols = []

        # First pass: collect continuous features and one-hot encoded sections
        for i in range(self.n_features_in):
            if i in self._continuous_indices:
                result[:, i] = X[:, cont_out_idx]
                cont_out_idx += 1
            else:
                cat_in_idx = self._categorical_indices.index(i)
                n_categories = len(self.encoder.categories_[cat_in_idx])

                # Extract one-hot encoded columns for this categorical feature
                X_cat_part = X[:, cont_out_idx : cont_out_idx + n_categories]
                cat_encoded_cols.append(X_cat_part)
                cont_out_idx += n_categories

        # Use sklearn's inverse_transform for proper categorical decoding
        if cat_encoded_cols:
            X_cat_encoded = np.hstack(cat_encoded_cols)
            X_cat_decoded = self.encoder.inverse_transform(X_cat_encoded)

            # Put decoded categorical features back in their original positions
            cat_col_idx = 0
            for i in range(self.n_features_in):
                if i in self._categorical_indices:
                    result[:, i] = X_cat_decoded[:, cat_col_idx]
                    cat_col_idx += 1

        # Try to convert back to numeric if possible (all features are numeric)
        try:
            result = result.astype(np.float64)
        except (ValueError, TypeError):
            # Keep as object array if there are string values
            pass

        return result


class LabelOneHotEncodingStep(PreprocessingStep):
    """One-hot encode labels (y) using sklearn OneHotEncoder."""

    def __init__(self) -> None:
        self.encoder: Optional[OneHotEncoder] = None

    def fit(self, context: PreprocessingContext) -> "LabelOneHotEncodingStep":
        """Fit the encoder on available labels."""
        if context.y_train is None:
            raise ValueError("LabelOneHotEncodingStep requires y_train in the context.")

        y_train = context.y_train.reshape(-1, 1)
        if context.y_test is not None:
            y_all = np.concatenate([y_train, context.y_test.reshape(-1, 1)], axis=0)
        else:
            y_all = y_train

        self.encoder = OneHotEncoder(
            sparse_output=False, handle_unknown="ignore", dtype=np.float64
        )
        self.encoder.fit(y_all)
        return self

    def transform(self, context: PreprocessingContext) -> PreprocessingContext:
        """Transform labels to one-hot encoded arrays."""
        if self.encoder is None:
            raise ValueError("Call fit before transform in LabelOneHotEncodingStep.")

        y_train = (
            self.encoder.transform(context.y_train.reshape(-1, 1))
            if context.y_train is not None
            else None
        )
        y_test = (
            self.encoder.transform(context.y_test.reshape(-1, 1))
            if context.y_test is not None
            else None
        )

        return PreprocessingContext(
            X_train=context.X_train,
            X_test=context.X_test,
            y_train=y_train,
            y_test=y_test,
            categorical_indices=context.categorical_indices,
            continuous_indices=context.continuous_indices,
        )

    def inverse_transform(self, context: PreprocessingContext) -> PreprocessingContext:
        """Inverse transform one-hot labels back to original labels."""
        if self.encoder is None:
            raise ValueError(
                "Call fit before inverse_transform in LabelOneHotEncodingStep."
            )

        y_train = (
            self.encoder.inverse_transform(context.y_train)
            if context.y_train is not None
            else None
        )
        y_test = (
            self.encoder.inverse_transform(context.y_test)
            if context.y_test is not None
            else None
        )

        # Flatten back to 1D
        y_train = y_train.reshape(-1) if y_train is not None else None
        y_test = y_test.reshape(-1) if y_test is not None else None

        return PreprocessingContext(
            X_train=context.X_train,
            X_test=context.X_test,
            y_train=y_train,
            y_test=y_test,
            categorical_indices=context.categorical_indices,
            continuous_indices=context.continuous_indices,
        )
