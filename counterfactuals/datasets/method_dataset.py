from __future__ import annotations

from typing import Generator, Optional, Tuple

import numpy as np
import torch
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

from counterfactuals.datasets.file_dataset import FileDataset
from counterfactuals.datasets.initial_transforms import InitialTransformPipeline
from counterfactuals.preprocessing.base import PreprocessingContext, PreprocessingStep


class MethodDataset:
    """Dataset wrapper combining raw loading, initial transforms, and preprocessing.

    This class provides a unified interface for loading raw data via FileDataset,
    applying the configured initial transforms, performing the train/test split,
    and running optional preprocessing transformations suitable for counterfactual methods.

    If no preprocessing pipeline is provided, split data is returned without additional
    transformations beyond the initial dataset-level steps.

    Attributes:
        file_dataset: Underlying FileDataset instance.
        preprocessing_pipeline: Optional pipeline of preprocessing transformations.
        X_train_raw: Raw training features before preprocessing.
        X_test_raw: Raw test features before preprocessing.
        y_train: Training labels.
        y_test: Test labels.
        X_train: Training features (preprocessed if pipeline provided, raw otherwise).
        X_test: Test features (preprocessed if pipeline provided, raw otherwise).
    """

    def __init__(
        self,
        file_dataset: FileDataset,
        preprocessing_pipeline: Optional[PreprocessingStep] = None,
    ):
        """Initialize MethodDataset.

        Args:
            config_path: Path to the dataset configuration file.
            preprocessing_pipeline: Optional preprocessing pipeline. If None, no
                preprocessing is applied and raw data is returned.
        """
        self.file_dataset = file_dataset
        self.initial_transform_pipeline: Optional[InitialTransformPipeline] = (
            file_dataset.initial_transform_pipeline
        )

        # Split raw data into train/test sets
        (
            X_train,
            X_test,
            y_train,
            y_test,
        ) = self.file_dataset.split_data(self.file_dataset.X, self.file_dataset.y)
        self.X_train_raw = X_train.copy()
        self.X_test_raw = X_test.copy()
        self.y_train = y_train.copy()
        self.y_test = y_test.copy()

        # Store preprocessing pipeline (may be None)
        self.preprocessing_pipeline = preprocessing_pipeline

        print(self.X_test_raw.shape)

        # Apply preprocessing if pipeline provided
        if self.preprocessing_pipeline is not None:
            # Create context with raw data and feature indices
            context = PreprocessingContext(
                X_train=self.X_train_raw,
                X_test=self.X_test_raw,
                y_train=self.y_train,
                y_test=self.y_test,
                categorical_indices=self.file_dataset.categorical_features_indices,
                continuous_indices=self.file_dataset.numerical_features_indices,
            )

            # Fit and transform
            self.preprocessing_pipeline.fit(context)
            transformed_context = self.preprocessing_pipeline.transform(context)

            # Extract transformed data
            self.X_train = transformed_context.X_train
            self.X_test = transformed_context.X_test
            self.y_train = transformed_context.y_train
            self.y_test = transformed_context.y_test
            self.file_dataset.categorical_features_indices = (
                transformed_context.categorical_indices
            )
            self.file_dataset.numerical_features_indices = (
                transformed_context.continuous_indices
            )
        else:
            # No preprocessing, use raw data
            self.X_train = self.X_train_raw
            self.X_test = self.X_test_raw
            # y_train and y_test already set above from file_dataset
            # Feature indices remain unchanged from file_dataset

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Apply inverse preprocessing to recover original feature space.

        Args:
            X: Preprocessed features.

        Returns:
            Features in original space.
        """
        if self.preprocessing_pipeline is None:
            # No preprocessing was applied, return as is
            return X

        # Create context with the data to inverse transform
        context = PreprocessingContext(
            X_train=X,
            categorical_indices=self.file_dataset.categorical_features_indices,
            continuous_indices=self.file_dataset.numerical_features_indices,
        )

        # Apply inverse transform
        inv_context = self.preprocessing_pipeline.inverse_transform(context)
        return inv_context.X_train

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply preprocessing to new data.

        Args:
            X: Raw features in original space.

        Returns:
            Preprocessed features.
        """
        if self.preprocessing_pipeline is None:
            # No preprocessing, return as is
            return X

        # Create context with the data to transform
        context = PreprocessingContext(
            X_train=X,
            categorical_indices=self.file_dataset.categorical_features_indices,
            continuous_indices=self.file_dataset.numerical_features_indices,
        )

        # Apply transform
        transformed_context = self.preprocessing_pipeline.transform(context)
        return transformed_context.X_train

    @property
    def features(self):
        """Return feature names from the underlying dataset."""
        return self.file_dataset.features

    @property
    def numerical_features(self):
        """Return numerical feature names from the underlying dataset."""
        return self.file_dataset.numerical_features

    @property
    def categorical_features(self):
        """Return categorical feature names from the underlying dataset."""
        return self.file_dataset.categorical_features

    @property
    def numerical_features_indices(self):
        """Return numerical feature indices from the underlying dataset."""
        return self.file_dataset.numerical_features_indices

    @property
    def categorical_features_indices(self):
        """Return categorical feature indices from the underlying dataset."""
        return self.file_dataset.categorical_features_indices

    @property
    def actionable_features(self):
        """Return actionable feature names from the underlying dataset."""
        return self.file_dataset.actionable_features

    @property
    def config(self):
        """Return dataset configuration."""
        return self.file_dataset.config

    def __repr__(self) -> str:
        """String representation of MethodDataset."""
        preprocessing_status = (
            f"preprocessing={self.preprocessing_pipeline}"
            if self.preprocessing_pipeline
            else "preprocessing=None (raw data)"
        )
        return (
            f"MethodDataset(\n"
            f"  features={len(self.features)},\n"
            f"  train_samples={len(self.X_train)},\n"
            f"  test_samples={len(self.X_test)},\n"
            f"  {preprocessing_status}\n"
            f")"
        )

    @property
    def categorical_features_lists(self) -> list:
        """
        Return categorical feature groupings based on one-hot encoded features.

        Computes the indices of one-hot encoded features for each original categorical
        variable in the current preprocessed data space. This property dynamically
        reads from the current preprocessing pipeline state, so it correctly reflects
        the structure after refitting in cross-validation folds.

        Returns:
            list: List of lists, where each inner list contains the indices of
                  one-hot encoded features for each original categorical variable.

        Raises:
            ValueError: If preprocessing pipeline is None or onehot step not found.
            AttributeError: If onehot encoder has not been fitted yet.
        """
        if self.preprocessing_pipeline is not None:
            onehot_step = self.preprocessing_pipeline.get_step("onehot")
            if onehot_step is not None and onehot_step.encoder is not None:
                categorical_features_lists = []
                current_idx = len(self.numerical_features_indices)

                for categories in onehot_step.encoder.categories_:
                    n_categories = len(categories)
                    categorical_features_lists.append(
                        list(range(current_idx, current_idx + n_categories))
                    )
                    current_idx += n_categories

                return categorical_features_lists

        return self._categorical_lists_from_initial_encoding()

    def _categorical_lists_from_initial_encoding(self) -> list[list[int]]:
        """Return categorical groupings derived from initial one-hot transforms."""
        groups = getattr(self.file_dataset, "one_hot_feature_groups", None)
        if not groups:
            return []

        feature_indices = {name: idx for idx, name in enumerate(self.features)}
        categorical_lists: list[list[int]] = []
        for columns in groups.values():
            indices = [
                feature_indices[column]
                for column in columns
                if column in feature_indices
            ]
            if indices:
                categorical_lists.append(indices)
        return categorical_lists

    def get_cv_splits(
        self, n_splits: int = 5, shuffle: bool = True
    ) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], None, None]:
        """Generates stratified cross-validation splits.

        Note: This method updates instance variables (X_train, X_test, y_train, y_test)
        with each fold's data. After iteration completes, these will contain the last fold's data.

        Args:
            n_splits: Number of folds.
            shuffle: Whether to shuffle data before splitting.

        Yields:
            Tuples of (X_train, X_test, y_train, y_test) for each fold.

        Raises:
            ValueError: If preprocess has not been called before.
        """
        if self.file_dataset.X is None or self.file_dataset.y is None:
            raise ValueError("Call preprocess() before generating CV splits.")

        if self.file_dataset.task_type == "classification":
            cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=42)
            for train_idx, test_idx in cv.split(
                self.file_dataset.X, self.file_dataset.y
            ):
                X_train_fold = self.file_dataset.X[train_idx].copy()
                X_test_fold = self.file_dataset.X[test_idx].copy()
                y_train_fold = self.file_dataset.y[train_idx].copy()
                y_test_fold = self.file_dataset.y[test_idx].copy()

                categorical_indices_raw = [
                    self.file_dataset.features.index(f)
                    for f in self.file_dataset.categorical_features
                ]
                continuous_indices_raw = [
                    self.file_dataset.features.index(f)
                    for f in self.file_dataset.numerical_features
                ]

                if self.preprocessing_pipeline is not None:
                    context = PreprocessingContext(
                        X_train=X_train_fold,
                        X_test=X_test_fold,
                        y_train=y_train_fold,
                        y_test=y_test_fold,
                        categorical_indices=categorical_indices_raw,
                        continuous_indices=continuous_indices_raw,
                    )

                    self.preprocessing_pipeline.fit(context)
                    transformed_context = self.preprocessing_pipeline.transform(context)

                    # Update instance variables with current fold's data
                    self.X_train = transformed_context.X_train
                    self.X_test = transformed_context.X_test
                    self.y_train = transformed_context.y_train
                    self.y_test = transformed_context.y_test
                    # Update feature indices to reflect the current preprocessed state
                    self.file_dataset.categorical_features_indices = (
                        transformed_context.categorical_indices
                    )
                    self.file_dataset.numerical_features_indices = (
                        transformed_context.continuous_indices
                    )
                    yield self.X_train, self.X_test, self.y_train, self.y_test
                else:
                    # Update instance variables with current fold's raw data
                    self.X_train = X_train_fold
                    self.X_test = X_test_fold
                    self.y_train = y_train_fold
                    self.y_test = y_test_fold
                    yield self.X_train, self.X_test, self.y_train, self.y_test
        else:
            cv = KFold(n_splits=n_splits, shuffle=shuffle, random_state=42)
            for train_idx, test_idx in cv.split(self.file_dataset.X):
                X_train_fold = self.file_dataset.X[train_idx].copy()
                X_test_fold = self.file_dataset.X[test_idx].copy()
                y_train_fold = self.file_dataset.y[train_idx].copy()
                y_test_fold = self.file_dataset.y[test_idx].copy()

                if self.preprocessing_pipeline is not None:
                    # Use original raw feature indices
                    categorical_indices_raw = [
                        self.file_dataset.features.index(f)
                        for f in self.file_dataset.categorical_features
                    ]
                    continuous_indices_raw = [
                        self.file_dataset.features.index(f)
                        for f in self.file_dataset.numerical_features
                    ]

                    context = PreprocessingContext(
                        X_train=X_train_fold,
                        X_test=X_test_fold,
                        y_train=y_train_fold,
                        y_test=y_test_fold,
                        categorical_indices=categorical_indices_raw,
                        continuous_indices=continuous_indices_raw,
                    )

                    self.preprocessing_pipeline.fit(context)
                    transformed_context = self.preprocessing_pipeline.transform(context)

                    # Update instance variables with current fold's data
                    self.X_train = transformed_context.X_train
                    self.X_test = transformed_context.X_test
                    self.y_train = transformed_context.y_train
                    self.y_test = transformed_context.y_test
                    # Update feature indices to reflect the current preprocessed state
                    self.file_dataset.categorical_features_indices = (
                        transformed_context.categorical_indices
                    )
                    self.file_dataset.numerical_features_indices = (
                        transformed_context.continuous_indices
                    )
                    yield self.X_train, self.X_test, self.y_train, self.y_test
                else:
                    # Update instance variables with current fold's raw data
                    self.X_train = X_train_fold
                    self.X_test = X_test_fold
                    self.y_train = y_train_fold
                    self.y_test = y_test_fold
                    yield self.X_train, self.X_test, self.y_train, self.y_test

    def train_dataloader(
        self, batch_size: int, shuffle: bool, noise_lvl=0, **kwargs_dataloader
    ):
        def collate_fn(batch):
            X, y = zip(*batch)
            X = torch.stack(X)
            y = torch.stack(y)

            # Add Gaussian noise to train features
            noise = torch.randn_like(X[:, self.numerical_features_indices]) * noise_lvl
            X[:, self.numerical_features_indices] = (
                X[:, self.numerical_features_indices] + noise
            )
            return X, y

        return DataLoader(
            TensorDataset(
                torch.from_numpy(self.X_train), torch.from_numpy(self.y_train)
            ),
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn if noise_lvl else None,
            **kwargs_dataloader,
        )

    def test_dataloader(self, batch_size: int, shuffle: bool, **kwargs_dataloader):
        return DataLoader(
            TensorDataset(torch.from_numpy(self.X_test), torch.from_numpy(self.y_test)),
            batch_size=batch_size,
            shuffle=shuffle,
            **kwargs_dataloader,
        )
