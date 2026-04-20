import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from typing import List, Optional, Dict
import logging

from counterfactuals.datasets.base import AbstractDataset
from counterfactuals.discriminative_models.base import BaseDiscModel

logger = logging.getLogger('counterfactual')


class MulticlassCounterfactualDataset(Dataset):
    """
    PyTorch Dataset for multiclass counterfactual training with normalizing flows.
    This class handles any multiclass dataset, conditioning on factual points
    to generate counterfactual points from different classes.

    The dataset organizes samples by factual points, allowing for efficient
    batching where all samples in a batch have the same factual conditioning point
    but different counterfactual targets from various classes.
    """
    def __init__(
        self,
        X_factual: np.ndarray,
        X_counterfactual_dict: Dict[int, np.ndarray],
        p_values: List[float],
        masks: np.ndarray,
        classes: List[int] = [0, 1],
        n_nearest: int = 5,
        noise_level: float = 0.05,
        classifier: BaseDiscModel = None,
        prob_threshold: float = 0.0,
        *,
        numerical_features,
        categorical_features
    ):
        """
        Args:
            X_factual: Array of factual points (NxD)
            X_counterfactual_dict: Dictionary mapping class labels to arrays of counterfactual points
            p_values: List of norms used for calculating distance
            masks: Array of immutable features masks
            n_nearest: Number of nearest counterfactual points to use for each factual point per class
            noise_level: Standard deviation of Gaussian noise to add to counterfactual points
            classifier: Classifier model
            prob_threshold: Probability threshold for classifier
        """
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.X_factual = X_factual.astype(np.float32)
        self.X_counterfactual_dict = {k: v.astype(np.float32) for k, v in X_counterfactual_dict.items()}
        self.classes = classes
        self.n_nearest = n_nearest
        self.noise_level = noise_level
        self.counterfactual_classes = list(X_counterfactual_dict.keys())
        self.classifier = classifier
        self.prob_threshold = prob_threshold

        # Compute distance matrices between factual and counterfactual points for each class
        self.dist_matrices = {}
        self.nearest_indices = {}
        self.factual_to_cf_indices = {}

        self.p_values = p_values
        self.masks = masks

        for mask_idx, mask in enumerate(self.masks):
            self.factual_to_cf_indices[mask_idx] = {}
            mask = mask[np.newaxis, np.newaxis, :]
            for p in self.p_values:
                self.factual_to_cf_indices[mask_idx][p] = {}

                for cf_class, X_counterfactual in self.X_counterfactual_dict.items():
                    # Compute distance matrix with p-norm and feature mask
                    dist_matrix = np.abs(X_factual[:, np.newaxis, :] - X_counterfactual[np.newaxis, :, :]) ** p
                    dist_matrix = np.sum(dist_matrix * mask, axis=-1) ** (1 / p)

                    # If a classifier is provided with a threshold, filter the distance matrix
                    if self.classifier is not None and self.prob_threshold > 0:
                        posterior_probs = self.classifier.predict_proba(X_counterfactual)[:, cf_class]
                        below_threshold_mask = posterior_probs < self.prob_threshold
                        # Set distances to infinity for points with probability below threshold
                        logger.info(f"Setting {below_threshold_mask.sum()} distances to infinity out of {len(below_threshold_mask)} for class {cf_class}")
                        dist_matrix[:, below_threshold_mask] = np.inf


                    # For each factual point, find the n_nearest counterfactual points
                    nearest_indices = np.argsort(dist_matrix, axis=1)[:, :n_nearest]

                    # Create a mapping from factual index to list of nearest counterfactual indices
                    factual_to_cf_indices = {}
                    for f_idx in range(len(X_factual)):
                        factual_to_cf_indices[f_idx] = nearest_indices[f_idx]

                    self.factual_to_cf_indices[mask_idx][p][cf_class] = factual_to_cf_indices

        # Create an index mapping for the dataset
        # Each entry is (mask_idx, p, f_idx, cf_class, cf_idx) where:
        # - mask_idx is the feature mask index
        # - p is the p-norm
        # - f_idx is the factual point index
        # - cf_class is the counterfactual class
        # - cf_idx is the counterfactual point index
        self.index_mapping = []
        for mask_idx, mask in enumerate(self.masks):
            for p in self.p_values:
                for f_idx in range(len(X_factual)):
                    for cf_class in self.counterfactual_classes:
                        for cf_idx in self.factual_to_cf_indices[mask_idx][p][cf_class][f_idx]:
                            self.index_mapping.append((mask_idx, p, f_idx, cf_class, cf_idx))

    def __len__(self):
        return len(self.index_mapping)

    def __getitem__(self, idx):
        # Get the mask, p, factual, counterfactual class, and counterfactual indices from the mapping
        mask_idx, p, f_idx, cf_class, cf_idx = self.index_mapping[idx]

        # Get the factual point (used as condition)
        cond = self.X_factual[f_idx]

        # Get the counterfactual point (target to generate)
        x = self.X_counterfactual_dict[cf_class][cf_idx].copy()

        # Add small Gaussian noise to counterfactual point (target)
        if self.noise_level > 0:
            x = x + np.random.normal(0, self.noise_level, size=x.shape)

        # Create a one-hot encoding for the counterfactual class
        class_one_hot = np.zeros(len(self.classes))
        class_idx = self.classes.index(cf_class)
        class_one_hot[class_idx] = 1

        # Get feature mask
        mask = self.masks[mask_idx]

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(cond, dtype=torch.float32),
            torch.tensor(class_one_hot, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32),
            torch.tensor([p], dtype=torch.float32)
        )

    def get_grouped_batches(self, batch_size=None, shuffle=True, balanced=True):
        """
        Create batches where all samples in a batch share the same factual point.
        Each batch contains a factual point and its closest counterfactual points from different classes.

        Args:
            batch_size: Maximum batch size (defaults to n_nearest * num_classes if None)
            shuffle: Whether to shuffle the order of batches
            balanced: Whether to ensure balanced representation of classes in each batch

        Returns:
            List of batches, where each batch is a tuple of (counterfactual_batch, factual_batch, class_batch, p_batch, mask_batch)
        """
        batches = []

        # Default batch_size to n_nearest * num_classes if not specified
        if batch_size is None:
            batch_size = self.n_nearest * len(self.counterfactual_classes)

        # For each factual point
        for mask_idx, mask in enumerate(self.masks):
            for p in self.p_values:
                for f_idx in range(len(self.X_factual)):
                    batch_cf = []
                    batch_classes = []

                    # Get the factual point as conditioning
                    cond = self.X_factual[f_idx]

                    if balanced:
                        # Ensure balanced representation of classes in each batch
                        points_per_class = min(self.n_nearest, batch_size // len(self.counterfactual_classes))

                        for cf_class in self.counterfactual_classes:
                            cf_indices = self.factual_to_cf_indices[mask_idx][p][cf_class][f_idx]

                            # Shuffle counterfactual indices if requested
                            if shuffle:
                                np.random.shuffle(cf_indices)

                            # For each counterfactual point in this batch
                            for cf_idx in cf_indices[:points_per_class]:
                                # Get the counterfactual point
                                x = self.X_counterfactual_dict[cf_class][cf_idx].copy()

                                # Add noise
                                x[self.numerical_features] = (
                                        x[self.numerical_features] +
                                        np.random.normal(0, 1,
                                                         size=x[self.numerical_features].shape)*self.noise_level
                                )
                                x[self.categorical_features] = (
                                        x[self.categorical_features] +
                                        np.random.normal(0, 0.08, size=x[self.categorical_features].shape)
                                )

                                batch_cf.append(torch.tensor(x, dtype=torch.float32))

                                # Create a one-hot encoding for the counterfactual class
                                class_one_hot = np.zeros(len(self.classes))
                                class_idx = self.classes.index(cf_class)
                                class_one_hot[class_idx] = 1
                                batch_classes.append(torch.tensor(class_one_hot, dtype=torch.float32))
                    else:
                        # Not balanced - just take the closest points regardless of class
                        all_cf_indices = []
                        for cf_class in self.counterfactual_classes:
                            cf_indices = self.factual_to_cf_indices[mask_idx][p][cf_class][f_idx]
                            for cf_idx in cf_indices:
                                all_cf_indices.append((cf_class, cf_idx))

                        # Shuffle all counterfactual indices if requested
                        if shuffle:
                            np.random.shuffle(all_cf_indices)

                        # For each counterfactual point in this batch
                        for cf_class, cf_idx in all_cf_indices[:batch_size]:
                            # Get the counterfactual point
                            x = self.X_counterfactual_dict[cf_class][cf_idx].copy()

                            # Add noise
                            if self.noise_level > 0:
                                x = x + np.random.normal(0, self.noise_level, size=x.shape)

                            batch_cf.append(torch.tensor(x, dtype=torch.float32))

                            # Create a one-hot encoding for the counterfactual class
                            class_one_hot = np.zeros(len(self.counterfactual_classes))
                            class_idx = self.counterfactual_classes.index(cf_class)
                            class_one_hot[class_idx] = 1
                            batch_classes.append(torch.tensor(class_one_hot, dtype=torch.float32))

                    # Skip if no counterfactual points were added
                    if not batch_cf:
                        continue

                    # Create batch tensors
                    batch_x = torch.stack(batch_cf)
                    batch_classes = torch.stack(batch_classes)

                    # Create a batch of identical factual points (one for each counterfactual)
                    # Convert numpy array to tensor first
                    cond_tensor = torch.tensor(cond, dtype=torch.float32)
                    # Then create a batch by repeating it
                    batch_cond = cond_tensor.repeat(len(batch_cf), 1)
                    batch_cond[:, self.numerical_features] = (
                            batch_cond[:, self.numerical_features] +
                            torch.randn(size=batch_cond[:, self.numerical_features].shape)*self.noise_level/10
                    )

                    p_tensor = torch.tensor(p, dtype=torch.float32)
                    p_tensor = p_tensor.repeat(len(batch_cf), 1)

                    mask_ohe = np.zeros(len(self.masks))
                    mask_ohe[mask_idx] = 1.
                    mask_tensor = torch.tensor(mask_ohe, dtype=torch.float32)
                    mask_tensor = mask_tensor.repeat(len(batch_cf), 1)

                    batches.append((batch_x, batch_cond, batch_classes, p_tensor, mask_tensor))

        # Shuffle the order of batches if requested
        if shuffle:
            np.random.shuffle(batches)

        return batches


class GroupedBatchDataLoader:
    """Custom DataLoader for grouped batches"""
    def __init__(self, batches, batch_size, shuffle=True):
        self.batches = batches
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        # Shuffle the order of batches if requested
        indices = list(range(len(self.batches)))
        if self.shuffle:
            np.random.shuffle(indices)

        # Group indices into batches
        for start_idx in range(0, len(indices), self.batch_size):
            end_idx = start_idx + self.batch_size
            yield self.batches[start_idx:end_idx]

    def __len__(self):
        return len(self.batches)


class MulticlassCounterfactualWrapper(AbstractDataset):
    """
    Wrapper for generic dataset that supports multiclass counterfactual generation
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        factual_classes: Optional[List[int]] = None,
        p_values: List[float] = None,
        masks: np.ndarray = None,
        n_nearest: int = 5,
        noise_level: float = 0.05,
        test_size: float = 0.2,
        random_state: int = 42,
        log_level: str = 'INFO',
        classifier: BaseDiscModel = None,
        prob_threshold: float = 0.0,
        *,
        numerical_pos
    ):
        """
        Initialize the multiclass counterfactual wrapper

        Args:
            X: Feature matrix
            y: Labels (multiclass)
            factual_classes: List of classes to use as factual (if None, use all classes)
            p_values: List of norms used for calculating distance
            masks: Array of immutable features masks
            n_nearest: Number of nearest counterfactual points to consider
            noise_level: Standard deviation of Gaussian noise to add to counterfactual points
            test_size: Fraction of data to use for testing
            random_state: Random seed
            log_level: Logging level
            classifier: Classifier model
            prob_threshold: Probability threshold for classifier
        """
        # Configure logging
        numeric_level = getattr(logging, log_level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {log_level}")
        logging.basicConfig(
            level=numeric_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('counterfactual')

        # Store dataset
        self.X = X
        self.y = y
        self.classes = np.unique(y)
        self.n_nearest = n_nearest
        self.noise_level = noise_level
        self.classifier = classifier
        self.prob_threshold = prob_threshold

        # Get unique classes
        self.classes = np.unique(y)
        self.logger.info(f"Found {len(self.classes)} classes: {self.classes}")

        self.p_values = p_values
        self.masks = masks

        # Set factual classes (if None, use all classes)
        if factual_classes is None:
            self.factual_classes = self.classes
        else:
            self.factual_classes = np.array(factual_classes)
            # Validate that all specified classes exist in the dataset
            for cls in self.factual_classes:
                if cls not in self.classes:
                    raise ValueError(f"Class {cls} not found in dataset")

        self.logger.info(f"Using {len(self.factual_classes)} factual classes: {self.factual_classes}")

        # Separate points by class
        self.X_by_class = {}
        self.X_by_class_scaled = {}

        for cls in self.classes:
            self.X_by_class[cls] = X[y == cls]
            self.X_by_class_scaled[cls] = self.X_by_class[cls]
            self.logger.info(f"Class {cls}: {len(self.X_by_class[cls])} points")

        # Set feature properties
        self.numerical_features = list(range(numerical_pos))
        self.categorical_features = list(range(numerical_pos, X.shape[1]))
        self.actionable_features = list(range(X.shape[1]))
        self.categorical_columns = []

        self.logger.info(f"Preprocessing complete. Dataset ready with {X.shape[1]} features.")

    def preprocess(self, X_train, X_test, y_train, y_test):
        """
        Dummy method to satisfy abstract class
        """
        return X_train, X_test, y_train, y_test

    def transform(self, X_train, X_test, y_train, y_test):
        """
        Scale the features to [0, 1] range
        """
        self.feature_transformer = MinMaxScaler()
        X_train = self.feature_transformer.fit_transform(X_train)
        X_test = self.feature_transformer.transform(X_test)

        # Convert to correct types
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
        y_train = y_train.astype(np.int64)
        y_test = y_test.astype(np.int64)

        return X_train, X_test, y_train, y_test

    def get_counterfactual_dataloaders(self, batch_size=64, shuffle=True, balanced=True):
        """
        Returns DataLoaders for multiclass counterfactual training

        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle data
            balanced: Whether to ensure balanced representation of classes in each batch

        Returns:
            train_loader, test_loader
        """
        local_batch_size = self.n_nearest * len(self.classes)

        # Create datasets for each factual class
        datasets = []

        for factual_class in self.factual_classes:
            # Get factual points for this class
            X_factual = self.X_by_class_scaled[factual_class]

            # Create dictionary of counterfactual points for other classes
            X_counterfactual_dict = {}
            for cf_class in self.classes:
                if cf_class != factual_class:
                    X_counterfactual_dict[cf_class] = self.X_by_class_scaled[cf_class]

            # Create dataset for this factual class
            dataset = MulticlassCounterfactualDataset(
                X_factual=X_factual,
                X_counterfactual_dict=X_counterfactual_dict,
                p_values=self.p_values,
                masks=self.masks,
                n_nearest=self.n_nearest,
                noise_level=self.noise_level,
                classes=list(self.classes),
                classifier=self.classifier,
                prob_threshold=self.prob_threshold,
                numerical_features=self.numerical_features,
                categorical_features=self.categorical_features
            )

            datasets.append(dataset)

        # Get all batches from all datasets
        all_batches = []
        for dataset in datasets:
            batches = dataset.get_grouped_batches(batch_size=local_batch_size, shuffle=shuffle, balanced=balanced)
            all_batches.extend(batches)

        # Split into train and test
        train_size = int(0.8 * len(all_batches))
        indices = np.arange(len(all_batches))
        np.random.shuffle(indices)

        train_indices = indices[:train_size]
        test_indices = indices[train_size:]

        train_batches = [all_batches[i] for i in train_indices]
        test_batches = [all_batches[i] for i in test_indices]

        train_loader = GroupedBatchDataLoader(train_batches, batch_size=batch_size, shuffle=shuffle)
        test_loader = GroupedBatchDataLoader(test_batches, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader
