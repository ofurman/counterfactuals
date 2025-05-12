import numpy as np
import pandas as pd
from sklearn.inspection import DecisionBoundaryDisplay
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import MinMaxScaler
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Tuple, Dict, Union, Optional, Set
import logging
import time
from sklearn.model_selection import train_test_split
from collections import defaultdict

from counterfactuals.datasets.base import AbstractDataset
from counterfactuals.discriminative_models.base import BaseDiscModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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
                                if self.noise_level > 0:
                                    x = x + np.random.normal(0, self.noise_level, size=x.shape)
                                
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

                    p_tensor = torch.tensor(p, dtype=torch.float32)
                    p_tensor = p_tensor.repeat(len(batch_cf), 1)

                    mask_tensor = torch.tensor(mask, dtype=torch.float32)
                    mask_tensor = mask_tensor.repeat(len(batch_cf), 1)
                    
                    batches.append((batch_x, batch_cond, batch_classes, p_tensor, mask_tensor))
        
        # Shuffle the order of batches if requested
        if shuffle:
            np.random.shuffle(batches)
        
        return batches


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
        prob_threshold: float = 0.0
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
        
        #  Set p_values (if None, use only p=2.0)
        #if p_values is None:
        #    p_values = [2.0]
        self.p_values = p_values

        # Set masks (if None, use only basic mask)
        #if masks is None:
        #    masks = np.ones(1, X.shape[1]) * 1e-2
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
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Apply transformations
        self.X_train, self.X_test, self.y_train, self.y_test = self.transform(
            self.X_train, self.X_test, self.y_train, self.y_test
        )
        
        # Separate points by class
        self.X_by_class = {}
        self.X_by_class_scaled = {}
        
        for cls in self.classes:
            self.X_by_class[cls] = X[y == cls]
            self.X_by_class_scaled[cls] = self.feature_transformer.transform(self.X_by_class[cls])
            self.logger.info(f"Class {cls}: {len(self.X_by_class[cls])} points")
        
        # Set feature properties
        self.numerical_features = list(range(X.shape[1]))
        self.categorical_features = []
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
                prob_threshold=self.prob_threshold
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
        
        class GroupedBatchDataLoader:
            """Custom DataLoader for grouped batches"""
            def __init__(self, batches, shuffle=True):
                self.batches = batches
                self.shuffle = shuffle
                
            def __iter__(self):
                # Shuffle the order of batches if requested
                indices = list(range(len(self.batches)))
                if self.shuffle:
                    np.random.shuffle(indices)
                
                # Group indices into batches
                for start_idx in range(0, len(indices), batch_size):
                    end_idx = start_idx + batch_size
                    yield self.batches[start_idx:end_idx]
                    
            def __len__(self):
                return len(self.batches)
        
        train_loader = GroupedBatchDataLoader(train_batches, shuffle=shuffle)
        test_loader = GroupedBatchDataLoader(test_batches, shuffle=False)
        
        return train_loader, test_loader


def __transform_batch_data(data: List, device: str):
    data = torch.stack(data)
    data = data.reshape(-1, data.shape[-1])
    return data.to(device).float()


def train_multiclass_counterfactual_flow_model(
    dataset: MulticlassCounterfactualWrapper,
    flow_model_class,
    hidden_features: int = 64,
    num_layers: int = 5,
    num_blocks_per_layer: int = 2,
    learning_rate: float = 1e-3,
    batch_size: Optional[int] = None,
    num_epochs: int = 1000,
    patience: int = 300,
    noise_level: float = 0.03,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_dir: str = "results",
    log_interval: int = 10,
    balanced: bool = True,
    load_from_save_dir: bool = False
):
    """
    Train a Conditional Normalizing Flow model for multiclass counterfactual generation.
    The model conditions on factual points to generate counterfactual points from different classes.
    
    Args:
        dataset: MulticlassCounterfactualWrapper instance
        flow_model_class: Class of the flow model to use (e.g., MaskedAutoregressiveFlow)
        hidden_features: Number of hidden features in flow model
        num_layers: Number of layers in flow model
        num_blocks_per_layer: Number of blocks per layer in flow model
        learning_rate: Learning rate for optimizer
        batch_size: Batch size for training (defaults to n_nearest * num_classes if None)
        num_epochs: Number of epochs to train
        patience: Number of epochs to wait for improvement before early stopping
        noise_level: Standard deviation of Gaussian noise to add during training
        device: Device to use for training ("cuda" or "cpu")
        save_dir: Directory to save results
        log_interval: Interval for logging detailed metrics
        balanced: Whether to ensure balanced representation of classes in each batch
    
    Returns:
        Trained flow model
    """
    start_time = time.time()
    logger.info(f"Starting multiclass counterfactual flow model training on device: {device}")
    logger.info(f"Model architecture: {num_layers} layers with {hidden_features} hidden features")
    logger.info(f"Training with balanced batches: {balanced}")
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Setup metrics logging directory
    metrics_dir = os.path.join(save_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Set dataset noise level
    dataset.noise_level = noise_level
    logger.info(f"Using noise level: {noise_level}")
    
    # Get data loaders
    logger.info("Preparing data loaders...")
    train_loader, test_loader = dataset.get_counterfactual_dataloaders(
        batch_size=batch_size,
        shuffle=True,
        balanced=balanced
    )
    logger.info(f"Created data loaders - Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    
    # Initialize model
    context_features = 2*dataset.X.shape[1] + 1  # Dimensionality of factual points
    features = dataset.X.shape[1]  # Dimensionality of counterfactual points
    num_classes = len(dataset.classes)
    
    # Add class one-hot encoding to context
    context_features += num_classes
    
    logger.info(f"Initializing model with {context_features} context features and {features} output features")
    model = flow_model_class(
        features=features,
        hidden_features=hidden_features,
        context_features=context_features,
        num_layers=num_layers,
        num_blocks_per_layer=num_blocks_per_layer,
        device=device
    ).to(device)
    logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    if load_from_save_dir:
        model_path = os.path.join(save_dir, "flow_model.pth")
        if os.path.exists(model_path):
            logger.info(f"Loading model from {model_path}")
            model.load_state_dict(torch.load(model_path)["model_state_dict"])
            return model
        else:
            raise FileNotFoundError(f"Model file not found at {model_path}")
    # Train model
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    logger.info(f"Using Adam optimizer with learning rate: {learning_rate}")
    
    # Tracking metrics
    best_test_loss = float('inf')
    patience_counter = 0
    train_losses = []
    test_losses = []
    
    # Training loop
    logger.info(f"Starting training for {num_epochs} epochs with patience {patience}")
    for epoch in (pbar := tqdm(range(num_epochs))):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        batch_times = []
        
        for batch_idx, batch_data in enumerate(train_loader):
            batch_start = time.time()
            
            # Unpack batch data
            x_batch, cond_batch, class_batch, p, mask = zip(*batch_data)

            # Transform batch data and move it to device
            x_batch = __transform_batch_data(x_batch, device)
            cond_batch = __transform_batch_data(cond_batch, device)
            class_batch = __transform_batch_data(class_batch, device)
            p = __transform_batch_data(p, device)
            mask = __transform_batch_data(mask, device)
            
            # Combine condition, class one-hot encoding, mask and p-norm
            combined_cond = torch.cat([cond_batch, class_batch, mask, p], dim=1)
            
            # Forward pass
            optimizer.zero_grad()
            log_prob = model(x_batch, combined_cond)
            loss = -log_prob.mean()
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            train_loss += loss.item()
            
            # Detailed logging at intervals
            if batch_idx % log_interval == 0 and epoch % log_interval == 0:
                logger.debug(f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, "
                           f"Loss: {loss.item():.4f}, Batch time: {batch_time:.4f}s")
        
        avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Evaluation phase
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_data in test_loader:
                # Unpack batch data
                x_batch, cond_batch, class_batch, p, mask = zip(*batch_data)

                # Transform batch data and move it to device
                x_batch = __transform_batch_data(x_batch, device)
                cond_batch = __transform_batch_data(cond_batch, device)
                class_batch = __transform_batch_data(class_batch, device)
                p = __transform_batch_data(p, device)
                mask = __transform_batch_data(mask, device)
                
                # Combine condition, class one-hot encoding, mask and p-norm
                combined_cond = torch.cat([cond_batch, class_batch, mask, p], dim=1)
                
                # Forward pass
                log_prob = model(x_batch, combined_cond)
                loss = -log_prob.mean()
                
                test_loss += loss.item()
        
        test_loss /= len(test_loader)
        test_losses.append(test_loss)
        
        epoch_time = time.time() - epoch_start_time
        
        # Progress bar update
        pbar.set_description(
            f"Epoch {epoch}, Train: {train_loss:.4f}, Test: {test_loss:.4f}, "
            f"Patience: {patience_counter}, Time: {epoch_time:.2f}s"
        )
        
        # Log detailed metrics periodically
        if epoch % log_interval == 0:
            logger.info(
                f"Epoch {epoch}/{num_epochs} - "
                f"Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}, "
                f"Epoch time: {epoch_time:.2f}s, Avg batch time: {avg_batch_time:.4f}s, "
                f"Patience counter: {patience_counter}"
            )
        
        # Early stopping and model saving
        if test_loss < best_test_loss - 1e-5:
            improvement = best_test_loss - test_loss
            best_test_loss = test_loss
            patience_counter = 0
            
            # Save model
            model_path = os.path.join(save_dir, "flow_model.pth")
            torch.save({
                'model_state_dict': model.state_dict(),
                'context_features': context_features,
                'features': features,
                'num_classes': num_classes
            }, model_path)
            logger.info(f"Epoch {epoch}: Test loss improved by {improvement:.6f}. Model saved to {model_path}")
        else:
            patience_counter += 1
            logger.debug(f"Epoch {epoch}: No improvement for {patience_counter} epochs")
            
        if patience_counter > patience:
            logger.info(f"Early stopping triggered after {epoch} epochs")
            break
        
        # Save loss curves periodically
        if epoch % (log_interval * 5) == 0 or epoch == num_epochs - 1:
            save_training_curves(train_losses, test_losses, metrics_dir, epoch)
    
    # Training complete - load best model
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    logger.info(f"Best test loss: {best_test_loss:.6f}")
    
    # Final loss curves
    save_training_curves(train_losses, test_losses, metrics_dir, "final")
    
    # Load best model
    checkpoint = torch.load(os.path.join(save_dir, "flow_model.pth"))
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info("Loaded best model weights")
    
    return model


def save_training_curves(train_losses, test_losses, save_dir, epoch_or_label):
    """Save training and test loss curves"""
    plt.figure(figsize=(12, 5))
    
    # Plot train and test losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(test_losses, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Negative Log Likelihood')
    plt.legend()
    plt.title('Training and Test Loss')
    
    # Plot only test loss for better visibility of improvements
    plt.subplot(1, 2, 2)
    plt.plot(test_losses, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Negative Log Likelihood')
    plt.title('Test Loss')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"loss_curves_{epoch_or_label}.png"))
    plt.close()


def generate_multiclass_counterfactuals(
    model,
    factual_points: np.ndarray,
    target_class: int,
    p_value: float,
    mask: np.ndarray,
    n_samples: int = 10,
    temperature: float = 0.8,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_classes: int = None
):
    """
    Generate counterfactual samples for given factual points targeting a specific class.
    
    Args:
        model: Trained flow model
        factual_points: Array of factual points to generate counterfactuals for
        target_class: Target class to generate counterfactuals for
        p_value: p-norm sparsity
        mask: Immutable features mask
        n_samples: Number of counterfactual samples to generate per factual point
        temperature: Temperature for sampling (higher = more diverse)
        device: Device to use for generation
        num_classes: Number of classes in the dataset
    
    Returns:
        Array of generated counterfactual samples of shape (factual_points.shape[0], n_samples, factual_points.shape[1])
    """
    model.eval()
    all_counterfactuals = np.zeros((factual_points.shape[0], n_samples, factual_points.shape[1]))

    p = torch.tensor([p_value], dtype=torch.float32).unsqueeze(0).to(device)
    mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        for factual_idx, factual in enumerate(factual_points):
            # Convert to tensor and add batch dimension
            factual_tensor = torch.tensor(factual, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Create a one-hot encoding for the target class
            class_one_hot = np.zeros(num_classes)
            class_one_hot[target_class] = 1
            class_tensor = torch.tensor(class_one_hot, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Combine factual point, class one-hot encoding, feature mask and p-norm
            context = torch.cat([factual_tensor, class_tensor, mask, p], dim=1)
            
            # Generate samples
            samples, log_probs = model.sample_and_log_prob(
                num_samples=n_samples + 5,
                context=context,
                temp=temperature
            )
            log_probs = log_probs.squeeze(0).cpu().numpy()
            samples = samples.squeeze(0)
            # Sort samples by log probability and take top n_samples
            top_indices = np.argsort(log_probs)[-n_samples:]
            samples = samples[top_indices]
            samples = samples.cpu().numpy()
            # remove samples with log prob less than median log prob
            # median_log_prob = np.nanquantile(log_probs, 0.25)
            # samples = samples[log_probs >= median_log_prob]
            # log_probs = log_probs[log_probs >= median_log_prob]
            # Convert to numpy
            # samples = samples.cpu().numpy()
            
            # Add to results
            all_counterfactuals[factual_idx] = samples
    
    return all_counterfactuals


def visualize_multiclass_counterfactual_generation(
    model,
    dataset,
    disc_model,
    masks,
    p_values,
    num_factual=5,
    num_samples=20,
    temperature=0.8,
    save_dir=None,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Visualize multiclass counterfactual generation results.
    
    Args:
        model: Trained flow model
        dataset: MulticlassCounterfactualWrapper instance
        disc_model: Discriminator model for visualizing decision boundaries
        num_factual: Number of factual points to generate counterfactuals for
        num_samples: Number of counterfactual samples to generate per factual point
        temperature: Temperature for sampling (higher = more diversity)
        save_dir: Directory to save visualizations
        device: Device to use for generation
    """
    # Only works for 2D data
    assert dataset.X.shape[1] == 2, "Only 2D data is supported for visualization"
    
    # Create samples directory
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        samples_dir = os.path.join(save_dir, "samples")
        os.makedirs(samples_dir, exist_ok=True)
    
    results = []

    # Function to plot decision boundaries
    def plot_decision_boundary(ax=None, alpha=0.3):
        """Plot decision boundaries from discriminator model"""
        # Check if we have a subplot or create a new one
        if ax is None:
            ax = plt.gca()
            
        # Create a grid of points
        x_min, x_max = plt.xlim()
        y_min, y_max = plt.ylim()
        
        # Create mesh grid
        xline = torch.linspace(x_min, x_max, 200)
        yline = torch.linspace(y_min, y_max, 200)
        xgrid, ygrid = torch.meshgrid(xline, yline)
        xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)

        y_hat = disc_model.predict(xyinput)
        y_hat = y_hat.reshape(200, 200)

        display = DecisionBoundaryDisplay(xx0=xgrid, xx1=ygrid, response=y_hat)
        ax = display.plot(plot_method="contour", ax=ax, alpha=0.3).ax_
        return ax
    

    for mask_idx, mask in enumerate(masks):
        logger.info(f"Generating counterfactuals for mask {mask}")
        for p_value in p_values:
            logger.info(f"Generating counterfactuals for p-norm {p_value}")
            # For each factual class
            for factual_class in dataset.factual_classes:
                logger.info(f"Generating counterfactuals for factual class: {factual_class}")
                
                # Get factual points for this class
                factual_scaled = dataset.X_by_class_scaled[factual_class]
                factual_original = dataset.X_by_class[factual_class]
                
                # Sample factual points
                if len(factual_scaled) <= num_factual:
                    factual_indices = np.arange(len(factual_scaled))
                else:
                    factual_indices = np.random.choice(
                        len(factual_scaled), 
                        size=num_factual, 
                        replace=False
                    )
                
                factual_points = factual_scaled[factual_indices]
                
                # For each target class
                for target_class in dataset.classes:
                    if target_class == factual_class:
                        continue  # Skip generating counterfactuals for the same class
                        
                    logger.info(f"Generating counterfactuals from class {factual_class} to class {target_class} with mask {mask} and p-value {p_value}")
                    
                    # Generate counterfactuals
                    generated_cfs = generate_multiclass_counterfactuals(
                        model=model,
                        factual_points=factual_points,
                        target_class=target_class,
                        n_samples=num_samples,
                        temperature=temperature,
                        device=device,
                        num_classes=len(dataset.classes),
                        mask=mask,
                        p_value=p_value
                    )
                    
                    # Convert to original scale for better interpretability
                    factual_orig = factual_points
                    
                    # Fix: Handle the 3D structure of generated_cfs by reshaping before inverse_transform
                    generated_cfs_orig = []
                    for cf_batch in generated_cfs:
                        # Check if we have a 3D array and reshape if needed
                        if cf_batch.ndim == 3:  # Shape is [batch_size, num_samples, features]
                            batch_size, n_samples, n_features = cf_batch.shape
                            reshaped_batch = cf_batch.reshape(-1, n_features)  # Flatten to 2D
                            transformed = dataset.feature_transformer.inverse_transform(reshaped_batch)
                            # Reshape back to original 3D shape
                            transformed = transformed.reshape(batch_size, n_samples, n_features)
                            generated_cfs_orig.append(transformed)
                        else:  # Regular 2D case
                            generated_cfs_orig.append(dataset.feature_transformer.inverse_transform(cf_batch))
                    
                    # Store results for this class pair
                    results.append({
                        'factual_class': factual_class,
                        'target_class': target_class,
                        'factual_indices': factual_indices,
                        'factual_points': factual_points,
                        'factual_orig': factual_orig,
                        'generated_cfs': generated_cfs,
                        'generated_cfs_orig': generated_cfs_orig
                    })
                    
                    # Plot settings
                    colors = plt.cm.tab10(np.linspace(0, 1, num_factual))
                    
                    # Create overview plot for this class pair
                    fig, ax = plt.subplots(figsize=(14, 10))
                    
                    # Plot all original data points with low opacity
                    plt.scatter(
                        dataset.X[:, 0],
                        dataset.X[:, 1],
                        c=dataset.y,
                        cmap=plt.cm.tab10,
                        alpha=0.2,
                        s=30
                    )
                    
                    # Plot decision boundaries before adding other elements
                    plot_decision_boundary(ax=ax, alpha=0.6)
                    
                    # Add a legend for the original classes
                    for cls in dataset.classes:
                        plt.scatter([], [], color=plt.cm.tab10(cls % 10), label=f'Class {cls}')
                    
                    # For each factual point
                    for i, (f_idx, factual, cf_samples) in enumerate(zip(factual_indices, factual_orig, generated_cfs_orig)):
                        # Plot factual point
                        plt.scatter(
                            factual[0],
                            factual[1],
                            color=colors[i],
                            s=150,
                            marker='*',
                            edgecolor='black',
                            label=f'Factual {i+1}' if i < 5 else None  # Limit legend entries
                        )
                        
                        # Plot generated counterfactuals - Fix to handle potential 3D array
                        if cf_samples.ndim == 3:
                            # If we have a 3D array, take the first dimension
                            cf_to_plot = cf_samples[0]
                        else:
                            cf_to_plot = cf_samples
                            
                        plt.scatter(
                            cf_to_plot[:, 0],
                            cf_to_plot[:, 1],
                            color=colors[i],
                            alpha=0.6,
                            marker='x',
                            s=50,
                            label=f'Generated CFs for Factual {i+1}' if i < 5 else None
                        )
                        
                        # Draw lines to a few counterfactuals
                        for j in range(min(5, len(cf_to_plot))):
                            plt.plot(
                                [factual[0], cf_to_plot[j, 0]],
                                [factual[1], cf_to_plot[j, 1]],
                                color=colors[i],
                                linestyle='--',
                                alpha=0.3
                            )
                        
                        # Create individual plot for this factual point
                        if save_dir:
                            fig_ind, ax_ind = plt.subplots(figsize=(10, 8))
                            
                            # Plot original data with low opacity
                            plt.scatter(
                                dataset.X[:, 0],
                                dataset.X[:, 1],
                                c=dataset.y,
                                cmap=plt.cm.tab10,
                                alpha=0.2,
                                s=30
                            )
                            
                            # Plot decision boundaries
                            plot_decision_boundary(ax=ax_ind, alpha=0.6)
                            
                            # Plot factual point
                            plt.scatter(
                                factual[0],
                                factual[1],
                                color=colors[i],
                                s=200,
                                marker='*',
                                edgecolor='black',
                                label=f'Factual Point {i+1}'
                            )
                            
                            # Plot generated counterfactuals - Fix to handle potential 3D array
                            if cf_samples.ndim == 3:
                                # If we have a 3D array, take the first dimension
                                cf_to_plot = cf_samples[0]
                            else:
                                cf_to_plot = cf_samples
                                
                            plt.scatter(
                                cf_to_plot[:, 0],
                                cf_to_plot[:, 1],
                                color=colors[i],
                                alpha=0.7,
                                marker='x',
                                s=80,
                                label='Generated Counterfactuals'
                            )
                            
                            # Draw lines to all counterfactuals
                            for j in range(len(cf_to_plot)):
                                plt.plot(
                                    [factual[0], cf_to_plot[j, 0]],
                                    [factual[1], cf_to_plot[j, 1]],
                                    color=colors[i],
                                    linestyle='--',
                                    alpha=0.3
                                )
                            
                            plt.title(f"Counterfactuals Generated for Factual Point {i+1} (Class {factual_class} → Class {target_class})")
                            plt.legend()
                            plt.grid(True, alpha=0.3)
                            
                            # Create subdirectory for each class pair if needed
                            subdir = os.path.join(samples_dir, f"class_{factual_class}_to_{target_class}")
                            os.makedirs(subdir, exist_ok=True)
                            
                            plt.savefig(os.path.join(subdir, f"factual_{i+1}_counterfactuals.png"))
                            plt.close()
                    
                    # Finish and save overview plot
                    plt.title(f"Overview of Generated Counterfactuals (Class {factual_class} → Class {target_class}) with mask {mask} and p-value {p_value}")
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    
                    if save_dir:
                        plt.savefig(os.path.join(save_dir, f"counterfactual_overview_class_{factual_class}_to_{target_class}_mask_{mask}_p_value_{p_value}.png"))
                        plt.close()

    return results 


def visualize_single_factual_counterfactuals(
    model,
    dataset,
    disc_model,
    factual_point: np.ndarray,
    factual_class: int,
    target_class: int,
    mask: np.ndarray,
    p_value: float,
    num_samples: int = 20,
    temperature: float = 0.8,
    save_path: Optional[str] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Visualize counterfactuals generated for a single factual point.
    
    Args:
        model: Trained flow model
        dataset: MulticlassCounterfactualWrapper instance
        disc_model: Discriminator model for visualizing decision boundaries
        factual_point: The factual point to generate counterfactuals for
        factual_class: Class of the factual point
        target_class: Target class to generate counterfactuals for
        mask: Feature mask for immutable features
        p_value: p-norm value for distance calculation
        num_samples: Number of counterfactual samples to generate
        temperature: Temperature for sampling (higher = more diverse)
        save_path: Path to save the visualization (if None, plot is shown)
        device: Device to use for generation
    """
    # Only works for 2D data
    assert dataset.X.shape[1] == 2, "Only 2D data is supported for visualization"
    
    # Generate counterfactuals
    generated_cfs = generate_multiclass_counterfactuals(
        model=model,
        factual_points=factual_point[np.newaxis, :],  # Add batch dimension
        target_class=target_class,
        n_samples=num_samples,
        temperature=temperature,
        device=device,
        num_classes=len(dataset.classes),
        mask=mask,
        p_value=p_value
    )
    
    # Convert to original scale
    factual_orig = dataset.feature_transformer.inverse_transform(factual_point[np.newaxis, :])[0]
    generated_cfs_orig = dataset.feature_transformer.inverse_transform(generated_cfs[0])
    
    # Create the plot
    plt.figure(figsize=(6, 5))
    # fontsize
    plt.rcParams.update({'font.size': 14})

    ax = plt.gca()
    plt.xlim(0, 0.8)
    plt.ylim(0.2, 1)
    
    # Plot decision boundaries
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    
    # Create mesh grid
    xline = torch.linspace(x_min, x_max, 200)
    yline = torch.linspace(y_min, y_max, 200)
    xgrid, ygrid = torch.meshgrid(xline, yline)
    xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)
    
    y_hat = disc_model.predict(xyinput)
    y_hat = y_hat.reshape(200, 200)
    
    display = DecisionBoundaryDisplay(xx0=xgrid, xx1=ygrid, response=y_hat)
    display.plot(plot_method="contour", ax=ax, alpha=0.3)
    # add legend green line for decision boundary
    ax.plot([x_min, x_min], [x_min, x_min], color='green', label='Decision Boundary')

    # Plot training data points with slightly higher opacity
    ax.scatter(
        dataset.X_test[dataset.y_test == factual_class, 0],
        dataset.X_test[dataset.y_test == factual_class, 1],
        c='red',
        alpha=0.10,  # Higher alpha for training points
        s=40,
        label='Original Class'
    )

    ax.scatter(
        dataset.X_test[dataset.y_test == target_class, 0],
        dataset.X_test[dataset.y_test == target_class, 1],
        c='blue',
        alpha=0.10,  # Higher alpha for training points
        s=40,
        label='Target Class'
    )
    
    # Plot factual point
    ax.scatter(
        factual_orig[0],
        factual_orig[1],
        color='red',
        s=250,
        marker='*',
        edgecolor='black',
        label=f'Factual Points'
    )
    
    # Plot generated counterfactuals
    ax.scatter(
        generated_cfs_orig[:, 0],
        generated_cfs_orig[:, 1],
        color='blue',
        alpha=0.8,
        marker='x',
        s=100,
        label=f'Counterfactual Points'
    )
    
    # Draw lines to all counterfactuals
    for j in range(len(generated_cfs_orig)):
        ax.plot(
            [factual_orig[0], generated_cfs_orig[j, 0]],
            [factual_orig[1], generated_cfs_orig[j, 1]],
            color='gray',
            linestyle='--',
            alpha=0.3
        )
    
    # Add title and labels
    # plt.title(
    #     f"Counterfactuals for Factual Point (Class {factual_class} → Class {target_class}). p-value: {p_value}\n"
    #     f"Mask: {mask}"
    # )
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    # set legend to lower right
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Save or show the plot
    if save_path:
        # tight layout
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show() 