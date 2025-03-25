import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import MinMaxScaler
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Tuple, Dict, Union, Optional
import logging
import time
from sklearn.model_selection import train_test_split

from counterfactuals.datasets.base import AbstractDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('counterfactual')


class GenericCounterfactualDataset(Dataset):
    """
    PyTorch Dataset for counterfactual training with normalizing flows.
    This class handles any binary dataset, conditioning on factual points
    to generate counterfactual points.
    
    The dataset organizes samples by factual points, allowing for efficient
    batching where all samples in a batch have the same factual conditioning point
    but different counterfactual targets.
    """
    def __init__(
        self, 
        X_factual: np.ndarray, 
        X_counterfactual: np.ndarray, 
        n_nearest: int = 5,
        noise_level: float = 0.05,
        distance_metric: str = 'euclidean'
    ):
        """
        Args:
            X_factual: Array of factual points (NxD)
            X_counterfactual: Array of counterfactual points (MxD)
            n_nearest: Number of nearest counterfactual points to use for each factual point
            noise_level: Standard deviation of Gaussian noise to add to counterfactual points
            distance_metric: Distance metric to use ('euclidean', 'manhattan', etc.)
        """
        self.X_factual = X_factual.astype(np.float32)
        self.X_counterfactual = X_counterfactual.astype(np.float32)
        self.n_nearest = n_nearest
        self.noise_level = noise_level
        
        # Compute distance matrix between factual and counterfactual points
        if distance_metric == 'euclidean':
            self.dist_matrix = euclidean_distances(X_factual, X_counterfactual)
        else:
            # Default to euclidean distance
            self.dist_matrix = euclidean_distances(X_factual, X_counterfactual)
        
        # For each factual point, find the n_nearest counterfactual points
        self.nearest_indices = np.argsort(self.dist_matrix, axis=1)[:, :n_nearest]
        
        # Create a mapping from factual index to list of nearest counterfactual indices
        self.factual_to_cf_indices = {}
        for f_idx in range(len(X_factual)):
            self.factual_to_cf_indices[f_idx] = self.nearest_indices[f_idx]
        
        # Create an index mapping for the dataset
        # Each entry is (f_idx, cf_idx) where:
        # - f_idx is the factual point index
        # - cf_idx is the counterfactual point index
        self.index_mapping = []
        for f_idx, cf_indices in self.factual_to_cf_indices.items():
            for cf_idx in cf_indices:
                self.index_mapping.append((f_idx, cf_idx))
    
    def __len__(self):
        return len(self.index_mapping)
    
    def __getitem__(self, idx):
        # Get the factual and counterfactual indices from the mapping
        f_idx, cf_idx = self.index_mapping[idx]
        
        # Get the factual point (used as condition)
        cond = self.X_factual[f_idx]
        
        # Get the counterfactual point (target to generate)
        x = self.X_counterfactual[cf_idx].copy()
        
        # Add small Gaussian noise to counterfactual point (target)
        if self.noise_level > 0:
            x = x + np.random.normal(0, self.noise_level, size=x.shape)
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(cond, dtype=torch.float32)
    
    def get_grouped_batches(self, batch_size=None, shuffle=True):
        """
        Create batches where all samples in a batch share the same factual point.
        Each batch contains a factual point and its closest counterfactual points.
        
        Args:
            batch_size: Maximum batch size (defaults to n_nearest if None)
            shuffle: Whether to shuffle the order of batches
            
        Returns:
            List of batches, where each batch is a tuple of (counterfactual_batch, factual_batch)
        """
        batches = []
        
        # Default batch_size to n_nearest if not specified
        if batch_size is None:
            batch_size = self.n_nearest
        
        # For each factual point
        for f_idx, cf_indices in self.factual_to_cf_indices.items():
            batch_cf = []
            
            # Shuffle counterfactual indices if requested
            if shuffle:
                np.random.shuffle(cf_indices)
            
            # Get the factual point as conditioning
            cond = self.X_factual[f_idx]
            
            # For each counterfactual point in this batch
            for cf_idx in cf_indices[:batch_size]:  # Limit to batch_size
                # Get the counterfactual point
                x = self.X_counterfactual[cf_idx].copy()
                
                # Add noise
                if self.noise_level > 0:
                    x = x + np.random.normal(0, self.noise_level, size=x.shape)
                
                batch_cf.append(torch.tensor(x, dtype=torch.float32))
            
            # Create batch tensors
            batch_x = torch.stack(batch_cf)
            
            # Create a batch of identical factual points (one for each counterfactual)
            # Convert numpy array to tensor first
            cond_tensor = torch.tensor(cond, dtype=torch.float32)
            # Then create a batch by repeating it
            batch_cond = cond_tensor.repeat(len(batch_cf), 1)
            
            batches.append((batch_x, batch_cond))
        
        # Shuffle the order of batches if requested
        if shuffle:
            np.random.shuffle(batches)
        
        return batches


class CounterfactualWrapper(AbstractDataset):
    """
    Wrapper for generic dataset that supports counterfactual generation
    """
    
    def __init__(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        factual_class: int = 0,
        counterfactual_class: int = 1,
        n_nearest: int = 5,
        noise_level: float = 0.05,
        test_size: float = 0.2,
        random_state: int = 42,
        distance_metric: str = 'euclidean',
        log_level: str = 'INFO',
        bidirectional: bool = False
    ):
        """
        Initialize the counterfactual wrapper
        
        Args:
            X: Feature matrix
            y: Labels (binary)
            factual_class: Class to use as factual
            counterfactual_class: Class to use as counterfactual
            n_nearest: Number of nearest counterfactual points to consider
            noise_level: Standard deviation of Gaussian noise to add to counterfactual points
            test_size: Fraction of data to use for testing
            random_state: Random seed
            distance_metric: Distance metric to use for nearest neighbors
            log_level: Logging level
            bidirectional: If True, enables bidirectional counterfactual generation
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
        self.factual_class = factual_class
        self.counterfactual_class = counterfactual_class
        self.n_nearest = n_nearest
        self.noise_level = noise_level
        self.distance_metric = distance_metric
        self.bidirectional = bidirectional
        
        self.logger.info(f"Initializing CounterfactualWrapper with {len(X)} samples")
        self.logger.info(f"Class distribution: Class {factual_class}: {np.sum(y == factual_class)}, Class {counterfactual_class}: {np.sum(y == counterfactual_class)}")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Apply transformations
        self.X_train, self.X_test, self.y_train, self.y_test = self.transform(
            self.X_train, self.X_test, self.y_train, self.y_test
        )
        
        # Separate factual and counterfactual points
        self.X_factual = X[y == factual_class]
        self.X_counterfactual = X[y == counterfactual_class]
        self.logger.info(f"Factual points: {len(self.X_factual)}, Counterfactual points: {len(self.X_counterfactual)}")
        
        # Scale factual and counterfactual points
        self.X_factual_scaled = self.feature_transformer.transform(self.X_factual)
        self.X_counterfactual_scaled = self.feature_transformer.transform(self.X_counterfactual)
        
        # If bidirectional, also prepare reverse datasets
        if self.bidirectional:
            self.X_factual_rev = X[y == counterfactual_class]
            self.X_counterfactual_rev = X[y == factual_class]
            self.X_factual_scaled_rev = self.feature_transformer.transform(self.X_factual_rev)
            self.X_counterfactual_scaled_rev = self.feature_transformer.transform(self.X_counterfactual_rev)
        
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
    
    def get_counterfactual_dataloaders(self, batch_size=None, shuffle=True, direction='forward'):
        """
        Returns DataLoaders for counterfactual training
        
        Args:
            batch_size: Batch size (if None, uses self.n_nearest)
            shuffle: Whether to shuffle data
            direction: 'forward' for factual→counterfactual, 'reverse' for counterfactual→factual, 
                       'both' for both directions (only works when bidirectional=True)
        
        Returns:
            train_loader, test_loader
        """
        if batch_size is None:
            batch_size = self.n_nearest
        
        # Validate direction parameter
        if direction not in ['forward', 'reverse', 'both']:
            raise ValueError("direction must be one of: 'forward', 'reverse', 'both'")
        
        # Check if bidirectional mode is enabled when using 'reverse' or 'both'
        if (direction in ['reverse', 'both']) and not self.bidirectional:
            raise ValueError(f"Cannot use direction='{direction}' when bidirectional=False")
            
        # Create forward dataset (factual→counterfactual)
        if direction in ['forward', 'both']:
            forward_dataset = GenericCounterfactualDataset(
                X_factual=self.X_factual_scaled,
                X_counterfactual=self.X_counterfactual_scaled,
                n_nearest=self.n_nearest,
                noise_level=self.noise_level,
                distance_metric=self.distance_metric
            )
        
        # Create reverse dataset (counterfactual→factual)
        if direction in ['reverse', 'both']:
            reverse_dataset = GenericCounterfactualDataset(
                X_factual=self.X_factual_scaled_rev,  # Using counterfactual class as factual
                X_counterfactual=self.X_counterfactual_scaled_rev,  # Using factual class as counterfactual
                n_nearest=self.n_nearest,
                noise_level=self.noise_level,
                distance_metric=self.distance_metric
            )
            
        # Handle 'both' direction by combining datasets
        if direction == 'both':
            combined_batches = []
            forward_batches = forward_dataset.get_grouped_batches(batch_size=batch_size, shuffle=shuffle)
            reverse_batches = reverse_dataset.get_grouped_batches(batch_size=batch_size, shuffle=shuffle)
            
            # Add direction information to each batch
            forward_batches = [(batch_x, batch_cond, torch.zeros(batch_x.shape[0], 1)) for batch_x, batch_cond in forward_batches]
            reverse_batches = [(batch_x, batch_cond, torch.ones(batch_x.shape[0], 1)) for batch_x, batch_cond in reverse_batches]
            
            # Combine batches from both directions
            combined_batches = forward_batches + reverse_batches
            
            # Split into train and test
            train_size = int(0.8 * len(combined_batches))
            train_batches = combined_batches[:train_size]
            test_batches = combined_batches[train_size:]
            
            class DirectionalBatchDataLoader:
                """Custom DataLoader for combined directional batches"""
                def __init__(self, batches, shuffle=True):
                    self.batches = batches
                    self.shuffle = shuffle
                    
                def __iter__(self):
                    indices = list(range(len(self.batches)))
                    if self.shuffle:
                        np.random.shuffle(indices)
                    
                    for i in indices:
                        yield self.batches[i]
                        
                def __len__(self):
                    return len(self.batches)
            
            train_loader = DirectionalBatchDataLoader(train_batches, shuffle=shuffle)
            test_loader = DirectionalBatchDataLoader(test_batches, shuffle=False)
            
            return train_loader, test_loader
        
        # Handle single direction ('forward' or 'reverse')
        dataset = forward_dataset if direction == 'forward' else reverse_dataset
        
        # Get all batches
        all_batches = dataset.get_grouped_batches(batch_size=batch_size, shuffle=shuffle)
        
        # Split into train and test
        train_size = int(0.8 * len(all_batches))
        train_batches = all_batches[:train_size]
        test_batches = all_batches[train_size:]
        
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
                
                # Return batches
                for i in indices:
                    yield self.batches[i]
                    
            def __len__(self):
                return len(self.batches)
        
        train_loader = GroupedBatchDataLoader(train_batches, shuffle=shuffle)
        test_loader = GroupedBatchDataLoader(test_batches, shuffle=False)
        
        return train_loader, test_loader


def train_counterfactual_flow_model(
    dataset: CounterfactualWrapper,
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
    direction: str = 'forward',
    bidirectional_model: bool = False
):
    """
    Train a Conditional Normalizing Flow model for counterfactual generation.
    The model conditions on factual points to generate counterfactual points.
    
    Args:
        dataset: CounterfactualWrapper instance
        flow_model_class: Class of the flow model to use (e.g., MaskedAutoregressiveFlow)
        hidden_features: Number of hidden features in flow model
        num_layers: Number of layers in flow model
        num_blocks_per_layer: Number of blocks per layer in flow model
        learning_rate: Learning rate for optimizer
        batch_size: Batch size for training (defaults to n_nearest if None)
        num_epochs: Number of epochs to train
        patience: Number of epochs to wait for improvement before early stopping
        noise_level: Standard deviation of Gaussian noise to add during training
        device: Device to use for training ("cuda" or "cpu")
        save_dir: Directory to save results
        log_interval: Interval for logging detailed metrics
        direction: 'forward', 'reverse', or 'both' for the direction of counterfactual generation
        bidirectional_model: If True, trains a model that can handle both directions with a direction indicator
    
    Returns:
        Trained flow model
    """
    start_time = time.time()
    logger.info(f"Starting counterfactual flow model training on device: {device}")
    logger.info(f"Model architecture: {num_layers} layers with {hidden_features} hidden features")
    logger.info(f"Training direction: {direction}, Bidirectional model: {bidirectional_model}")
    
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
        direction=direction
    )
    logger.info(f"Created data loaders - Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    
    # Initialize model
    context_features = dataset.X_factual.shape[1]  # Dimensionality of factual points
    features = dataset.X_counterfactual.shape[1]  # Dimensionality of counterfactual points
    
    # If bidirectional model, add direction indicator to context
    if bidirectional_model:
        context_features += 1  # Add one feature to indicate direction
    
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
            
            # Unpack batch data - format depends on whether we're using bidirectional mode
            if bidirectional_model and direction == 'both':
                x_batch, cond_batch, dir_indicator = batch_data
                # Combine condition and direction indicator
                cond_batch = torch.cat([cond_batch, dir_indicator], dim=1)
            else:
                x_batch, cond_batch = batch_data
            
            # Move data to device and ensure dtype is float32
            x_batch = x_batch.to(device).float()
            cond_batch = cond_batch.to(device).float()
            
            # Forward pass
            optimizer.zero_grad()
            log_prob = model(x_batch, cond_batch)
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
                # Unpack batch data - format depends on whether we're using bidirectional mode
                if bidirectional_model and direction == 'both':
                    x_batch, cond_batch, dir_indicator = batch_data
                    # Combine condition and direction indicator
                    cond_batch = torch.cat([cond_batch, dir_indicator], dim=1)
                else:
                    x_batch, cond_batch = batch_data
                
                # Move data to device and ensure dtype is float32
                x_batch = x_batch.to(device).float()
                cond_batch = cond_batch.to(device).float()
                
                # Forward pass
                log_prob = model(x_batch, cond_batch)
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
                'bidirectional': bidirectional_model,
                'direction': direction,
                'context_features': context_features,
                'features': features
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


def generate_counterfactuals(
    model,
    factual_points: np.ndarray,
    n_samples: int = 10,
    temperature: float = 0.8,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    direction_indicator: Optional[float] = None,
    bidirectional_model: bool = False
):
    """
    Generate counterfactual samples for given factual points.
    
    Args:
        model: Trained flow model
        factual_points: Array of factual points to generate counterfactuals for
        n_samples: Number of counterfactual samples to generate per factual point
        temperature: Temperature for sampling (higher = more diverse)
        device: Device to use for generation
        direction_indicator: For bidirectional models, indicates direction (0=forward, 1=reverse)
        bidirectional_model: Whether the model was trained bidirectionally
    
    Returns:
        Array of generated counterfactual samples
    """
    model.eval()
    
    all_counterfactuals = []
    
    with torch.no_grad():
        for factual in factual_points:
            # Convert to tensor and add batch dimension
            factual_tensor = torch.tensor(factual, dtype=torch.float32).unsqueeze(0).to(device)
            
            # If using bidirectional model, append direction indicator to context
            if bidirectional_model and direction_indicator is not None:
                # Create direction tensor
                dir_tensor = torch.tensor([direction_indicator], dtype=torch.float32).unsqueeze(0).to(device)
                # Concatenate with factual point
                context = torch.cat([factual_tensor, dir_tensor], dim=1)
            else:
                context = factual_tensor
            
            # Generate samples
            samples, _ = model.sample_and_log_prob(
                num_samples=n_samples,
                context=context,
                temp=temperature
            )
            
            # Convert to numpy
            samples = samples.cpu().numpy()
            
            # Add to results
            all_counterfactuals.append(samples)
    
    return all_counterfactuals


def visualize_batch(batch, save_path=None, title="Batch Visualization"):
    """
    Visualize a batch of data for 2D datasets.
    
    Args:
        batch: Tuple of (counterfactual_points, factual_points)
        save_path: Path to save the visualization (if None, just displays it)
        title: Title for the plot
    """
    x_batch, cond_batch = batch
    
    # Convert to numpy for plotting
    x_np = x_batch.numpy()
    cond_np = cond_batch.numpy()
    
    # Only works for 2D data
    assert x_np.shape[1] == 2 and cond_np.shape[1] == 2, "Only 2D data is supported for visualization"
    
    plt.figure(figsize=(10, 8))
    
    # Plot factual points (conditioning)
    plt.scatter(
        cond_np[:, 0], 
        cond_np[:, 1], 
        color='blue', 
        s=100, 
        alpha=0.8,
        label='Factual Point (Condition)',
        marker='*',
        edgecolor='black'
    )
    
    # Plot counterfactual points (targets)
    plt.scatter(
        x_np[:, 0], 
        x_np[:, 1], 
        color='red', 
        s=80, 
        alpha=0.7,
        label='Counterfactual Points (Targets)',
        edgecolor='black'
    )
    
    # Draw lines connecting factual to counterfactual
    for i in range(len(x_np)):
        plt.plot(
            [cond_np[i, 0], x_np[i, 0]], 
            [cond_np[i, 1], x_np[i, 1]], 
            'k--', 
            alpha=0.5
        )
    
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
    
    return plt


def visualize_factual_counterfactual_mapping(
    dataset,
    num_points=10,
    save_path=None,
    title="Factual-Counterfactual Mapping"
):
    """
    Visualize the factual to counterfactual mapping for a sample of points.
    Shows how each factual point maps to its nearest counterfactual points.
    
    Args:
        dataset: CounterfactualWrapper instance
        num_points: Number of factual points to visualize
        save_path: Path to save the visualization
        title: Title for the plot
    """
    # Only works for 2D data
    assert dataset.X_factual.shape[1] == 2, "Only 2D data is supported for visualization"
    
    # Get a sample of factual points
    if len(dataset.X_factual_scaled) <= num_points:
        factual_indices = np.arange(len(dataset.X_factual_scaled))
    else:
        factual_indices = np.random.choice(
            len(dataset.X_factual_scaled), 
            size=num_points, 
            replace=False
        )
    
    # Get the counterfactual dataset
    cf_dataset = GenericCounterfactualDataset(
        dataset.X_factual_scaled,
        dataset.X_counterfactual_scaled,
        n_nearest=dataset.n_nearest,
        noise_level=0  # No noise for visualization
    )
    
    plt.figure(figsize=(12, 10))
    
    # Plot all factual points with low opacity
    plt.scatter(
        dataset.X_factual_scaled[:, 0],
        dataset.X_factual_scaled[:, 1],
        color='blue',
        alpha=0.2,
        label='All Factual Points'
    )
    
    # Plot all counterfactual points with low opacity
    plt.scatter(
        dataset.X_counterfactual_scaled[:, 0],
        dataset.X_counterfactual_scaled[:, 1],
        color='red',
        alpha=0.2,
        label='All Counterfactual Points'
    )
    
    # Colors for different factual points
    colors = plt.cm.tab10(np.linspace(0, 1, num_points))
    
    # For selected factual points
    for i, f_idx in enumerate(factual_indices):
        # Get factual point
        factual = dataset.X_factual_scaled[f_idx]
        
        # Get nearest counterfactual indices
        cf_indices = cf_dataset.nearest_indices[f_idx]
        
        # Get counterfactual points
        counterfactuals = dataset.X_counterfactual_scaled[cf_indices]
        
        # Plot factual point with high opacity
        plt.scatter(
            factual[0],
            factual[1],
            color=colors[i],
            s=150,
            marker='*',
            edgecolor='black',
            label=f'Factual {i+1}' if i < 5 else None  # Limit legend entries
        )
        
        # Plot nearest counterfactual points
        plt.scatter(
            counterfactuals[:, 0],
            counterfactuals[:, 1],
            color=colors[i],
            s=80,
            alpha=0.7,
            marker='o',
            edgecolor='black'
        )
        
        # Draw lines from factual to counterfactuals
        for cf in counterfactuals:
            plt.plot(
                [factual[0], cf[0]],
                [factual[1], cf[1]],
                color=colors[i],
                linestyle='--',
                alpha=0.5
            )
    
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
    
    return plt


def visualize_batch_distribution(dataset, batch_size=None, save_dir=None):
    """
    Visualize the distribution of batches created by the dataset.
    
    Args:
        dataset: CounterfactualWrapper instance
        batch_size: Batch size for training (defaults to n_nearest if None)
        save_dir: Directory to save visualizations
    """
    # Get data loaders
    train_loader, _ = dataset.get_counterfactual_dataloaders(
        batch_size=batch_size,
        shuffle=True
    )
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Create batches subfolder
    batches_dir = os.path.join(save_dir, "batches") if save_dir else None
    if batches_dir:
        os.makedirs(batches_dir, exist_ok=True)
    
    # Select some batches to visualize
    num_batches = min(5, len(train_loader))
    batch_indices = np.random.choice(len(train_loader.batches), num_batches, replace=False)
    
    for i, idx in enumerate(batch_indices):
        batch = train_loader.batches[idx]
        
        # Visualize the batch
        if batches_dir:
            save_path = os.path.join(batches_dir, f"batch_{i+1}.png")
            visualize_batch(
                batch, 
                save_path=save_path,
                title=f"Batch {i+1}: {len(batch[0])} points"
            )
    
    # Visualize the mapping between factual and counterfactual points
    if save_dir:
        save_path = os.path.join(save_dir, "factual_counterfactual_mapping.png")
        visualize_factual_counterfactual_mapping(
            dataset,
            num_points=10,
            save_path=save_path
        )
    
    # Histogram of distances
    plt.figure(figsize=(10, 6))
    
    cf_dataset = GenericCounterfactualDataset(
        dataset.X_factual_scaled,
        dataset.X_counterfactual_scaled,
        n_nearest=dataset.n_nearest
    )
    
    # Flatten the distances to nearest counterfactuals
    distances = []
    for f_idx in range(len(dataset.X_factual_scaled)):
        cf_indices = cf_dataset.nearest_indices[f_idx]
        for cf_idx in cf_indices:
            distances.append(cf_dataset.dist_matrix[f_idx, cf_idx])
    
    plt.hist(distances, bins=30, alpha=0.7)
    plt.xlabel('Distance from Factual to Counterfactual')
    plt.ylabel('Frequency')
    plt.title('Distribution of Factual-Counterfactual Distances')
    plt.grid(True, alpha=0.3)
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, "distance_histogram.png"))
        plt.close()
    else:
        plt.show()
    
    # Statistics about batches
    batch_sizes = [len(batch[0]) for batch in train_loader.batches]
    
    plt.figure(figsize=(10, 6))
    plt.hist(batch_sizes, bins=range(min(batch_sizes), max(batch_sizes) + 2), alpha=0.7)
    plt.xlabel('Batch Size')
    plt.ylabel('Frequency')
    plt.title('Distribution of Batch Sizes')
    plt.grid(True, alpha=0.3)
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, "batch_size_histogram.png"))
        plt.close()
    else:
        plt.show()
    
    # Log batch statistics
    logger.info(f"Batch statistics: Min size={min(batch_sizes)}, Max size={max(batch_sizes)}, "
               f"Average size={np.mean(batch_sizes):.2f}")
    logger.info(f"Distance statistics: Min={np.min(distances):.4f}, Max={np.max(distances):.4f}, "
               f"Average={np.mean(distances):.4f}")
    
    return num_batches


def visualize_counterfactual_generation(
    model,
    dataset,
    num_factual=5,
    num_samples=20,
    temperature=0.8,
    save_dir=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
    direction="forward",
    bidirectional_model=False
):
    """
    Visualize counterfactual generation results.
    
    Args:
        model: Trained flow model
        dataset: CounterfactualWrapper instance
        num_factual: Number of factual points to generate counterfactuals for
        num_samples: Number of counterfactual samples to generate per factual point
        temperature: Temperature for sampling (higher = more diversity)
        save_dir: Directory to save visualizations
        device: Device to use for generation
        direction: Direction of counterfactual generation ('forward', 'reverse', 'both')
        bidirectional_model: Whether the model was trained bidirectionally
    """
    # Only works for 2D data
    assert dataset.X_factual.shape[1] == 2, "Only 2D data is supported for visualization"
    
    # Create samples directory
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        samples_dir = os.path.join(save_dir, "samples")
        os.makedirs(samples_dir, exist_ok=True)
    
    results = []
    
    # Handle different directions
    directions_to_process = []
    if direction == 'both' and dataset.bidirectional:
        directions_to_process = ['forward', 'reverse']
    else:
        directions_to_process = [direction]
    
    for curr_direction in directions_to_process:
        logger.info(f"Generating counterfactuals for direction: {curr_direction}")
        
        # Select factual and counterfactual points based on direction
        if curr_direction == 'forward':
            factual_scaled = dataset.X_factual_scaled
            factual_original = dataset.X_factual
            counterfactual_class = dataset.counterfactual_class
            direction_indicator = 0.0
        elif curr_direction == 'reverse':
            if not dataset.bidirectional:
                raise ValueError("Cannot use 'reverse' direction when dataset is not bidirectional")
            factual_scaled = dataset.X_factual_scaled_rev
            factual_original = dataset.X_factual_rev
            counterfactual_class = dataset.factual_class
            direction_indicator = 1.0
        else:
            raise ValueError(f"Invalid direction: {curr_direction}")
        
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
        
        # Generate counterfactuals
        logger.info(f"Generating {num_samples} counterfactuals for {len(factual_points)} factual points")
        generated_cfs = generate_counterfactuals(
            model=model,
            factual_points=factual_points,
            n_samples=num_samples,
            temperature=temperature,
            device=device,
            direction_indicator=direction_indicator if bidirectional_model else None,
            bidirectional_model=bidirectional_model
        )
        
        # Convert to original scale for better interpretability
        factual_orig = dataset.feature_transformer.inverse_transform(factual_points)
        
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
        
        # Store results for this direction
        results.append({
            'direction': curr_direction,
            'factual_indices': factual_indices,
            'factual_points': factual_points,
            'factual_orig': factual_orig,
            'generated_cfs': generated_cfs,
            'generated_cfs_orig': generated_cfs_orig,
            'counterfactual_class': counterfactual_class
        })
        
        # Plot settings
        colors = plt.cm.tab10(np.linspace(0, 1, num_factual))
        
        # Create overview plot for this direction
        plt.figure(figsize=(14, 10))
        
        # Plot all original data points with low opacity
        plt.scatter(
            dataset.X[:, 0],
            dataset.X[:, 1],
            c=dataset.y,
            cmap=plt.cm.coolwarm,
            alpha=0.2,
            s=30
        )
        
        # Add a legend for the original classes
        plt.scatter([], [], color=plt.cm.coolwarm(0), label=f'Class {dataset.factual_class}')
        plt.scatter([], [], color=plt.cm.coolwarm(1), label=f'Class {dataset.counterfactual_class}')
        
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
                plt.figure(figsize=(10, 8))
                
                # Plot original data with low opacity
                plt.scatter(
                    dataset.X[:, 0],
                    dataset.X[:, 1],
                    c=dataset.y,
                    cmap=plt.cm.coolwarm,
                    alpha=0.2,
                    s=30
                )
                
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
                
                # Add density contour of generated points
                try:
                    from scipy.stats import gaussian_kde
                    
                    # If we have enough points, create a density plot
                    if len(cf_to_plot) >= 10:
                        kde = gaussian_kde(cf_to_plot.T)
                        
                        # Create a grid of points
                        x_min, x_max = plt.xlim()
                        y_min, y_max = plt.ylim()
                        
                        xx, yy = np.meshgrid(
                            np.linspace(x_min, x_max, 100),
                            np.linspace(y_min, y_max, 100)
                        )
                        
                        # Evaluate KDE on grid
                        positions = np.vstack([xx.ravel(), yy.ravel()])
                        zz = kde(positions).reshape(xx.shape)
                        
                        # Plot contour
                        plt.contour(
                            xx, yy, zz, 
                            cmap=plt.cm.Oranges, 
                            alpha=0.5
                        )
                except Exception as e:
                    logger.warning(f"Could not create density plot: {e}")
                
                # Draw lines to all counterfactuals
                for j in range(len(cf_to_plot)):
                    plt.plot(
                        [factual[0], cf_to_plot[j, 0]],
                        [factual[1], cf_to_plot[j, 1]],
                        color=colors[i],
                        linestyle='--',
                        alpha=0.3
                    )
                
                direction_label = "→" if curr_direction == 'forward' else "←"
                plt.title(f"Counterfactuals Generated for Factual Point {i+1} ({curr_direction} {direction_label})")
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Create subdirectory for each direction if needed
                subdir = os.path.join(samples_dir, curr_direction)
                os.makedirs(subdir, exist_ok=True)
                
                plt.savefig(os.path.join(subdir, f"factual_{i+1}_counterfactuals.png"))
                plt.close()
        
        # Finish and save overview plot
        direction_label = "→" if curr_direction == 'forward' else "←"
        plt.title(f"Overview of Generated Counterfactuals ({curr_direction} {direction_label})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, f"counterfactual_overview_{curr_direction}.png"))
            plt.close()
    
    # If we processed both directions and need to return results
    if len(results) > 1:
        # Merge results from both directions
        factual_points = np.vstack([r['factual_points'] for r in results])
        generated_cfs = []
        for r in results:
            generated_cfs.extend(r['generated_cfs'])
        return generated_cfs, factual_points
    elif len(results) == 1:
        return results[0]['generated_cfs'], results[0]['factual_points']
    else:
        return [], [] 