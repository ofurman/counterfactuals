import os
import time
import logging
from typing import Optional

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from counterfactuals.cf_methods.dicoflex.dataset import MulticlassCounterfactualWrapper
from counterfactuals.cf_methods.dicoflex.utils import transform_batch_data

logger = logging.getLogger('counterfactual')


def train_multiclass_counterfactual_flow_model(
    dataset: MulticlassCounterfactualWrapper,
    flow_model_class,
    mask_features,
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
    context_features = dataset.X.shape[1] + mask_features + 1 # Dimensionality of factual points
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
            x_batch = transform_batch_data(x_batch, device)
            cond_batch = transform_batch_data(cond_batch, device)
            class_batch = transform_batch_data(class_batch, device)
            p = transform_batch_data(p, device)
            mask = transform_batch_data(mask, device)

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
                x_batch = transform_batch_data(x_batch, device)
                cond_batch = transform_batch_data(cond_batch, device)
                class_batch = transform_batch_data(class_batch, device)
                p = transform_batch_data(p, device)
                mask = transform_batch_data(mask, device)

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
