import os
from typing import List, Tuple, Optional

import numpy as np
import torch

from counterfactuals.cf_methods.dicoflex.dataset import MulticlassCounterfactualWrapper
from counterfactuals.cf_methods.dicoflex.training import train_multiclass_counterfactual_flow_model
from counterfactuals.cf_methods.dicoflex.generation import generate_multiclass_counterfactuals


class DiCoFlex:
    """DiCoFlex: Diverse Counterfactual Explanations via Normalizing Flows.

    A normalizing flow-based approach for generating diverse, sparsity-constrained
    counterfactual explanations across multiple target classes.

    Usage:
        dicoflex = DiCoFlex(flow_model_class=MaskedAutoregressiveFlow)
        dicoflex.fit(X_train, y_train, masks=masks, p_values=[1e-2, 2.0])
        cfs, log_probs = dicoflex.generate(factual_points, target_class=1, p_value=2.0, mask=mask)
    """

    def __init__(
        self,
        flow_model_class,
        hidden_features: int = 64,
        num_layers: int = 5,
        num_blocks_per_layer: int = 2,
        device: str = None,
    ):
        self.flow_model_class = flow_model_class
        self.hidden_features = hidden_features
        self.num_layers = num_layers
        self.num_blocks_per_layer = num_blocks_per_layer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.dataset_wrapper = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        masks: np.ndarray,
        p_values: List[float],
        disc_model=None,
        prob_threshold: float = 0.0,
        n_nearest: int = 16,
        noise_level: float = 0.01,
        learning_rate: float = 1e-3,
        num_epochs: int = 10000,
        patience: int = 50,
        batch_size: int = 256,
        save_dir: str = "results",
        balanced: bool = True,
        load_from_save_dir: bool = False,
        numerical_pos: int = 0,
        log_interval: int = 10,
    ):
        """Build the counterfactual dataset and train the conditional flow model.

        Args:
            X: Feature matrix (N x D)
            y: Labels
            masks: Array of feature masks
            p_values: List of p-norms for distance computation
            disc_model: Optional discriminator for probability thresholding
            prob_threshold: Probability threshold for classifier filtering
            n_nearest: Number of nearest counterfactual points per factual
            noise_level: Gaussian noise std for training augmentation
            learning_rate: Adam optimizer learning rate
            num_epochs: Maximum training epochs
            patience: Early stopping patience
            batch_size: Training batch size
            save_dir: Directory to save model checkpoints
            balanced: Whether to balance classes in batches
            load_from_save_dir: Load pre-trained model from save_dir
            numerical_pos: Number of numerical features (first numerical_pos columns)
            log_interval: Logging interval in epochs
        """
        self.dataset_wrapper = MulticlassCounterfactualWrapper(
            X=X,
            y=y,
            factual_classes=np.unique(y),
            p_values=p_values,
            masks=masks,
            n_nearest=n_nearest,
            noise_level=noise_level,
            classifier=disc_model,
            prob_threshold=prob_threshold,
            log_level='INFO',
            numerical_pos=numerical_pos,
        )

        mask_features = len(masks)

        self.model = train_multiclass_counterfactual_flow_model(
            dataset=self.dataset_wrapper,
            flow_model_class=self.flow_model_class,
            mask_features=mask_features,
            hidden_features=self.hidden_features,
            num_layers=self.num_layers,
            num_blocks_per_layer=self.num_blocks_per_layer,
            learning_rate=learning_rate,
            batch_size=batch_size,
            num_epochs=num_epochs,
            patience=patience,
            noise_level=noise_level,
            device=self.device,
            save_dir=os.path.join(save_dir, "multiclass_model"),
            log_interval=log_interval,
            balanced=balanced,
            load_from_save_dir=load_from_save_dir,
        )

        return self

    def generate(
        self,
        factual_points: np.ndarray,
        target_class: int,
        p_value: float,
        mask: np.ndarray,
        n_samples: int = 100,
        temperature: float = 0.8,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate counterfactual samples for given factual points.

        Args:
            factual_points: Points to generate counterfactuals for (N x D)
            target_class: Target class label
            p_value: p-norm value
            mask: Feature mask (one-hot encoded mask index)
            n_samples: Number of counterfactual samples per factual point
            temperature: Sampling temperature (higher = more diverse)

        Returns:
            Tuple of (counterfactuals, log_probs) where counterfactuals has shape
            (N, n_samples, D) and log_probs has shape (N, n_samples)
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        return generate_multiclass_counterfactuals(
            model=self.model,
            factual_points=factual_points,
            target_class=target_class,
            p_value=p_value,
            mask=mask,
            n_samples=n_samples,
            temperature=temperature,
            device=self.device,
            num_classes=len(self.dataset_wrapper.classes),
        )

    def save(self, path: str):
        """Save the trained model to disk."""
        if self.model is None:
            raise RuntimeError("No model to save. Call fit() first.")
        torch.save({
            'model_state_dict': self.model.state_dict(),
        }, path)

    def load(self, path: str):
        """Load a trained model from disk."""
        if self.model is None:
            raise RuntimeError("Initialize model via fit() with load_from_save_dir=True instead.")
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
