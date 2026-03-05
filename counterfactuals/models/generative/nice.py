from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from nflows.flows import SimpleRealNVP as _SimpleRealNVP
from tqdm import tqdm

from counterfactuals.models.generative_mixin import GenerativePytorchMixin
from counterfactuals.models.pytorch_base import PytorchBase


class NICE(PytorchBase, GenerativePytorchMixin):
    def __init__(
        self,
        features: int,
        hidden_features: int,
        context_features: Optional[int] = None,
        num_layers: int = 5,
        num_blocks_per_layer: int = 2,
        use_residual_blocks: bool = True,
        use_random_masks: bool = False,
        use_random_permutations: bool = False,
        activation=F.relu,
        dropout_probability: float = 0.0,
        batch_norm_within_layers: bool = False,
        batch_norm_between_layers: bool = False,
        device: str = "cpu",
    ):
        super(NICE, self).__init__(features, context_features)
        self.features = features
        self.hidden_features = hidden_features
        self.context_features = context_features
        self.num_layers = num_layers
        self.num_blocks_per_layer = num_blocks_per_layer
        self.use_residual_blocks = use_residual_blocks
        self.use_random_masks = use_random_masks
        self.use_random_permutations = use_random_permutations
        self.activation = activation
        self.dropout_probability = dropout_probability
        self.batch_norm_within_layers = batch_norm_within_layers
        self.batch_norm_between_layers = batch_norm_between_layers
        self.device = device
        self.model = _SimpleRealNVP(
            features=features,
            hidden_features=hidden_features,
            context_features=context_features,
            num_layers=num_layers,
            num_blocks_per_layer=num_blocks_per_layer,
            activation=activation,
            use_volume_preserving=True,
            dropout_probability=dropout_probability,
            batch_norm_within_layers=batch_norm_within_layers,
            batch_norm_between_layers=batch_norm_between_layers,
        )

    def forward(self, x, context=None):
        if context is not None:
            context = context.view(-1, 1)
        return self.model.log_prob(inputs=x, context=context)

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        epochs: int = 100,
        lr: float = 1e-3,
        patience: int = 20,
        eps: float = 1e-3,
        checkpoint_path: str = "best_model.pth",
    ):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        patience_counter = 0
        min_test_loss = float("inf")

        for epoch in (pbar := tqdm(range(epochs))):
            self.train()
            train_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                labels = labels.type(torch.float32)
                labels = None if self.context_features is None else labels
                optimizer.zero_grad()
                log_likelihood = self(inputs, labels)
                loss = -log_likelihood.mean()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            self.eval()
            test_loss = 0.0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    labels = labels.type(torch.float32)
                    labels = None if self.context_features is None else labels
                    log_likelihood = self(inputs, labels)
                    loss = -log_likelihood.mean().item()
                    test_loss += loss
            test_loss /= len(test_loader)
            pbar.set_description(
                f"Epoch {epoch}, Train: {train_loss:.4f}, test: {test_loss:.4f}, patience: {patience_counter}"
            )
            if test_loss < (min_test_loss + eps):
                min_test_loss = test_loss
                patience_counter = 0
                self.save(checkpoint_path)
            else:
                patience_counter += 1
            if patience_counter > patience:
                break
        self.load(checkpoint_path)

    def predict_log_prob(self, dataloader):
        self.eval()
        log_probs = []

        with torch.no_grad():
            for inputs, labels in dataloader:
                labels = labels.type(torch.float32)
                outputs = self(inputs, labels)
                log_probs.append(outputs)

        return torch.hstack(log_probs)

    def sample_and_log_proba(self, n_samples: int, context: Optional[np.ndarray] = None):
        """Sample from the model and return (samples, log_probs) as numpy arrays."""
        if context is not None and self.context_features is not None:
            if isinstance(context, np.ndarray):
                context = torch.from_numpy(context).float()
            context = context.view(-1, self.context_features)
        self.eval()
        with torch.no_grad():
            samples, log_probs = self.model.sample_and_log_prob(
                num_samples=n_samples, context=context
            )
            return samples.cpu().numpy(), log_probs.cpu().numpy()

    def predict_log_proba(
        self, X_test: np.ndarray, context: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Predict log probabilities for input data (numpy array) and return numpy array."""
        if isinstance(X_test, np.ndarray):
            X_test = torch.from_numpy(X_test).float()
            if context is not None:
                context = torch.from_numpy(context).float()

        self.eval()
        with torch.no_grad():
            log_probs = self(X_test, context=context)
            return log_probs.cpu().numpy()

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
