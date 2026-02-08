"""CeFlow GMM model with variational dequantization."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.optim as optim
from nflows.flows import SimpleRealNVP
from tqdm import tqdm

from cel.dequantization.variational_dequantizer import (
    VariationalDequantizer,
)
from cel.models.generative.gmm_base import GMMBaseDistribution
from cel.models.generative_mixin import GenerativePytorchMixin
from cel.models.pytorch_base import PytorchBase


class CeFlowGMM(PytorchBase, GenerativePytorchMixin):
    """CeFlow with class-conditional GMM base distribution."""

    def __init__(
        self,
        features: int,
        n_classes: int = 2,
        categorical_groups: Optional[list[list[int]]] = None,
        hidden_features: int = 64,
        num_layers: int = 5,
        num_blocks_per_layer: int = 2,
        dequant_hidden_dim: int = 32,
        gmm_init_std: float = 1.0,
        learn_gmm_covariance: bool = True,
        dropout_probability: float = 0.0,
        batch_norm_within_layers: bool = False,
        device: str = "cpu",
    ) -> None:
        super().__init__(num_inputs=features, num_targets=n_classes)
        self.features = features
        self.n_classes = n_classes
        self.categorical_groups = categorical_groups or []
        self.device = device

        self.dequantizer = VariationalDequantizer(
            categorical_groups=self.categorical_groups,
            total_features=features,
            hidden_dim=dequant_hidden_dim,
        )
        self.flow = SimpleRealNVP(
            features=features,
            hidden_features=hidden_features,
            num_layers=num_layers,
            num_blocks_per_layer=num_blocks_per_layer,
            use_volume_preserving=False,
            dropout_probability=dropout_probability,
            batch_norm_within_layers=batch_norm_within_layers,
        )
        self.gmm_base = GMMBaseDistribution(
            features=features,
            n_classes=n_classes,
            init_std=gmm_init_std,
            learn_covariance=learn_gmm_covariance,
        )
        self.to(device)

    def forward(
        self, x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute log probabilities for inputs."""
        dq_output = self.dequantizer(x, return_log_prob=True)
        z_dequant = dq_output.values
        log_q_dequant = dq_output.log_q

        z, log_det = self.flow._transform.forward(z_dequant)
        log_prob_base = self.gmm_base.log_prob(z, y)
        if log_q_dequant is None:
            log_q_dequant = torch.zeros_like(log_prob_base)
        return log_prob_base + log_det - log_q_dequant

    def transform_to_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Transform inputs to latent space deterministically."""
        with torch.no_grad():
            dq_output = self.dequantizer(x, return_log_prob=False, use_mean=True)
            z_dequant = dq_output.values
            z, _ = self.flow._transform.forward(z_dequant)
        return z

    def transform_to_data(self, z: torch.Tensor) -> torch.Tensor:
        """Transform latent vectors back to data space."""
        with torch.no_grad():
            output = self.flow._transform.inverse(z)
            x_dequant = output[0] if isinstance(output, tuple) else output
            x = self.dequantizer.inverse(x_dequant)
        return x

    def get_class_means(self) -> dict[int, torch.Tensor]:
        """Return class means from the GMM base distribution."""
        return {idx: self.gmm_base.get_class_mean(idx) for idx in range(self.n_classes)}

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        epochs: int = 100,
        lr: float = 1e-3,
        patience: int = 20,
        eps: float = 1e-3,
        checkpoint_path: str = "ceflow_gmm_best.pt",
        dequantizer: Optional[object] = None,
    ) -> None:
        """Train CeFlow GMM with joint objective."""
        for x_batch, _ in train_loader:
            self.dequantizer.fit(x_batch.numpy())
            break

        optimizer = optim.Adam(self.parameters(), lr=lr)
        patience_counter = 0
        min_test_loss = float("inf")

        for epoch in (pbar := tqdm(range(epochs))):
            self.train()
            train_loss = 0.0
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device).long()
                optimizer.zero_grad()
                log_prob = self(x_batch, y_batch)
                loss = -log_prob.mean()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            self.eval()
            test_loss = 0.0
            with torch.no_grad():
                for x_batch, y_batch in test_loader:
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device).long()
                    log_prob = self(x_batch, y_batch)
                    test_loss += -log_prob.mean().item()
            test_loss /= len(test_loader)

            pbar.set_description(
                "Epoch %s, Train: %.4f, Test: %.4f, Patience: %s"
                % (epoch, train_loss, test_loss, patience_counter)
            )

            if test_loss < (min_test_loss - eps):
                min_test_loss = test_loss
                patience_counter = 0
                self.save(checkpoint_path)
            else:
                patience_counter += 1
            if patience_counter > patience:
                break

        self.load(checkpoint_path)

    def predict_log_prob(self, dataloader: torch.utils.data.DataLoader) -> torch.Tensor:
        """Compute log probabilities for a dataloader."""
        self.eval()
        log_probs = []
        with torch.no_grad():
            for x_batch, y_batch in dataloader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device).long()
                log_prob = self(x_batch, y_batch)
                log_probs.append(log_prob)
        return torch.cat(log_probs)

    def predict_log_proba(
        self, X_test: np.ndarray, context: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Predict log probabilities for numpy array inputs."""
        x = torch.from_numpy(X_test).float().to(self.device)
        y = None
        if context is not None:
            y = torch.from_numpy(context).long().to(self.device)
        self.eval()
        with torch.no_grad():
            log_prob = self(x, y)
        return log_prob.cpu().numpy()

    def sample_and_log_proba(
        self, n_samples: int, context: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample from the model and return log probabilities."""
        y = None
        if context is not None:
            y = torch.from_numpy(context).long().to(self.device)
        self.eval()
        with torch.no_grad():
            z = self.gmm_base.sample(n_samples, y)
            x = self.transform_to_data(z)
            log_prob = self(x, y)
        return x.cpu().numpy(), log_prob.cpu().numpy()

    def save(self, path: str) -> None:
        """Save model state."""
        categorical_groups = [list(group) for group in self.categorical_groups]
        dequantizer_dividers = (
            [int(val) for val in self.dequantizer.dividers]
            if self.dequantizer.dividers is not None
            else None
        )
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "features": self.features,
                "n_classes": self.n_classes,
                "categorical_groups": categorical_groups,
                "dequantizer_dividers": dequantizer_dividers,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load model state."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.load_state_dict(checkpoint["model_state_dict"])
        if "dequantizer_dividers" in checkpoint:
            self.dequantizer.dividers = checkpoint["dequantizer_dividers"]
