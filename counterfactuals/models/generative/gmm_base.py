"""Gaussian mixture base distribution for conditional flows."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class GMMBaseDistribution(nn.Module):
    """Class-conditional Gaussian mixture base distribution."""

    def __init__(
        self,
        features: int,
        n_classes: int,
        init_std: float = 1.0,
        learn_covariance: bool = True,
    ) -> None:
        super().__init__()
        self.features = features
        self.n_classes = n_classes
        self.learn_covariance = learn_covariance
        self.means = nn.Parameter(torch.randn(n_classes, features) * 0.1)
        if learn_covariance:
            self.log_stds = nn.Parameter(
                torch.full((n_classes, features), np.log(init_std))
            )
        else:
            self.register_buffer("log_stds", torch.zeros(n_classes, features))

    def log_prob(
        self, z: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute log probabilities under the GMM."""
        if y is not None:
            return self._log_prob_conditional(z, y)
        return self._log_prob_marginal(z)

    def _log_prob_conditional(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y = y.long().view(-1)
        mu = self.means[y]
        log_std = self.log_stds[y]
        std = torch.exp(log_std)
        log_prob = -0.5 * (
            self.features * np.log(2 * np.pi)
            + 2 * log_std.sum(dim=1)
            + ((z - mu) / std).pow(2).sum(dim=1)
        )
        return log_prob

    def _log_prob_marginal(self, z: torch.Tensor) -> torch.Tensor:
        log_probs = []
        for idx in range(self.n_classes):
            mu = self.means[idx]
            log_std = self.log_stds[idx]
            std = torch.exp(log_std)
            log_prob = -0.5 * (
                self.features * np.log(2 * np.pi)
                + 2 * log_std.sum()
                + ((z - mu) / std).pow(2).sum(dim=1)
            )
            log_probs.append(log_prob)
        stacked = torch.stack(log_probs, dim=1)
        return -np.log(self.n_classes) + torch.logsumexp(stacked, dim=1)

    def sample(self, n_samples: int, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sample from the GMM."""
        device = self.means.device
        if y is None:
            y = torch.randint(0, self.n_classes, (n_samples,), device=device)
        y = y.long().view(-1)
        mu = self.means[y]
        std = torch.exp(self.log_stds[y])
        eps = torch.randn(n_samples, self.features, device=device)
        return mu + std * eps

    def get_class_mean(self, k: int) -> torch.Tensor:
        """Get mean vector for class k."""
        return self.means[k].detach()

    def get_all_means(self) -> torch.Tensor:
        """Get all class means."""
        return self.means.detach()
