"""Variational dequantizer for categorical features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

ALPHA = 1e-6


@dataclass
class DequantizerOutput:
    """Dequantization output container."""

    values: torch.Tensor
    log_q: Optional[torch.Tensor]


class VariationalDequantizer(nn.Module):
    """Learnable dequantization for categorical features."""

    def __init__(
        self,
        categorical_groups: list[list[int]],
        total_features: int,
        hidden_dim: int = 32,
        init_std: float = 0.1,
    ) -> None:
        super().__init__()
        self.categorical_groups = categorical_groups
        self.total_features = total_features
        self.hidden_dim = hidden_dim

        self.cat_indices = sorted(
            {idx for group in categorical_groups for idx in group}
        )
        self.con_indices = [
            idx for idx in range(total_features) if idx not in self.cat_indices
        ]
        self.dividers: Optional[list[int]] = None

        n_cat_features = len(self.cat_indices)
        if n_cat_features > 0:
            self.noise_net = nn.Sequential(
                nn.Linear(n_cat_features, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.mu_head = nn.Linear(hidden_dim, n_cat_features)
            self.log_sigma_head = nn.Linear(hidden_dim, n_cat_features)
            nn.init.zeros_(self.mu_head.weight)
            nn.init.zeros_(self.mu_head.bias)
            nn.init.constant_(self.log_sigma_head.bias, np.log(init_std))
        else:
            self.noise_net = None
            self.mu_head = None
            self.log_sigma_head = None

    def fit(self, X: np.ndarray) -> "VariationalDequantizer":
        """Fit dividers for categorical features."""
        if self.cat_indices:
            self.dividers = []
            for idx in self.cat_indices:
                max_val = int(X[:, idx].max())
                self.dividers.append(max(max_val + 1, 2))
        return self

    def forward(
        self,
        x: torch.Tensor,
        return_log_prob: bool = True,
        use_mean: bool = False,
    ) -> DequantizerOutput:
        """Apply variational dequantization."""
        if not self.cat_indices:
            log_q = (
                torch.zeros(x.shape[0], device=x.device) if return_log_prob else None
            )
            return DequantizerOutput(values=x, log_q=log_q)

        x_cat = x[:, self.cat_indices]
        h = self.noise_net(x_cat)
        mu = self.mu_head(h)
        log_sigma = self.log_sigma_head(h)
        sigma = torch.exp(log_sigma)

        if use_mean:
            u_raw = mu
        else:
            eps = torch.randn_like(mu)
            u_raw = mu + sigma * eps
        u = torch.sigmoid(u_raw)

        log_q = None
        if return_log_prob:
            log_q_gaussian = -0.5 * (
                np.log(2 * np.pi) + 2 * log_sigma + ((u_raw - mu) / sigma).pow(2)
            )
            log_jacobian = torch.log(u * (1 - u) + 1e-8)
            log_q = (log_q_gaussian - log_jacobian).sum(dim=1)

        z_cat = x_cat.clone()
        if self.dividers is not None:
            for idx, divider in enumerate(self.dividers):
                z_cat[:, idx] = (x_cat[:, idx] + u[:, idx]) / divider
        else:
            z_cat = x_cat + u

        z_cat = self._logit_transform(z_cat)
        z = x.clone()
        z[:, self.cat_indices] = z_cat
        return DequantizerOutput(values=z, log_q=log_q)

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Inverse transform for categorical features."""
        if not self.cat_indices:
            return z

        x = z.clone()
        z_cat = z[:, self.cat_indices]
        z_cat = torch.sigmoid(z_cat)
        z_cat = (z_cat - ALPHA) / (1 - 2 * ALPHA)

        if self.dividers is not None:
            for idx, divider in enumerate(self.dividers):
                scaled = z_cat[:, idx] * divider
                x[:, self.cat_indices[idx]] = torch.clamp(
                    torch.round(scaled), 0, divider - 1
                )
        else:
            x[:, self.cat_indices] = torch.round(z_cat)

        return x

    @staticmethod
    def _logit_transform(x: torch.Tensor) -> torch.Tensor:
        x = ALPHA + (1 - 2 * ALPHA) * x
        return torch.log(x / (1.0 - x))


class VariationalGroupDequantizer(nn.Module):
    """Group wrapper for variational dequantizer."""

    def __init__(
        self,
        categorical_groups: list[list[int]],
        total_features: int,
        hidden_dim: int = 32,
        init_std: float = 0.1,
    ) -> None:
        super().__init__()
        self.dequantizer = VariationalDequantizer(
            categorical_groups=categorical_groups,
            total_features=total_features,
            hidden_dim=hidden_dim,
            init_std=init_std,
        )

    def fit(self, X: np.ndarray) -> "VariationalGroupDequantizer":
        """Fit dividers for categorical features."""
        self.dequantizer.fit(X)
        return self

    def forward(
        self,
        x: torch.Tensor,
        return_log_prob: bool = True,
        use_mean: bool = False,
    ) -> DequantizerOutput:
        """Apply variational dequantization."""
        return self.dequantizer(x, return_log_prob, use_mean)

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Inverse transform categorical features."""
        return self.dequantizer.inverse(z)
