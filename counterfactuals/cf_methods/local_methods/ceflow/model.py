from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from nflows.flows import SimpleRealNVP as _SimpleRealNVP
from torch import nn


@dataclass(frozen=True)
class CeFlowConfig:
    """Configuration for the CeFlow generative model."""

    n_coupling_layers: int = 6
    hidden_dim: int = 256
    dequant_hidden_dim: int = 128


class DequantizationNetwork(nn.Module):
    """Variational dequantization network for categorical features."""

    def __init__(self, cat_cardinalities: list[int], hidden_dim: int = 128) -> None:
        super().__init__()
        self.cat_cardinalities = cat_cardinalities
        n_categorical = len(cat_cardinalities)
        if n_categorical == 0:
            raise ValueError("DequantizationNetwork requires at least one categorical feature.")

        embed_dim = max(1, hidden_dim // n_categorical)
        self.embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, embed_dim) for cardinality in cat_cardinalities]
        )

        input_dim = embed_dim * n_categorical
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, n_categorical)
        self.logvar_head = nn.Linear(hidden_dim, n_categorical)

    def forward(self, x_cat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        h = torch.cat(embeddings, dim=-1)
        h = self.network(h)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        return mu, logvar


class VariationalDequantizer(nn.Module):
    """Variational dequantizer for categorical inputs."""

    def __init__(self, cat_cardinalities: list[int], hidden_dim: int = 128) -> None:
        super().__init__()
        self.dequant_net = DequantizationNetwork(cat_cardinalities, hidden_dim=hidden_dim)

    def forward(self, x_cat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = self.dequant_net(x_cat)
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(mu)
        u = mu + std * epsilon
        u_bounded = torch.sigmoid(u)
        z_cat = x_cat.float() + u_bounded

        log_q_gaussian = -0.5 * (epsilon**2 + logvar + np.log(2 * np.pi))
        log_sigmoid_deriv = torch.log(u_bounded * (1 - u_bounded) + 1e-8)
        log_q = (log_q_gaussian - log_sigmoid_deriv).sum(dim=-1)
        return z_cat, log_q

    def dequantize_deterministic(self, x_cat: torch.Tensor) -> torch.Tensor:
        mu, _ = self.dequant_net(x_cat)
        u_bounded = torch.sigmoid(mu)
        return x_cat.float() + u_bounded


class GaussianMixturePrior(nn.Module):
    """Class-conditional Gaussian mixture prior."""

    def __init__(self, dim: int, n_classes: int) -> None:
        super().__init__()
        self.dim = dim
        self.n_classes = n_classes
        self.register_buffer("means", torch.randn(n_classes, dim))
        self.register_buffer("log_vars", torch.zeros(n_classes, dim))

    def log_prob(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mu = self.means[y]
        log_var = self.log_vars[y]
        log_p = -0.5 * (
            self.dim * np.log(2 * np.pi)
            + log_var.sum(dim=-1)
            + ((z - mu) ** 2 / torch.exp(log_var)).sum(dim=-1)
        )
        return log_p


class RealNVPFlow(nn.Module):
    """RealNVP flow wrapper exposing forward and inverse transforms."""

    def __init__(self, dim: int, n_coupling_layers: int, hidden_dim: int) -> None:
        super().__init__()
        self.model = _SimpleRealNVP(
            features=dim,
            hidden_features=hidden_dim,
            num_layers=n_coupling_layers,
            num_blocks_per_layer=2,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z, log_det = self.model._transform.forward(x, context=None)
        return z, log_det

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        x, _ = self.model._transform.inverse(z, context=None)
        return x


class CeFlowModel(nn.Module):
    """CeFlow model combining dequantization, flow, and Gaussian mixture prior."""

    def __init__(
        self,
        n_continuous: int,
        n_categorical: int,
        cat_cardinalities: list[int],
        n_classes: int,
        config: CeFlowConfig,
    ) -> None:
        super().__init__()
        self.n_continuous = n_continuous
        self.n_categorical = n_categorical
        self.total_dim = n_continuous + n_categorical

        if n_categorical > 0:
            self.dequantizer: VariationalDequantizer | None = VariationalDequantizer(
                cat_cardinalities, hidden_dim=config.dequant_hidden_dim
            )
        else:
            self.dequantizer = None

        self.flow = RealNVPFlow(
            dim=self.total_dim,
            n_coupling_layers=config.n_coupling_layers,
            hidden_dim=config.hidden_dim,
        )
        self.prior = GaussianMixturePrior(self.total_dim, n_classes)
        self.class_means: dict[int, torch.Tensor] | None = None

    def forward(
        self, x_con: torch.Tensor, x_cat: torch.Tensor | None, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.dequantizer is not None and x_cat is not None:
            z_cat, log_q_cat = self.dequantizer(x_cat)
            x_full = torch.cat([x_con, z_cat], dim=-1)
        else:
            x_full = x_con
            log_q_cat = 0.0

        z, log_det = self.flow.forward(x_full)
        log_p_z = self.prior.log_prob(z, y)
        log_p_x = log_p_z + log_det - log_q_cat
        return log_p_x, z

    def encode(self, x_con: torch.Tensor, x_cat: torch.Tensor | None) -> torch.Tensor:
        if self.dequantizer is not None and x_cat is not None:
            z_cat = self.dequantizer.dequantize_deterministic(x_cat)
            x_full = torch.cat([x_con, z_cat], dim=-1)
        else:
            x_full = x_con
        z, _ = self.flow.forward(x_full)
        return z

    def decode(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        x_full = self.flow.inverse(z)
        if self.n_categorical > 0:
            x_con = x_full[:, : self.n_continuous]
            x_cat = x_full[:, self.n_continuous :]
            return x_con, x_cat
        return x_full, None
