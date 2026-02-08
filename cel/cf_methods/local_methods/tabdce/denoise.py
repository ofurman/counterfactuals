from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embedding used for diffusion time steps."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed diffusion steps into a sinusoidal representation."""
        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, device=t.device, dtype=torch.float32)
            * (-math.log(10000.0) / (half - 1))
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1), mode="constant", value=0.0)
        return emb


class FiLM(nn.Module):
    """Feature-wise linear modulation conditioning block."""

    def __init__(self, hdim: int, cdim: int) -> None:
        super().__init__()
        self.to_params = nn.Linear(cdim, hdim * 2)

    def forward(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Apply FiLM modulation using conditioning input."""
        params = self.to_params(c)
        gamma, beta = params.chunk(2, dim=-1)
        gamma = 1.0 + torch.tanh(gamma) * 0.1
        beta = torch.tanh(beta) * 0.1
        return h * gamma + beta


class ResBlock(nn.Module):
    """Residual block with FiLM conditioning."""

    def __init__(self, dim: int, cdim: int, hidden: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(dim)
        self.film = FiLM(hidden, cdim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Apply residual transform with conditioning."""
        res = x
        h = self.fc1(x)
        h = self.norm1(h)
        h = self.act(self.film(h, c))
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.norm2(h)
        return self.act(h + res)


class TabularEpsModel(nn.Module):
    """Conditional epsilon model for tabular diffusion."""

    def __init__(
        self,
        xdim: int,
        cat_dims: list[int],
        y_classes: int,
        hidden: int = 256,
        nblocks: int = 4,
        tdim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_numerical = xdim - sum(cat_dims)
        self.cat_dims = cat_dims

        self.inp_proj = nn.Linear(xdim, hidden)
        self.time_emb = SinusoidalTimeEmbedding(tdim)
        self.y_emb = nn.Embedding(y_classes, 32)

        self.cond_dim = xdim + 32 + tdim

        self.blocks = nn.ModuleList(
            [ResBlock(hidden, self.cond_dim, hidden, dropout) for _ in range(nblocks)]
        )

        self.out_proj = nn.Linear(hidden, xdim)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        x_orig: torch.Tensor,
        y_target: torch.Tensor,
    ) -> torch.Tensor:
        """Predict diffusion noise conditioned on input and target."""
        temb = self.time_emb(t)
        yemb = self.y_emb(y_target)
        cond = torch.cat([x_orig, yemb, temb], dim=-1)

        h = self.inp_proj(x_t)
        for blk in self.blocks:
            h = blk(h, cond)

        return self.out_proj(h)
