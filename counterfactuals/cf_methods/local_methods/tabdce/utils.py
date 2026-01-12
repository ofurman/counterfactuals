from __future__ import annotations

from dataclasses import dataclass

import torch


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    """Extract coefficients for a batch of timesteps and reshape for broadcasting."""
    out = a.gather(0, t)
    return out.view(-1, *([1] * (len(x_shape) - 1))).expand(x_shape)


@dataclass
class DiffusionSchedule:
    """Container for diffusion schedule coefficients."""

    T: int
    betas: torch.Tensor

    @staticmethod
    def from_name(
        name: str, T: int, device: torch.device, dtype: torch.dtype = torch.float32
    ) -> "DiffusionSchedule":
        """Create diffusion schedule coefficients for a named schedule."""
        if name == "linear":
            scale = 1000.0 / T
            beta_start = scale * 1e-4
            beta_end = scale * 2e-2
            betas = torch.linspace(beta_start, beta_end, T, device=device, dtype=dtype)
        elif name == "cosine":
            s = 0.008
            steps = torch.arange(T + 1, device=device, dtype=dtype)
            f = torch.cos(((steps / T + s) / (1 + s)) * torch.pi / 2) ** 2
            alphas_bar = f / f[0]
            betas = 1 - (alphas_bar[1:] / alphas_bar[:-1]).clamp(min=1e-5, max=0.9999)
        else:
            raise ValueError(f"Unknown schedule: {name}")
        return DiffusionSchedule(T=T, betas=betas)
