"""Visualization helpers for DiCoFlex training and inference."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .context_utils import build_context_matrix

LOGGER = logging.getLogger(__name__)


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def visualize_training_batch(
    batch_cf: torch.Tensor,
    batch_context: torch.Tensor,
    feature_names: Sequence[str],
    save_path: Path,
    max_points: int | None = 200,
    dataset_points: np.ndarray | None = None,
) -> None:
    """Plot an entire training batch of factual points with their sampled neighbours."""
    if batch_cf.shape[1] < 2:
        LOGGER.info("Skipping training-batch visualization (need at least 2 features).")
        return

    factuals = batch_context[:, : batch_cf.shape[1]].cpu().numpy()
    neighbors = batch_cf.cpu().numpy()
    total = neighbors.shape[0]
    if max_points is not None and total > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(total, size=max_points, replace=False)
        neighbors = neighbors[idx]
        factuals = factuals[idx]

    plt.figure(figsize=(6, 5))
    if dataset_points is not None and dataset_points.shape[1] >= 2:
        plt.scatter(
            dataset_points[:, 0],
            dataset_points[:, 1],
            c="lightgray",
            alpha=0.2,
            label="Dataset",
            edgecolors="none",
        )
    plt.scatter(
        neighbors[:, 0],
        neighbors[:, 1],
        c="tab:blue",
        alpha=0.6,
        label="Neighbour samples",
        edgecolors="none",
    )
    plt.scatter(
        factuals[:, 0],
        factuals[:, 1],
        c="tab:orange",
        marker="*",
        s=60,
        label="Factual points",
        edgecolor="k",
    )
    plt.title("Training batch (factuals vs neighbours)")
    x_label = feature_names[0] if feature_names else "feature_0"
    y_label = feature_names[1] if len(feature_names) > 1 else "feature_1"
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.tight_layout()
    _ensure_dir(save_path)
    plt.savefig(save_path, dpi=200)
    plt.close()


def visualize_flow_contours(
    gen_model: torch.nn.Module,
    factual_point: np.ndarray,
    target_label: int,
    mask_vector: np.ndarray,
    p_value: float,
    class_to_index: dict[int, int],
    feature_bounds: Sequence[tuple[float, float]],
    save_path: Path,
    feature_names: Sequence[str] | None = None,
    grid_size: int = 200,
    device: str = "cpu",
    dataset_points: np.ndarray | None = None,
) -> None:
    """Plot contour lines of the trained flow's log-density around a factual point."""
    if factual_point.shape[0] < 2:
        LOGGER.info("Skipping contour visualization (need at least 2 features).")
        return

    (x_min, x_max), (y_min, y_max) = feature_bounds
    grid_x = np.linspace(x_min, x_max, grid_size)
    grid_y = np.linspace(y_min, y_max, grid_size)
    mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)

    repeated = np.repeat(factual_point[np.newaxis, :], grid_size**2, axis=0)
    repeated[:, 0] = mesh_x.ravel()
    repeated[:, 1] = mesh_y.ravel()
    contexts = build_context_matrix(
        factual_points=np.repeat(factual_point[np.newaxis, :], grid_size**2, axis=0),
        labels=np.full(grid_size**2, target_label),
        mask_vector=mask_vector,
        p_value=p_value,
        class_to_index=class_to_index,
    )
    with torch.no_grad():
        scores = (
            gen_model(
                torch.from_numpy(repeated).float().to(device),
                context=contexts.to(device),
            )
            .cpu()
            .numpy()
            .reshape(grid_size, grid_size)
        )
    scores = np.clip(scores, a_min=-10.0, a_max=None)

    plt.figure(figsize=(6, 5))
    contour = plt.contour(
        mesh_x,
        mesh_y,
        scores,
        levels=20,
        cmap="viridis",
    )
    plt.clabel(contour, inline=True, fontsize=8)
    plt.colorbar(contour, label="log p(x | factual)")
    if dataset_points is not None and dataset_points.shape[1] >= 2:
        plt.scatter(
            dataset_points[:, 0],
            dataset_points[:, 1],
            c="lightgray",
            alpha=0.2,
            label="Dataset",
            edgecolors="none",
        )
    plt.scatter(
        factual_point[0],
        factual_point[1],
        c="red",
        marker="*",
        s=120,
        edgecolor="k",
        label="Factual",
    )
    x_label = feature_names[0] if feature_names else "feature_0"
    y_label = feature_names[1] if feature_names and len(feature_names) > 1 else "feature_1"
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title("DiCoFlex flow log-density contour")
    plt.legend(loc="upper right")
    plt.tight_layout()
    _ensure_dir(save_path)
    plt.savefig(save_path, dpi=200)
    plt.close()


def visualize_counterfactual_samples(
    factual_points: np.ndarray,
    counterfactual_points: np.ndarray,
    feature_names: Sequence[str],
    save_path: Path,
    dataset_points: np.ndarray | None = None,
    max_points: int = 200,
) -> None:
    """Plot sampled counterfactuals alongside their factual anchors."""
    if factual_points.shape[1] < 2:
        LOGGER.info("Skipping counterfactual scatter (need at least 2 features).")
        return

    total_points = factual_points.shape[0]
    if total_points > max_points:
        idx = np.random.default_rng(0).choice(total_points, size=max_points, replace=False)
        factual_points = factual_points[idx]
        counterfactual_points = counterfactual_points[idx]

    plt.figure(figsize=(6, 5))
    if dataset_points is not None and dataset_points.shape[1] >= 2:
        plt.scatter(
            dataset_points[:, 0],
            dataset_points[:, 1],
            c="lightgray",
            alpha=0.2,
            label="Dataset",
            edgecolors="none",
        )
    plt.scatter(
        counterfactual_points[:, 0],
        counterfactual_points[:, 1],
        c="tab:green",
        alpha=0.7,
        label="Counterfactuals",
        edgecolors="none",
    )
    plt.scatter(
        factual_points[:, 0],
        factual_points[:, 1],
        c="tab:orange",
        marker="*",
        s=60,
        label="Factuals",
        edgecolor="k",
    )
    x_label = feature_names[0] if feature_names else "feature_0"
    y_label = feature_names[1] if len(feature_names) > 1 else "feature_1"
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title("Sampled counterfactuals")
    plt.legend()
    plt.tight_layout()
    _ensure_dir(save_path)
    plt.savefig(save_path, dpi=200)
    plt.close()
