import logging
import math
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.inspection import DecisionBoundaryDisplay

logger = logging.getLogger(__name__)


def plot_generative_model_distribution(ax, model, prob_threshold, num_classes):
    xline = torch.linspace(-0, 1, 200)
    yline = torch.linspace(-0, 1, 200)
    xgrid, ygrid = torch.meshgrid(xline, yline)
    xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)

    for i in range(num_classes):
        with torch.no_grad():
            zgrid = model(xyinput, i * torch.ones(40000, 1)).exp().reshape(200, 200)
            zgrid = zgrid.numpy()
            _ = ax.contour(
                xgrid.numpy(),
                ygrid.numpy(),
                zgrid,
                levels=10,
                cmap="Greys",
                linewidths=0.4,
                antialiased=True,
            )

        if prob_threshold is not None:
            prob_threshold_exp = np.exp(prob_threshold)
            _ = ax.contourf(
                xgrid.numpy(),
                ygrid.numpy(),
                zgrid,
                levels=[prob_threshold_exp, prob_threshold_exp * 10.00],
                alpha=0.1,
                colors="#DC143C",
            )  # 10.00 is an arbitrary huge value to colour the whole distribution.

    return ax


def plot_classifier_decision_region(ax, model):
    xline = torch.linspace(-0, 1, 400)
    yline = torch.linspace(-0, 1, 400)
    xgrid, ygrid = torch.meshgrid(xline, yline)
    xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)

    y_hat = model.predict(xyinput)
    y_hat = y_hat.reshape(400, 400)

    # ax.contour(xgrid.numpy(), ygrid.numpy(), y_hat.numpy(), alpha=0.1, cmap="tab10", levels=1)

    display = DecisionBoundaryDisplay(xx0=xgrid, xx1=ygrid, response=y_hat)
    ax = display.plot(plot_method="contour", ax=ax, alpha=0.3).ax_
    return ax


def plot_counterfactuals(ax, counterfactuals):
    ax.scatter(
        counterfactuals[:, 0], counterfactuals[:, 1], c="orange", s=50, alpha=0.8
    )
    return ax


def plot_observations(ax, observations, targets, colors=None):
    # colors is a list of colors for each observation
    ax.scatter(
        observations[:, 0],
        observations[:, 1],
        c=colors if colors is not None else targets,
        cmap=matplotlib.colormaps["tab10"],
        s=50,
        alpha=0.8,
    )
    return ax


def plot_arrows(ax, observations, counterfactuals):
    for i in range(len(observations)):
        ax.arrow(
            observations[i, 0],
            observations[i, 1],
            counterfactuals[i, 0] - observations[i, 0],
            counterfactuals[i, 1] - observations[i, 1],
            width=0.001,
            lw=0.001,
            length_includes_head=True,
            alpha=0.5,
            color="k",
        )
    return ax


def plot_metrics_textbox(ax, metrics_series):
    text_str = "\n".join(
        f"{metric}: {value:.3f}" for metric, value in metrics_series.items()
    )

    props = dict(boxstyle="round", facecolor="wheat", alpha=0.7)
    ax.text(
        0.05,
        0.95,
        text_str,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=props,
    )
    return ax


def plot(flow, disc_model, X_test, y_test, X_cf, metrics, prob_threshold):
    assert X_test.shape == X_cf.shape, (
        f"Sizes of test set and counterfactuals are not equal. Actual sizes: X_test: {X_test.shape}, X_cf: {X_cf.shape}"
    )
    assert y_test.shape[0] == X_cf.shape[0], (
        f"Sizes of targets and counterfactuals are not equal. Actual sizes: X_cf: {X_cf.shape}, y_test: {y_test.shape}"
    )

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(20, 12)

    ax = plot_classifier_decision_region(ax, disc_model)
    ax = plot_generative_model_distribution(
        ax, flow, prob_threshold=prob_threshold, num_classes=len(np.unique(y_test))
    )
    ax = plot_observations(ax, X_test, y_test)
    ax = plot_counterfactuals(ax, X_cf)
    ax = plot_arrows(ax, X_test, X_cf)
    ax = plot_metrics_textbox(ax, metrics)
    return ax


def create_grid_image(folders, output_filename, classifier):
    """Creates a grid image combining images from multiple folders, with automatic grid calculation.

    Args:
        folders (list): A list of paths to folders containing images.
        output_filename (str): Name of the output image file.
    """

    images = []
    for folder in folders:
        for filename in os.listdir(folder):
            if filename.endswith(f"{classifier}.png"):
                images.append(Image.open(os.path.join(folder, filename)))

    # Calculate grid dimensions
    num_images = len(images)
    rows = math.ceil(math.sqrt(num_images))
    cols = math.ceil(num_images / rows)

    # Calculate image dimensions (assuming all have reasonably similar size)
    image_width, image_height = images[0].size

    grid_width = image_width * cols
    grid_height = image_height * rows
    grid_image = Image.new("RGB", (grid_width, grid_height))

    index = 0
    for row in range(rows):
        for col in range(cols):
            if index < num_images:  # Prevent going out of bounds
                grid_image.paste(images[index], (col * image_width, row * image_height))
            index += 1

    grid_image.save(output_filename)


def plot_3d_regression_cfs(
    gen_model: torch.nn.Module,
    X_cfs: np.ndarray,
    X_origs: np.ndarray,
    y_origs: np.ndarray,
    y_targets: np.ndarray,
    save_path: str,
    delta: float,
    num_points: int = 10,
) -> None:
    """
    Create a 3D plot showing regression counterfactuals.

    Visualizes the original points, counterfactual points, and projections
    onto two vertical planes, with high density region on the bottom plane.
    Only works when data has exactly 2 features.

    Args:
        gen_model: Trained generative model
        X_cfs: Generated counterfactuals
        X_origs: Original instances
        y_origs: Original target values
        y_targets: Target values for counterfactuals
        save_path: Path to save the plot
        delta: Log probability threshold for high density region
        num_points: Number of points to subsample and plot
    """
    if X_cfs.shape[1] != 2:
        logger.info(
            f"Skipping 3D plot: data has {X_cfs.shape[1]} features (requires exactly 2)"
        )
        return

    logger.info("Creating 3D regression counterfactual plot with %d points", num_points)

    # Filter points with original target in range [0.3, 0.35]
    target_range_mask = (y_origs.ravel() >= 0.39) & (y_origs.ravel() <= 0.41)
    X_origs_filtered = X_origs[target_range_mask]
    y_origs_filtered = y_origs[target_range_mask]

    # If no points in range, fall back to all points
    if len(X_origs_filtered) == 0:
        logger.warning("No points found in target range [0.3, 0.35], using all points")
        X_origs_filtered = X_origs
        y_origs_filtered = y_origs
        filtered_indices = np.arange(len(X_origs))
    else:
        filtered_indices = np.where(target_range_mask)[0]

    # Sample points with low plausibility from the filtered set
    n_samples = min(num_points, len(X_origs_filtered))
    with torch.no_grad():
        X_origs_torch = torch.tensor(X_origs_filtered).float()
        y_origs_torch = torch.tensor(y_origs_filtered).float()
        log_probs = gen_model(X_origs_torch, y_origs_torch)
    low_prob_local_indices = np.argpartition(log_probs.numpy().ravel(), n_samples)[:n_samples]
    low_prob_indices = filtered_indices[low_prob_local_indices]

    X_cfs_sub = X_cfs[low_prob_indices]
    X_origs_sub = X_origs[low_prob_indices]
    y_origs_sub = y_origs[low_prob_indices]
    y_targets_sub = y_targets[low_prob_indices]

    # Calculate z-axis limits based on point values
    z_min = min(y_origs_sub.min(), y_targets_sub.min())
    z_max = 0.6

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Plot high density region on bottom plane
    xline = torch.linspace(0, 1, 300)
    yline = torch.linspace(0, 1, 300)
    xgrid, ygrid = torch.meshgrid(xline, yline)
    xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)

    # Use target value 0.6 as context for the generative model
    with torch.no_grad():
        zgrid = gen_model(
            xyinput, 0.6 * torch.ones(len(xyinput), 1)
        ).exp().reshape(300, 300)
        zgrid = zgrid.numpy()

    # Plot high density region as heatmap on z_min plane
    ax.plot_surface(
        xgrid.numpy(),
        ygrid.numpy(),
        np.full_like(zgrid, z_min),
        facecolors=plt.cm.Greens(zgrid / zgrid.max()),
        shade=False,
        alpha=0.3,
        rasterized=True,
        antialiased=False,
        linewidth=0,
    )

    # Plot original points
    ax.scatter(
        X_origs_sub[:, 0],
        X_origs_sub[:, 1],
        y_origs_sub.ravel(),
        c="blue",
        marker="o",
        s=50,
        alpha=0.7,
        label="Original points",
        depthshade=False,
        zorder=10,
    )

    # Plot counterfactual points
    ax.scatter(
        X_cfs_sub[:, 0],
        X_cfs_sub[:, 1],
        y_targets_sub.ravel(),
        c="red",
        marker="^",
        s=80,
        alpha=0.9,
        label="Counterfactuals",
        depthshade=False,
        zorder=10,
    )

    # Add projections onto feature2=1.0 plane
    for i in range(n_samples):
        # Projection from original point
        ax.plot(
            [X_origs_sub[i, 0], X_origs_sub[i, 0]],
            [X_origs_sub[i, 1], 1.0],
            [float(y_origs_sub.ravel()[i]), float(y_origs_sub.ravel()[i])],
            color="blue",
            linestyle="--",
            alpha=0.3,
            linewidth=1,
            zorder=5,
        )
        # Projection from counterfactual point
        ax.plot(
            [X_cfs_sub[i, 0], X_cfs_sub[i, 0]],
            [X_cfs_sub[i, 1], 1.0],
            [float(y_targets_sub.ravel()[i]), float(y_targets_sub.ravel()[i])],
            color="red",
            linestyle="--",
            alpha=0.3,
            linewidth=1,
            zorder=5,
        )

    # Add projections onto feature1=0.0 plane
    for i in range(n_samples):
        # Projection from original point
        ax.plot(
            [X_origs_sub[i, 0], 0.0],
            [X_origs_sub[i, 1], X_origs_sub[i, 1]],
            [float(y_origs_sub.ravel()[i]), float(y_origs_sub.ravel()[i])],
            color="blue",
            linestyle="--",
            alpha=0.3,
            linewidth=1,
            zorder=5,
        )
        # Projection from counterfactual point
        ax.plot(
            [X_cfs_sub[i, 0], 0.0],
            [X_cfs_sub[i, 1], X_cfs_sub[i, 1]],
            [float(y_targets_sub.ravel()[i]), float(y_targets_sub.ravel()[i])],
            color="red",
            linestyle="--",
            alpha=0.3,
            linewidth=1,
            zorder=5,
        )

    # Add projections onto feature1-feature2 plane (z=z_min) for counterfactuals
    for i in range(n_samples):
        ax.plot(
            [X_cfs_sub[i, 0], X_cfs_sub[i, 0]],
            [X_cfs_sub[i, 1], X_cfs_sub[i, 1]],
            [float(y_targets_sub.ravel()[i]), z_min],
            color="red",
            linestyle="--",
            alpha=0.3,
            linewidth=1,
            zorder=5,
        )

    # Plot projection points on feature2=1.0 plane
    ax.scatter(
        X_origs_sub[:, 0],
        np.full(n_samples, 1.0),
        y_origs_sub.ravel(),
        c="blue",
        marker="o",
        s=30,
        alpha=0.4,
        facecolors="none",
        edgecolors="blue",
        linewidth=1,
        zorder=5,
    )
    ax.scatter(
        X_cfs_sub[:, 0],
        np.full(n_samples, 1.0),
        y_targets_sub.ravel(),
        c="red",
        marker="^",
        s=40,
        alpha=0.4,
        facecolors="none",
        edgecolors="red",
        linewidth=1,
        zorder=5,
    )

    # Plot projection points on feature1=0.0 plane
    ax.scatter(
        np.full(n_samples, 0.0),
        X_origs_sub[:, 1],
        y_origs_sub.ravel(),
        c="blue",
        marker="o",
        s=30,
        alpha=0.4,
        facecolors="none",
        edgecolors="blue",
        linewidth=1,
        zorder=5,
    )
    ax.scatter(
        np.full(n_samples, 0.0),
        X_cfs_sub[:, 1],
        y_targets_sub.ravel(),
        c="red",
        marker="^",
        s=40,
        alpha=0.4,
        facecolors="none",
        edgecolors="red",
        linewidth=1,
        zorder=5,
    )

    # Plot projection points on feature1-feature2 plane (z=z_min) for counterfactuals
    ax.scatter(
        X_cfs_sub[:, 0],
        X_cfs_sub[:, 1],
        np.full(n_samples, z_min),
        c="red",
        marker="^",
        s=40,
        alpha=0.4,
        facecolors="none",
        edgecolors="red",
        linewidth=1,
        zorder=5,
    )

    ax.set_xlabel("Feature 1", fontsize=12)
    ax.set_ylabel("Feature 2", fontsize=12)
    ax.set_zlabel("Target", fontsize=12)
    ax.set_zlim(z_min, z_max)
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_title("Regression Counterfactuals with Decision Surface", fontsize=14)
    ax.legend(loc="upper left", fontsize=10)

    # Ensure surface is rendered behind other elements
    if ax.collections:
        ax.collections[0].set_zorder(0)
        for coll in ax.collections[1:]:
            coll.set_zorder(10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"3D plot saved to {save_path}")


def plot_conditional_density_contours(
    gen_model: torch.nn.Module,
    X_cfs: np.ndarray,
    X_origs: np.ndarray,
    y_origs: np.ndarray,
    y_targets: np.ndarray,
    save_path: str,
    target_values: list[float] = None,
    delta: float = None,
    num_points: int = 10,
) -> None:
    """
    Create a 2D plot showing conditional density contours for multiple target values.

    Visualizes p(x|y) as contour lines for different target values, showing how the
    high-density regions shift across feature space. Original and counterfactual
    points are overlaid to show the transformation.

    Args:
        gen_model: Trained generative model
        X_cfs: Generated counterfactuals
        X_origs: Original instances
        y_origs: Original target values
        y_targets: Target values for counterfactuals
        save_path: Path to save the plot
        target_values: List of target values to plot contours for
        delta: Log probability threshold for highlighting high-density regions
        num_points: Number of points to subsample and plot
    """
    if X_cfs.shape[1] != 2:
        logger.info(
            f"Skipping contour plot: data has {X_cfs.shape[1]} features (requires exactly 2)"
        )
        return

    if target_values is None:
        target_values = [0.4, 0.6]

    # Use same sampling logic as 3D plot
    target_range_mask = (y_origs.ravel() >= 0.39) & (y_origs.ravel() <= 0.41)
    X_origs_filtered = X_origs[target_range_mask]
    y_origs_filtered = y_origs[target_range_mask]

    if len(X_origs_filtered) == 0:
        X_origs_filtered = X_origs
        y_origs_filtered = y_origs
        filtered_indices = np.arange(len(X_origs))
    else:
        filtered_indices = np.where(target_range_mask)[0]

    n_samples = min(num_points, len(X_origs_filtered))
    with torch.no_grad():
        X_origs_torch = torch.tensor(X_origs_filtered).float()
        y_origs_torch = torch.tensor(y_origs_filtered).float()
        log_probs = gen_model(X_origs_torch, y_origs_torch)
    low_prob_local_indices = np.argpartition(log_probs.numpy().ravel(), n_samples)[:n_samples]
    low_prob_indices = filtered_indices[low_prob_local_indices]

    X_cfs_sub = X_cfs[low_prob_indices]
    X_origs_sub = X_origs[low_prob_indices]
    y_origs_sub = y_origs[low_prob_indices]
    y_targets_sub = y_targets[low_prob_indices]

    logger.info("Creating conditional density contour plot with target values: %s", target_values)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Create grid for contour plotting
    xline = torch.linspace(0, 1, 200)
    yline = torch.linspace(0, 1, 200)
    xgrid, ygrid = torch.meshgrid(xline, yline)
    xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)

    # Define colors for each target value
    colors = plt.cm.tab10(np.linspace(0, 1, len(target_values)))

    # Plot contours for each target value
    contour_handles = []
    for idx, target_val in enumerate(target_values):
        with torch.no_grad():
            zgrid = gen_model(
                xyinput, target_val * torch.ones(len(xyinput), 1)
            ).exp().reshape(200, 200)
            zgrid = zgrid.numpy()

        # Clip very large values to avoid oversized contours
        z_clipped = np.clip(zgrid, a_min=0, a_max=np.percentile(zgrid, 99.9))

        # Plot filled contour for high-density region (above delta if provided)
        if delta is not None:
            delta_exp = np.exp(delta)
            ax.contourf(
                xgrid.numpy(),
                ygrid.numpy(),
                z_clipped,
                levels=[delta_exp, z_clipped.max()],
                alpha=0.1,
                colors=[colors[idx]],
            )

        # Plot contour lines with reasonable levels
        contour_levels = np.linspace(
            delta_exp if delta is not None else z_clipped.min(),
            z_clipped.max(),
            8
        )
        cs = ax.contour(
            xgrid.numpy(),
            ygrid.numpy(),
            z_clipped,
            levels=contour_levels,
            alpha=0.6,
            colors=[colors[idx]],
            linewidths=1,
        )
        # Save contour for legend
        contour_handles.append(cs)

    # Create legend handles for contours
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=colors[i], linewidth=2, label=f"p(x|y={target_values[i]:.1f})")
        for i in range(len(target_values))
    ]
    legend_handles.extend([
        Line2D([0], [0], marker="o", color="blue", markerfacecolor="blue", markersize=8, label="Original points", linestyle="None"),
        Line2D([0], [0], marker="^", color="red", markerfacecolor="red", markersize=10, label="Counterfactuals", linestyle="None"),
    ])

    # Plot original points
    ax.scatter(
        X_origs_sub[:, 0],
        X_origs_sub[:, 1],
        c="blue",
        marker="o",
        s=50,
        alpha=0.7,
        edgecolors="black",
        linewidth=0.5,
    )

    # Plot counterfactual points
    ax.scatter(
        X_cfs_sub[:, 0],
        X_cfs_sub[:, 1],
        c="red",
        marker="^",
        s=80,
        alpha=0.9,
        edgecolors="black",
        linewidth=0.5,
    )

    # Plot arrows from originals to counterfactuals
    for i in range(n_samples):
        ax.annotate(
            "",
            xy=(X_cfs_sub[i, 0], X_cfs_sub[i, 1]),
            xytext=(X_origs_sub[i, 0], X_origs_sub[i, 1]),
            arrowprops=dict(
                arrowstyle="->",
                color="gray",
                alpha=0.5,
                lw=1,
            ),
        )

    ax.set_xlabel("Feature 1", fontsize=12)
    ax.set_ylabel("Feature 2", fontsize=12)
    ax.set_xlim(0.2, 0.8)
    ax.set_ylim(0.2, 0.8)
    ax.set_title("Conditional Density Contours p(x|y) with Counterfactuals", fontsize=14)
    ax.legend(handles=legend_handles, loc="upper right", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Contour plot saved to {save_path}")


def plot_plausibility_comparison(
    gen_model: torch.nn.Module,
    X_cfs: np.ndarray,
    X_origs: np.ndarray,
    y_origs: np.ndarray,
    y_targets: np.ndarray,
    save_path: str,
    delta: float,
) -> None:
    """
    Create a scatter plot comparing plausibility of original vs counterfactual points.

    Visualizes p(x|y) vs p(x_cf|y_cf) with threshold lines and valid region shading.
    Shows whether counterfactuals remain plausible for their new target values.

    Args:
        gen_model: Trained generative model
        X_cfs: Generated counterfactuals
        X_origs: Original instances
        y_origs: Original target values
        y_targets: Target values for counterfactuals
        save_path: Path to save the plot
        delta: Log probability threshold for validity
    """
    logger.info("Creating plausibility comparison plot")

    # Calculate log probabilities for original points
    with torch.no_grad():
        X_origs_torch = torch.tensor(X_origs).float()
        y_origs_torch = torch.tensor(y_origs).float()
        log_probs_orig = gen_model(X_origs_torch, y_origs_torch)

    # Calculate log probabilities for counterfactuals
    with torch.no_grad():
        X_cfs_torch = torch.tensor(X_cfs).float()
        y_targets_torch = torch.tensor(y_targets).float()
        log_probs_cf = gen_model(X_cfs_torch, y_targets_torch)

    # Use log probabilities directly for plotting
    log_probs_orig_plot = log_probs_orig.numpy().ravel()
    log_probs_cf_plot = log_probs_cf.numpy().ravel()

    # Log probability statistics for debugging
    logger.info(f"Original log_probs - min: {log_probs_orig_plot.min():.2f}, max: {log_probs_orig_plot.max():.2f}")
    logger.info(f"CF log_probs - min: {log_probs_cf_plot.min():.2f}, max: {log_probs_cf_plot.max():.2f}")

    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot diagonal line (equal plausibility)
    min_log = min(log_probs_orig_plot.min(), log_probs_cf_plot.min())
    max_log = max(log_probs_orig_plot.max(), log_probs_cf_plot.max())
    ax.plot(
        [min_log, max_log],
        [min_log, max_log],
        "k--",
        alpha=0.5,
        linewidth=1,
        label="Equal plausibility",
    )

    # Plot threshold lines
    ax.axvline(
        delta,
        color="gray",
        linestyle="--",
        alpha=0.3,
        linewidth=1,
    )
    ax.axhline(
        delta,
        color="gray",
        linestyle="--",
        alpha=0.3,
        linewidth=1,
    )

    # Plot points colored by counterfactual plausibility
    scatter = ax.scatter(
        log_probs_orig_plot,
        log_probs_cf_plot,
        c=log_probs_cf_plot,
        cmap="viridis",
        marker="o",
        s=30,
        alpha=0.7,
        edgecolors="black",
        linewidth=0.5,
    )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("CF log-probability log p(x_cf|y_cf)", fontsize=11)

    ax.set_xlabel("log p(x|y) - Original log-probability", fontsize=12)
    ax.set_ylabel("log p(x_cf|y_cf) - CF log-probability", fontsize=12)
    ax.set_title("Plausibility Comparison: Original vs Counterfactual", fontsize=14)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Plausibility comparison plot saved to {save_path}")
