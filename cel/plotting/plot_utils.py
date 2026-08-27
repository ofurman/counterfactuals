from __future__ import annotations

import math
import os

import matplotlib
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.inspection import DecisionBoundaryDisplay


def plot_generative_model_distribution(
    ax: matplotlib.axes.Axes,
    model: torch.nn.Module,
    prob_threshold: float | None,
    num_classes: int,
) -> matplotlib.axes.Axes:
    """Plot the generative model's log-probability distribution as contours.

    Evaluates the model on a 200×200 grid over [0, 1]² for each class and
    draws contour lines. If ``prob_threshold`` is provided, fills the region
    above the threshold in red to indicate plausible counterfactual space.

    Args:
        ax: Matplotlib axes to draw on.
        model: Generative model with a ``forward(x, class_idx)`` interface.
        prob_threshold: Log-probability threshold for the plausibility fill.
            Pass ``None`` to skip the filled contour.
        num_classes: Number of classes to plot distributions for.

    Returns:
        The axes with contours added.
    """
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


def plot_classifier_decision_region(
    ax: matplotlib.axes.Axes,
    model: torch.nn.Module,
) -> matplotlib.axes.Axes:
    """Plot the classifier decision boundary over [0, 1]².

    Evaluates the model on a 400×400 grid and draws the decision boundary
    using :class:`~sklearn.inspection.DecisionBoundaryDisplay`.

    Args:
        ax: Matplotlib axes to draw on.
        model: Classifier with a ``predict(x)`` method returning class labels.

    Returns:
        The axes with the decision boundary added.
    """
    xline = torch.linspace(-0, 1, 400)
    yline = torch.linspace(-0, 1, 400)
    xgrid, ygrid = torch.meshgrid(xline, yline)
    xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)

    y_hat = model.predict(xyinput)
    y_hat = y_hat.reshape(400, 400)

    display = DecisionBoundaryDisplay(xx0=xgrid, xx1=ygrid, response=y_hat)
    ax = display.plot(plot_method="contour", ax=ax, alpha=0.3).ax_
    return ax


def plot_counterfactuals(
    ax: matplotlib.axes.Axes,
    counterfactuals: np.ndarray,
) -> matplotlib.axes.Axes:
    """Scatter-plot counterfactual examples on the axes.

    Args:
        ax: Matplotlib axes to draw on.
        counterfactuals: Array of shape ``(n, 2)`` with counterfactual coordinates.

    Returns:
        The axes with counterfactuals plotted in orange.
    """
    ax.scatter(counterfactuals[:, 0], counterfactuals[:, 1], c="orange", s=50, alpha=0.8)
    return ax


def plot_observations(
    ax: matplotlib.axes.Axes,
    observations: np.ndarray,
    targets: np.ndarray,
    colors: list | None = None,
) -> matplotlib.axes.Axes:
    """Scatter-plot observations coloured by class label.

    Args:
        ax: Matplotlib axes to draw on.
        observations: Array of shape ``(n, 2)`` with observation coordinates.
        targets: Array of shape ``(n,)`` with class labels used for colouring
            when ``colors`` is ``None``.
        colors: Optional list of per-observation colour values. When provided,
            overrides ``targets`` for colouring.

    Returns:
        The axes with observations plotted.
    """
    ax.scatter(
        observations[:, 0],
        observations[:, 1],
        c=colors if colors is not None else targets,
        cmap=matplotlib.colormaps["tab10"],
        s=50,
        alpha=0.8,
    )
    return ax


def plot_arrows(
    ax: matplotlib.axes.Axes,
    observations: np.ndarray,
    counterfactuals: np.ndarray,
) -> matplotlib.axes.Axes:
    """Draw arrows from each observation to its counterfactual.

    Args:
        ax: Matplotlib axes to draw on.
        observations: Array of shape ``(n, 2)`` with original coordinates.
        counterfactuals: Array of shape ``(n, 2)`` with counterfactual coordinates.

    Returns:
        The axes with arrows added.
    """
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


def plot_metrics_textbox(
    ax: matplotlib.axes.Axes,
    metrics_series: pd.Series,
) -> matplotlib.axes.Axes:
    """Render a metrics summary text box in the top-left corner of the axes.

    Args:
        ax: Matplotlib axes to draw on.
        metrics_series: Series mapping metric name to scalar value.

    Returns:
        The axes with the text box added.
    """
    text_str = "\n".join(f"{metric}: {value:.3f}" for metric, value in metrics_series.items())

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


def plot(
    flow: torch.nn.Module,
    disc_model: torch.nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_cf: np.ndarray,
    metrics: pd.Series,
    prob_threshold: float | None,
) -> matplotlib.axes.Axes:
    """Produce a composite visualisation of the CF experiment.

    Renders the classifier decision boundary, generative model distribution,
    original observations, generated counterfactuals, transition arrows, and
    a metrics text box on a single axes.

    Args:
        flow: Generative flow model passed to
            :func:`plot_generative_model_distribution`.
        disc_model: Discriminative classifier passed to
            :func:`plot_classifier_decision_region`.
        X_test: Test observations, shape ``(n, 2)``.
        y_test: Test labels, shape ``(n,)``.
        X_cf: Generated counterfactuals, shape ``(n, 2)``.
        metrics: Series of evaluation metrics to display.
        prob_threshold: Log-probability threshold for the plausibility region,
            or ``None`` to omit it.

    Returns:
        The populated matplotlib axes.
    """
    assert X_test.shape == X_cf.shape, (
        f"Sizes of test set and counterfactuals are not equal. "
        f"Actual sizes: X_test: {X_test.shape}, X_cf: {X_cf.shape}"
    )
    assert y_test.shape[0] == X_cf.shape[0], (
        f"Sizes of targets and counterfactuals are not equal. "
        f"Actual sizes: X_cf: {X_cf.shape}, y_test: {y_test.shape}"
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


def create_grid_image(
    folders: list[str],
    output_filename: str,
    classifier: str,
) -> None:
    """Combine per-folder PNG images into a single grid image.

    Scans each folder for files ending in ``f"{classifier}.png"``, arranges
    them in a near-square grid, and saves the result.

    Args:
        folders: Paths to folders containing the source images.
        output_filename: File path for the output grid image.
        classifier: Suffix used to select images within each folder
            (e.g. ``"MLP"`` matches ``"*MLP.png"``).
    """
    images = []
    for folder in folders:
        for filename in os.listdir(folder):
            if filename.endswith(f"{classifier}.png"):
                images.append(Image.open(os.path.join(folder, filename)))

    num_images = len(images)
    rows = math.ceil(math.sqrt(num_images))
    cols = math.ceil(num_images / rows)

    image_width, image_height = images[0].size

    grid_width = image_width * cols
    grid_height = image_height * rows
    grid_image = Image.new("RGB", (grid_width, grid_height))

    index = 0
    for row in range(rows):
        for col in range(cols):
            if index < num_images:
                grid_image.paste(images[index], (col * image_width, row * image_height))
            index += 1

    grid_image.save(output_filename)
