from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm


def plot_x_point(
    x: torch.Tensor, x_origin: torch.Tensor, model: torch.nn.Module
) -> None:
    """Plot a point and its origin over model samples.

    Args:
        x: Counterfactual point.
        x_origin: Original point.
        model: Generative model with ``sample_and_log_prob``.
    """
    model.eval()
    fig, ax = plt.subplots()
    x_res = x.detach().numpy().squeeze()
    x_origin = x_origin.numpy().squeeze()
    dist = np.linalg.norm(x_res - x_origin)

    with torch.no_grad():
        samples_zero, _ = model.sample_and_log_prob(512, context=torch.Tensor([[0]]))
        samples_one, _ = model.sample_and_log_prob(512, context=torch.ones(1, 1))

    ax.set_title(f"{x.detach().numpy()}, dist: {dist}")
    ax.scatter(samples_zero.squeeze()[:, 0], samples_zero.squeeze()[:, 1], c="g")
    ax.scatter(samples_one.squeeze()[:, 0], samples_one.squeeze()[:, 1], c="b")
    ax.arrow(
        x_origin[0],
        x_origin[1],
        x_res[0] - x_origin[0],
        x_res[1] - x_origin[1],
        width=0.025,
        length_includes_head=True,
        color="C3",
    )
    ax.scatter(x_origin[0], x_origin[1], c="r")


def plot_model_distribution(
    model: torch.nn.Module,
    median_prob: float | None = None,
    disc_model: torch.nn.Module | None = None,
) -> plt.Axes:
    """Plot a model density surface with optional median-probability contour.

    Args:
        model: Generative model with ``log_prob``.
        median_prob: Optional median log-probability to highlight.
        disc_model: Unused discriminator handle kept for backward compatibility.

    Returns:
        Matplotlib axes for further customization.
    """
    _ = disc_model
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(8, 6)

    xline = torch.linspace(-1.5, 2.5, 200)
    yline = torch.linspace(-0.75, 1.25, 200)
    xgrid, ygrid = torch.meshgrid(xline, yline)
    xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)

    with torch.no_grad():
        zgrid = model.log_prob(xyinput, torch.zeros(40000, 1)).exp().reshape(
            200, 200
        ) + model.log_prob(xyinput, torch.ones(40000, 1)).exp().reshape(200, 200)

    zgrid = zgrid.numpy()

    cs = ax.contourf(xgrid.numpy(), ygrid.numpy(), zgrid, levels=50, cmap=cm.PuBu)
    fig.colorbar(cs)
    if median_prob is not None:
        median_prob = np.exp(median_prob)
        cs = ax.contourf(
            xgrid.numpy(),
            ygrid.numpy(),
            zgrid,
            levels=[median_prob - 0.01, median_prob + 0.01],
            colors="r",
        )

    return ax


def plot_loss_space(
    x: torch.Tensor,
    x_origin: torch.Tensor,
    optim_function: Callable[..., dict[str, torch.Tensor]],
    **optim_function_params: Any,
) -> None:
    """Plot the optimization loss surface for a counterfactual search.

    Args:
        x: Counterfactual point.
        x_origin: Original point.
        optim_function: Optimization function returning a ``loss`` entry.
        **optim_function_params: Extra parameters for the optimizer function.
    """
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(8, 5)

    xline = torch.linspace(-1.5, 2.5, 200)
    yline = torch.linspace(-0.75, 1.25, 200)
    xgrid, ygrid = torch.meshgrid(xline, yline)
    xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)

    x_res = x.detach().numpy().squeeze()
    context_origin = torch.zeros((xyinput.shape[0], 1))
    context_target = torch.ones((xyinput.shape[0], 1))

    with torch.no_grad():
        zgrid = optim_function(
            xyinput, x_origin, context_origin, context_target, **optim_function_params
        )
        zgrid = zgrid["loss"]
        zgrid = zgrid.log().reshape(200, 200).numpy()

    cs = ax.contourf(xgrid.numpy(), ygrid.numpy(), zgrid, levels=100, cmap=cm.PuBu_r)
    fig.colorbar(cs)
    ax.scatter(x_origin[0, 0], x_origin[0, 1], c="r")
    min_i = np.argmin(zgrid)

    ax.scatter(xyinput[min_i, 0], xyinput[min_i, 1], c="y")
    ax.arrow(
        x_origin[0, 0],
        x_origin[0, 1],
        x_res[0] - x_origin[0, 0],
        x_res[1] - x_origin[0, 1],
        width=0.025,
        length_includes_head=True,
        color="C3",
    )
    plt.show()


def plot_distributions(
    x: torch.Tensor,
    x_orig: torch.Tensor,
    model: torch.nn.Module,
    optim_function: Callable[[torch.Tensor, torch.Tensor, torch.nn.Module, float], Any],
    alpha: float,
) -> None:
    """Compare model density and optimization surface.

    Args:
        x: Counterfactual point.
        x_orig: Original point.
        model: Generative model with ``log_prob``.
        optim_function: Objective function for the loss plot.
        alpha: Weighting parameter for ``optim_function``.
    """
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(12, 5)

    xline = torch.linspace(-1.5, 2.5, 200)
    yline = torch.linspace(-0.75, 1.25, 200)
    xgrid, ygrid = torch.meshgrid(xline, yline)
    xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)

    x_res = x.detach().numpy().squeeze()

    with torch.no_grad():
        zgrid0 = -model.log_prob(xyinput, torch.ones(40000, 1)).exp().reshape(
            200, 200
        ) - model.log_prob(xyinput, torch.zeros(40000, 1)).exp().reshape(200, 200)

        zgrid1 = optim_function(xyinput, x_orig, model, alpha).reshape(200, 200)

    ax[0].contourf(
        xgrid.numpy(), ygrid.numpy(), zgrid0.numpy(), levels=20, cmap=cm.PuBu_r
    )
    cs = ax[1].contourf(xgrid.numpy(), ygrid.numpy(), zgrid1, levels=50, cmap=cm.PuBu_r)
    fig.colorbar(cs)
    ax[0].scatter(x_orig[0, 0], x_orig[0, 1], c="r")
    ax[1].scatter(x_orig[0, 0], x_orig[0, 1], c="r")

    ax[0].arrow(
        x_orig[0, 0],
        x_orig[0, 1],
        x_res[0] - x_orig[0, 0],
        x_res[1] - x_orig[0, 1],
        width=0.025,
        length_includes_head=True,
        color="C3",
    )
    ax[1].arrow(
        x_orig[0, 0],
        x_orig[0, 1],
        x_res[0] - x_orig[0, 0],
        x_res[1] - x_orig[0, 1],
        width=0.025,
        length_includes_head=True,
        color="C3",
    )
    log_p = model.log_prob(x, context=torch.Tensor([[1]])).exp().item()
    log_p_origin = model.log_prob(x_orig, context=torch.Tensor([[0]])).exp().item()
    ax[0].set_title(f"log_p_orig: {log_p_origin:.2e}")
    ax[1].set_title(
        f"Distance: {np.linalg.norm(x_orig - x_res):0.2f}, log_p: {log_p:.2e}"
    )
    plt.show()


def flatten_dict(
    dd: dict[str, Any], separator: str = "_", prefix: str = ""
) -> dict[str, Any]:
    """Flatten a nested dictionary.

    Args:
        dd: Nested dictionary to flatten.
        separator: Separator used to join nested keys.
        prefix: Prefix for nested keys.

    Returns:
        Flattened dictionary.
    """
    return (
        {
            prefix + separator + k if prefix else k: v
            for kk, vv in dd.items()
            for k, v in flatten_dict(vv, separator, kk).items()
        }
        if isinstance(dd, dict)
        else {prefix: dd}
    )


def add_prefix_to_dict(data: dict[str, Any], prefix: str) -> dict[str, Any]:
    """Prefix all keys in a dictionary.

    Args:
        data: Dictionary to update.
        prefix: Prefix to prepend to keys.

    Returns:
        Dictionary with prefixed keys.
    """
    return {f"{prefix}/{key}": value for key, value in data.items()}


def process_classification_report(
    report: dict[str, Any],
    prefix: str | None = None,
) -> dict[str, Any]:
    """Flatten and optionally prefix a classification report.

    Args:
        report: Classification report from sklearn.
        prefix: Optional prefix for the report keys.

    Returns:
        Flattened report with optional prefixing.
    """
    report = flatten_dict(report)
    if prefix:
        report = add_prefix_to_dict(report, prefix)
    return report
