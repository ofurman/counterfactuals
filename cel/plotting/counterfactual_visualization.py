import matplotlib
import matplotlib.pyplot as plt

from .plot_utils import (
    plot_classifier_decision_region,
    plot_generative_model_distribution,
)

GROUP_COLORS = [
    "blue",
    "red",
    "green",
    "orange",
    "purple",
    "brown",
    "pink",
    "gray",
    "olive",
    "cyan",
]


def plot_counterfactuals(Xs, Xs_cfs, log_prob_threshold, disc_model, gen_model=None):
    fig, ax = plt.subplots(1, 1)

    ax.scatter(
        Xs_cfs[:, 0],
        Xs_cfs[:, 1],
        c="orange",
        cmap=matplotlib.colormaps["tab10"],
        s=40,
        alpha=0.6,
    )
    ax.scatter(
        Xs[:, 0],
        Xs[:, 1],
        c=GROUP_COLORS[0],
        cmap=matplotlib.colormaps["tab10"],
        s=40,
        alpha=0.6,
    )
    for i in range(len(Xs)):
        ax.arrow(
            Xs[i, 0],
            Xs[i, 1],
            Xs_cfs[i, 0] - Xs[i, 0],
            Xs_cfs[i, 1] - Xs[i, 1],
            head_width=0.00,
            head_length=-0.05,
            fc="grey",
            ec="grey",
            alpha=0.5,
        )

    if gen_model is not None:
        plot_generative_model_distribution(ax, gen_model, log_prob_threshold, 2)
    plot_classifier_decision_region(ax, disc_model)

    plt.grid(True, alpha=0.3)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.tight_layout()
    plt.show()
