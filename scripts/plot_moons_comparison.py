"""Script to plot factual and counterfactual points for different methods on moons dataset."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

from cel.datasets.file_dataset import FileDataset
from cel.models.classifier.logistic_regression import (
    MultinomialLogisticRegression,
)
from cel.models.classifier.multilayer_perceptron import MLPClassifier
from cel.plotting.plot_utils import (
    plot_classifier_decision_region,
    plot_generative_model_distribution,
)
from cel.models.generative.maf.maf import MaskedAutoregressiveFlow


def load_counterfactuals(cf_path: Path) -> np.ndarray:
    """Load counterfactuals from CSV file.

    Args:
        cf_path: Path to counterfactuals CSV file.

    Returns:
        Numpy array of counterfactual points.
    """
    df = pd.read_csv(cf_path, header=None)
    return df.values


def load_disc_model(model_path: Path, model_type: str, num_inputs: int = 2):
    """Load discriminator model from checkpoint.

    Args:
        model_path: Path to model checkpoint.
        model_type: Type of model ('MLPClassifier' or 'MultinomialLogisticRegression').
        num_inputs: Number of input features.

    Returns:
        Loaded model.
    """
    if model_type == "MLPClassifier":
        # Model was trained with hidden_layer_sizes=[256, 256]
        model = MLPClassifier(num_inputs=num_inputs, num_targets=2, hidden_layer_sizes=[256, 256])
    elif model_type == "MultinomialLogisticRegression":
        model = MultinomialLogisticRegression(num_inputs=num_inputs, num_targets=2)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load(model_path)
    model.eval()
    return model


def load_gen_model(model_path: Path, num_inputs: int = 2):
    """Load generative model from checkpoint.

    Args:
        model_path: Path to model checkpoint.
        num_inputs: Number of input features.

    Returns:
        Loaded generative model.
    """
    # MAF model with context features for class conditioning
    # Architecture: 8 transforms, 16 hidden features, 4 blocks per layer
    model = MaskedAutoregressiveFlow(
        features=num_inputs,
        hidden_features=16,
        context_features=1,  # For class labels
        num_layers=8,
        num_blocks_per_layer=4,
    )
    model.load(model_path)
    model.eval()
    return model


def plot_single_row(
    results_dir: Path,
    methods: list[str],
    classifier: str,
    fold: int = 0,
    n_samples: int = 10,
    origin_class: int = 0,
    row_name: str = "row",
):
    """Plot a single row of methods.

    Args:
        results_dir: Path to results directory.
        methods: List of method names to plot.
        classifier: Classifier name ('MLPClassifier' or 'MultinomialLogisticRegression').
        fold: Fold number to use.
        n_samples: Number of samples to plot (subsampled).
        origin_class: Origin class to filter samples (0 or 1).
        row_name: Name for the output file.
    """
    torch.manual_seed(0)
    # Load dataset
    dataset = FileDataset(config_path="config/datasets/moons.yaml")

    # Get train and test data for the specified fold
    cv_splits = list(dataset.get_cv_splits(n_splits=5))
    X_train, X_test, y_train, y_test = cv_splits[fold]

    # Train MinMaxScaler on the whole training set
    scaler = MinMaxScaler()
    scaler.fit(X_train)

    # Filter to get samples from the specified origin class
    mask = y_test == origin_class
    X_test_filtered = X_test[mask]
    y_test_filtered = y_test[mask]

    # Scale test points using the scaler fitted on training data
    X_test_filtered = scaler.transform(X_test_filtered)

    # Load discriminator model
    model_path = results_dir / f"fold_{fold}" / f"disc_model_{classifier}.pt"
    disc_model = load_disc_model(model_path, classifier)

    # Load generative model for density plots
    gen_model_path = (
        results_dir
        / f"fold_{fold}"
        / f"gen_model_MaskedAutoregressiveFlow_relabeled_by_{classifier}.pt"
    )
    gen_model = load_gen_model(gen_model_path)

    # Create figure with 1 row, 2 columns
    n_cols = len(methods)
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))
    if n_cols == 1:
        axes = np.array([axes])

    # Map method names to their counterfactual filename prefixes
    method_to_cf_prefix = {
        "AReS": "ARES",
        "Artelt": "Artelt",
        "CADEX": "CADEX",
        "CCHVAE": "CCHVAE",
        "CEGP": "CEGP",
        "CEM_CF": "CEM",
        "CaseBasedSACE": "CaseBasedSACE",
        "CeFlow": "CeFlow",
        "DICE": "DiceExplainerWrapper",
        "DiceExplainerWrapper": "DiceExplainerWrapper",
        "GLOBE_CE": "GLOBE_CE",
        "GLOBE-CE": "GLOBE_CE",
        "GlobalGLANCE": "GLANCE",
        "GroupGLANCE": "GLANCE",
        "PPCEF": "PPCEF",
        "TCREx": "TCREx",
        "GLANCE": "GLANCE",
        "WACH_OURS": "no_plaus_WACH_OURS",
    }

    # Map display names
    display_names = {
        "DiceExplainerWrapper": "DICE",
        "GLOBE_CE": "GLOBE-CE",
        "GlobalGLANCE": "GlobalGLANCE",
        "GroupGLANCE": "GroupGLANCE",
    }

    for idx, method in enumerate(methods):
        ax = axes[idx]

        if method is None:
            ax.axis("off")
            continue

        # Load counterfactuals
        cf_prefix = method_to_cf_prefix.get(method, method)
        cf_path = (
            results_dir / method / f"fold_{fold}" / f"counterfactuals_{cf_prefix}_{classifier}.csv"
        )

        if not cf_path.exists():
            print(f"Warning: Counterfactuals not found for {method} at {cf_path}")
            ax.set_title(f"{method}\n(No data)")
            ax.axis("off")
            continue

        X_cf_full = load_counterfactuals(cf_path)

        # Ensure we have matching number of samples
        n_total_samples = min(len(X_test_filtered), len(X_cf_full))

        # Subsample using indices to preserve factual-counterfactual pairs
        if n_total_samples > n_samples:
            indices = np.random.choice(n_total_samples, size=n_samples, replace=False)
        else:
            indices = np.arange(n_total_samples)

        X_test_plot = X_test_filtered[indices]
        X_cf_plot = X_cf_full[indices]

        # Plot decision boundary
        ax = plot_classifier_decision_region(ax, disc_model)

        # Plot density regions from generative model
        ax = plot_generative_model_distribution(ax, gen_model, prob_threshold=None, num_classes=2)

        # Plot factual points
        ax.scatter(
            X_test_plot[:, 0],
            X_test_plot[:, 1],
            c="blue",
            s=50,
            alpha=0.7,
            label="Factual",
            edgecolors="black",
            linewidths=0.5,
        )

        # Plot counterfactual points
        ax.scatter(
            X_cf_plot[:, 0],
            X_cf_plot[:, 1],
            c="red",
            s=50,
            alpha=0.7,
            label="Counterfactual",
            edgecolors="black",
            linewidths=0.5,
        )

        # Plot arrows from factual to counterfactual
        for i in range(len(X_test_plot)):
            ax.arrow(
                X_test_plot[i, 0],
                X_test_plot[i, 1],
                X_cf_plot[i, 0] - X_test_plot[i, 0],
                X_cf_plot[i, 1] - X_test_plot[i, 1],
                head_width=0.02,
                head_length=0.02,
                fc="gray",
                ec="gray",
                alpha=0.5,
                length_includes_head=True,
            )

        display_name = display_names.get(method, method)
        ax.set_title(display_name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure as PDF
    output_path = results_dir / f"moons_{row_name}_{classifier}_fold{fold}.pdf"
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    print(f"Saved plot to: {output_path}")
    plt.show()


def main():
    """Main function to generate comparison plots."""
    results_dir = Path("results/moons")

    # Plot each row separately
    # Row 0: DICE, PPCEF - origin class 1
    print("Generating row 0 plot (DICE, PPCEF) with origin class 1...")
    plot_single_row(
        results_dir=results_dir,
        methods=["DiceExplainerWrapper", "PPCEF"],
        classifier="MLPClassifier",
        fold=0,
        n_samples=10,
        origin_class=1,
        row_name="row0",
    )

    # Row 1: TCREx, GroupGLANCE - origin class 0
    print("Generating row 1 plot (TCREx, GroupGLANCE) with origin class 0...")
    plot_single_row(
        results_dir=results_dir,
        methods=["TCREx", "GroupGLANCE"],
        classifier="MLPClassifier",
        fold=0,
        n_samples=10,
        origin_class=0,
        row_name="row1",
    )

    # Row 2: GlobalGLANCE, GLOBE-CE - origin class 0
    print("Generating row 2 plot (GlobalGLANCE, GLOBE-CE) with origin class 0...")
    plot_single_row(
        results_dir=results_dir,
        methods=["GlobalGLANCE", "GLOBE_CE"],
        classifier="MLPClassifier",
        fold=0,
        n_samples=10,
        origin_class=0,
        row_name="row2",
    )


if __name__ == "__main__":
    main()
