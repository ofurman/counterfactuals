import os
import logging

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay

from counterfactuals.cf_methods.dicoflex.generation import generate_multiclass_counterfactuals

logger = logging.getLogger('counterfactual')


def visualize_multiclass_counterfactual_generation(
    model,
    dataset,
    disc_model,
    masks,
    p_values,
    num_factual=5,
    num_samples=20,
    temperature=0.8,
    save_dir=None,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Visualize multiclass counterfactual generation results.

    Args:
        model: Trained flow model
        dataset: MulticlassCounterfactualWrapper instance
        disc_model: Discriminator model for visualizing decision boundaries
        num_factual: Number of factual points to generate counterfactuals for
        num_samples: Number of counterfactual samples to generate per factual point
        temperature: Temperature for sampling (higher = more diversity)
        save_dir: Directory to save visualizations
        device: Device to use for generation
    """
    # Only works for 2D data
    assert dataset.X.shape[1] == 2, "Only 2D data is supported for visualization"

    # Create samples directory
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        samples_dir = os.path.join(save_dir, "samples")
        os.makedirs(samples_dir, exist_ok=True)

    results = []

    # Function to plot decision boundaries
    def plot_decision_boundary(ax=None, alpha=0.3):
        """Plot decision boundaries from discriminator model"""
        # Check if we have a subplot or create a new one
        if ax is None:
            ax = plt.gca()

        # Create a grid of points
        x_min, x_max = plt.xlim()
        y_min, y_max = plt.ylim()

        # Create mesh grid
        xline = torch.linspace(x_min, x_max, 100)
        yline = torch.linspace(y_min, y_max, 100)
        xgrid, ygrid = torch.meshgrid(xline, yline)
        xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)

        y_hat = disc_model.predict(xyinput)
        y_hat = y_hat.reshape(100, 100)

        display = DecisionBoundaryDisplay(xx0=xgrid, xx1=ygrid, response=y_hat)
        ax = display.plot(plot_method="contour", ax=ax, alpha=0.3).ax_
        return ax


    for mask_idx, mask in enumerate(masks):
        logger.info(f"Generating counterfactuals for mask {mask}")
        for p_value in p_values:
            logger.info(f"Generating counterfactuals for p-norm {p_value}")
            # For each factual class
            for factual_class in dataset.factual_classes:
                logger.info(f"Generating counterfactuals for factual class: {factual_class}")

                # Get factual points for this class
                factual_scaled = dataset.X_by_class_scaled[factual_class]
                factual_original = dataset.X_by_class[factual_class]

                # Sample factual points
                if len(factual_scaled) <= num_factual:
                    factual_indices = np.arange(len(factual_scaled))
                else:
                    factual_indices = np.random.choice(
                        len(factual_scaled),
                        size=num_factual,
                        replace=False
                    )

                factual_points = factual_scaled[factual_indices]

                # For each target class
                for target_class in dataset.classes:
                    if target_class == factual_class:
                        continue  # Skip generating counterfactuals for the same class

                    logger.info(f"Generating counterfactuals from class {factual_class} to class {target_class} with mask {mask} and p-value {p_value}")

                    # Generate counterfactuals
                    generated_cfs = generate_multiclass_counterfactuals(
                        model=model,
                        factual_points=factual_points,
                        target_class=target_class,
                        n_samples=num_samples,
                        temperature=temperature,
                        device=device,
                        num_classes=len(dataset.classes),
                        mask=mask,
                        p_value=p_value
                    )

                    # Convert to original scale for better interpretability
                    factual_orig = dataset.feature_transformer.inverse_transform(factual_points)

                    # Fix: Handle the 3D structure of generated_cfs by reshaping before inverse_transform
                    generated_cfs_orig = []
                    for cf_batch in generated_cfs:
                        # Check if we have a 3D array and reshape if needed
                        if cf_batch.ndim == 3:  # Shape is [batch_size, num_samples, features]
                            batch_size, n_samples, n_features = cf_batch.shape
                            reshaped_batch = cf_batch.reshape(-1, n_features)  # Flatten to 2D
                            transformed = dataset.feature_transformer.inverse_transform(reshaped_batch)
                            # Reshape back to original 3D shape
                            transformed = transformed.reshape(batch_size, n_samples, n_features)
                            generated_cfs_orig.append(transformed)
                        else:  # Regular 2D case
                            generated_cfs_orig.append(dataset.feature_transformer.inverse_transform(cf_batch))

                    # Store results for this class pair
                    results.append({
                        'factual_class': factual_class,
                        'target_class': target_class,
                        'factual_indices': factual_indices,
                        'factual_points': factual_points,
                        'factual_orig': factual_orig,
                        'generated_cfs': generated_cfs,
                        'generated_cfs_orig': generated_cfs_orig
                    })

                    # Plot settings
                    colors = plt.cm.tab10(np.linspace(0, 1, num_factual))

                    # Create overview plot for this class pair
                    fig, ax = plt.subplots(figsize=(14, 10))

                    # Plot all original data points with low opacity
                    plt.scatter(
                        dataset.X[:, 0],
                        dataset.X[:, 1],
                        c=dataset.y,
                        cmap=plt.cm.tab10,
                        alpha=0.2,
                        s=30
                    )

                    # Plot decision boundaries before adding other elements
                    plot_decision_boundary(ax=ax, alpha=0.6)

                    # Add a legend for the original classes
                    for cls in dataset.classes:
                        plt.scatter([], [], color=plt.cm.tab10(cls % 10), label=f'Class {cls}')

                    # For each factual point
                    for i, (f_idx, factual, cf_samples) in enumerate(zip(factual_indices, factual_orig, generated_cfs_orig)):
                        # Plot factual point
                        plt.scatter(
                            factual[0],
                            factual[1],
                            color=colors[i],
                            s=150,
                            marker='*',
                            edgecolor='black',
                            label=f'Factual {i+1}' if i < 5 else None  # Limit legend entries
                        )

                        # Plot generated counterfactuals - Fix to handle potential 3D array
                        if cf_samples.ndim == 3:
                            # If we have a 3D array, take the first dimension
                            cf_to_plot = cf_samples[0]
                        else:
                            cf_to_plot = cf_samples

                        plt.scatter(
                            cf_to_plot[:, 0],
                            cf_to_plot[:, 1],
                            color=colors[i],
                            alpha=0.6,
                            marker='x',
                            s=50,
                            label=f'Generated CFs for Factual {i+1}' if i < 5 else None
                        )

                        # Draw lines to a few counterfactuals
                        for j in range(min(5, len(cf_to_plot))):
                            plt.plot(
                                [factual[0], cf_to_plot[j, 0]],
                                [factual[1], cf_to_plot[j, 1]],
                                color=colors[i],
                                linestyle='--',
                                alpha=0.3
                            )

                        # Create individual plot for this factual point
                        if save_dir:
                            fig_ind, ax_ind = plt.subplots(figsize=(10, 8))

                            # Plot original data with low opacity
                            plt.scatter(
                                dataset.X[:, 0],
                                dataset.X[:, 1],
                                c=dataset.y,
                                cmap=plt.cm.tab10,
                                alpha=0.2,
                                s=30
                            )

                            # Plot decision boundaries
                            plot_decision_boundary(ax=ax_ind, alpha=0.6)

                            # Plot factual point
                            plt.scatter(
                                factual[0],
                                factual[1],
                                color=colors[i],
                                s=200,
                                marker='*',
                                edgecolor='black',
                                label=f'Factual Point {i+1}'
                            )

                            # Plot generated counterfactuals - Fix to handle potential 3D array
                            if cf_samples.ndim == 3:
                                # If we have a 3D array, take the first dimension
                                cf_to_plot = cf_samples[0]
                            else:
                                cf_to_plot = cf_samples

                            plt.scatter(
                                cf_to_plot[:, 0],
                                cf_to_plot[:, 1],
                                color=colors[i],
                                alpha=0.7,
                                marker='x',
                                s=80,
                                label='Generated Counterfactuals'
                            )

                            # Add density contour of generated points
                            try:
                                from scipy.stats import gaussian_kde

                                # If we have enough points, create a density plot
                                if len(cf_to_plot) >= 10:
                                    kde = gaussian_kde(cf_to_plot.T)

                                    # Create a grid of points
                                    x_min, x_max = plt.xlim()
                                    y_min, y_max = plt.ylim()

                                    xx, yy = np.meshgrid(
                                        np.linspace(x_min, x_max, 100),
                                        np.linspace(y_min, y_max, 100)
                                    )

                                    # Evaluate KDE on grid
                                    positions = np.vstack([xx.ravel(), yy.ravel()])
                                    zz = kde(positions).reshape(xx.shape)

                                    # Plot contour
                                    plt.contour(
                                        xx, yy, zz,
                                        cmap=plt.cm.Oranges,
                                        alpha=0.5
                                    )
                            except Exception as e:
                                logger.warning(f"Could not create density plot: {e}")

                            # Draw lines to all counterfactuals
                            for j in range(len(cf_to_plot)):
                                plt.plot(
                                    [factual[0], cf_to_plot[j, 0]],
                                    [factual[1], cf_to_plot[j, 1]],
                                    color=colors[i],
                                    linestyle='--',
                                    alpha=0.3
                                )

                            plt.title(f"Counterfactuals Generated for Factual Point {i+1} (Class {factual_class} → Class {target_class})")
                            plt.legend()
                            plt.grid(True, alpha=0.3)

                            # Create subdirectory for each class pair if needed
                            subdir = os.path.join(samples_dir, f"class_{factual_class}_to_{target_class}")
                            os.makedirs(subdir, exist_ok=True)

                            plt.savefig(os.path.join(subdir, f"factual_{i+1}_counterfactuals.png"))
                            plt.close()

                    # Finish and save overview plot
                    plt.title(f"Overview of Generated Counterfactuals (Class {factual_class} → Class {target_class}) with mask {mask} and p-value {p_value}")
                    plt.legend()
                    plt.grid(True, alpha=0.3)

                    if save_dir:
                        plt.savefig(os.path.join(save_dir, f"counterfactual_overview_class_{factual_class}_to_{target_class}_mask_{mask}_p_value_{p_value}.png"))
                        plt.close()

    return results
