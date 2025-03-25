import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.datasets import make_moons, make_circles
import os
import logging

from counterfactuals.datasets.generic_counterfactual import (
    CounterfactualWrapper, 
    train_counterfactual_flow_model,
    generate_counterfactuals,
    visualize_batch_distribution,
    visualize_counterfactual_generation
)
from counterfactuals.generative_models.maf import MaskedAutoregressiveFlow

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('counterfactual_example')


def visualize_dataset(X, y, title="Dataset", factual_class=0, counterfactual_class=1, save_path=None):
    """
    Visualize a 2D binary classification dataset
    """
    plt.figure(figsize=(10, 8))
    plt.scatter(
        X[y == factual_class, 0], 
        X[y == factual_class, 1], 
        color='blue', 
        alpha=0.6, 
        label=f'Class {factual_class} (Factual)'
    )
    plt.scatter(
        X[y == counterfactual_class, 0], 
        X[y == counterfactual_class, 1], 
        color='red', 
        alpha=0.6, 
        label=f'Class {counterfactual_class} (Counterfactual)'
    )
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    
    return plt


def visualize_counterfactuals(
    X, 
    y, 
    factual_indices, 
    generated_counterfactuals, 
    title="Generated Counterfactuals",
    factual_class=0,
    counterfactual_class=1,
    max_counterfactuals_per_point=10
):
    """
    Visualize generated counterfactuals for selected factual points
    """
    plt.figure(figsize=(12, 10))
    
    # Plot all original points with low opacity
    plt.scatter(
        X[y == factual_class, 0], 
        X[y == factual_class, 1], 
        color='blue', 
        alpha=0.3, 
        label=f'Class {factual_class} (Factual)'
    )
    plt.scatter(
        X[y == counterfactual_class, 0], 
        X[y == counterfactual_class, 1], 
        color='red', 
        alpha=0.3, 
        label=f'Class {counterfactual_class} (Counterfactual)'
    )
    
    # Plot selected factual points and their counterfactuals
    for idx, factual_idx in enumerate(factual_indices):
        factual_point = X[factual_idx]
        
        # Plot factual point with high opacity
        plt.scatter(
            factual_point[0], 
            factual_point[1], 
            color='blue', 
            s=100, 
            edgecolor='black', 
            alpha=1.0,
            marker='*'
        )
        
        # Get corresponding counterfactuals
        counterfactuals = generated_counterfactuals[idx]
        
        # Limit number of counterfactuals to avoid cluttering
        if len(counterfactuals) > max_counterfactuals_per_point:
            counterfactuals = counterfactuals[:max_counterfactuals_per_point]
        
        # Plot counterfactuals
        plt.scatter(
            counterfactuals[:, 0], 
            counterfactuals[:, 1], 
            color='green', 
            alpha=0.7, 
            marker='x',
            s=50
        )
        
        # Draw lines from factual to counterfactuals
        for cf in counterfactuals:
            plt.plot(
                [factual_point[0], cf[0]], 
                [factual_point[1], cf[1]], 
                'k--', 
                alpha=0.3
            )
    
    plt.title(title)
    plt.legend(['Factuals', 'Counterfactuals', 'Selected Factuals', 'Generated Counterfactuals'])
    plt.grid(True, alpha=0.3)
    return plt


def train_moons_unidirectional():
    """
    Example using the moons dataset with unidirectional counterfactual generation
    (class 0 → class 1)
    """
    logger.info("Starting moons dataset example (unidirectional)")
    
    # Generate the moons dataset
    X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)
    logger.info(f"Generated moons dataset with {len(X)} samples")
    logger.info(f"Class distribution: Class 0: {np.sum(y == 0)}, Class 1: {np.sum(y == 1)}")
    
    # Set save directory
    save_dir = "results/moons_unidirectional"
    os.makedirs(save_dir, exist_ok=True)
    
    # Visualize the dataset
    visualize_dataset(
        X, y, 
        title="Moons Dataset", 
        save_path=os.path.join(save_dir, "moons_dataset.png")
    )
    logger.info(f"Saved dataset visualization to {os.path.join(save_dir, 'moons_dataset.png')}")
    
    # Create the counterfactual wrapper
    logger.info("Creating counterfactual dataset wrapper (unidirectional)")
    dataset = CounterfactualWrapper(
        X=X,
        y=y,
        factual_class=0,
        counterfactual_class=1,
        n_nearest=8,
        noise_level=0.03,
        log_level='INFO',
        bidirectional=False
    )
    
    # Visualize batch distribution and factual-counterfactual mappings
    logger.info("Visualizing batch distribution and factual-counterfactual mappings")
    batch_vis_dir = os.path.join(save_dir, "batch_visualization")
    os.makedirs(batch_vis_dir, exist_ok=True)
    visualize_batch_distribution(
        dataset, 
        batch_size=None,  # Use default (n_nearest)
        save_dir=batch_vis_dir
    )
    logger.info(f"Saved batch visualizations to {batch_vis_dir}")
    
    # Train the model
    logger.info("Training counterfactual flow model (unidirectional)")
    model = train_counterfactual_flow_model(
        dataset=dataset,
        flow_model_class=MaskedAutoregressiveFlow,
        hidden_features=64,
        num_layers=5,
        num_blocks_per_layer=2,
        learning_rate=1e-3,
        batch_size=None,  # Use default (n_nearest)
        num_epochs=300,   # Reduced for the example
        patience=100,     # Reduced for the example
        noise_level=0.03,
        save_dir=save_dir,
        log_interval=10,
        direction='forward',
        bidirectional_model=False
    )
    logger.info("Model training complete")
    
    # Generate and visualize counterfactuals
    logger.info("Generating and visualizing counterfactuals")
    cf_vis_dir = os.path.join(save_dir, "counterfactual_visualization")
    os.makedirs(cf_vis_dir, exist_ok=True)
    
    generated_cfs, factual_points = visualize_counterfactual_generation(
        model=model,
        dataset=dataset,
        num_factual=8,
        num_samples=50,
        temperature=0.8,
        save_dir=cf_vis_dir,
        direction='forward',
        bidirectional_model=False
    )
    logger.info(f"Saved counterfactual visualizations to {cf_vis_dir}")
    
    return model, dataset


def train_moons_bidirectional():
    """
    Example using the moons dataset with bidirectional counterfactual generation
    (class 0 ↔ class 1)
    """
    logger.info("Starting moons dataset example (bidirectional)")
    
    # Generate the moons dataset
    X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)
    logger.info(f"Generated moons dataset with {len(X)} samples")
    logger.info(f"Class distribution: Class 0: {np.sum(y == 0)}, Class 1: {np.sum(y == 1)}")
    
    # Set save directory
    save_dir = "results/moons_bidirectional"
    os.makedirs(save_dir, exist_ok=True)
    
    # Visualize the dataset
    visualize_dataset(
        X, y, 
        title="Moons Dataset (Bidirectional)", 
        save_path=os.path.join(save_dir, "moons_dataset.png")
    )
    logger.info(f"Saved dataset visualization to {os.path.join(save_dir, 'moons_dataset.png')}")
    
    # Create the counterfactual wrapper with bidirectional=True
    logger.info("Creating counterfactual dataset wrapper (bidirectional)")
    dataset = CounterfactualWrapper(
        X=X,
        y=y,
        factual_class=0,
        counterfactual_class=1,
        n_nearest=8,
        noise_level=0.03,
        log_level='INFO',
        bidirectional=True  # Enable bidirectional mode
    )
    
    # Visualize batch distributions in both directions
    logger.info("Visualizing batch distribution for forward direction")
    fwd_batch_vis_dir = os.path.join(save_dir, "batch_visualization_forward")
    os.makedirs(fwd_batch_vis_dir, exist_ok=True)
    visualize_batch_distribution(
        dataset, 
        batch_size=None,
        save_dir=fwd_batch_vis_dir
    )
    logger.info(f"Saved forward batch visualizations to {fwd_batch_vis_dir}")
    
    # Approach 1: Train separate models for each direction
    logger.info("Training forward model (class 0 → class 1)")
    forward_model = train_counterfactual_flow_model(
        dataset=dataset,
        flow_model_class=MaskedAutoregressiveFlow,
        hidden_features=64,
        num_layers=5,
        num_blocks_per_layer=2,
        learning_rate=1e-3,
        batch_size=None,
        num_epochs=300,
        patience=100,
        noise_level=0.03,
        save_dir=os.path.join(save_dir, "forward_model"),
        log_interval=10,
        direction='forward',
        bidirectional_model=False
    )
    logger.info("Forward model training complete")
    
    logger.info("Training reverse model (class 1 → class 0)")
    reverse_model = train_counterfactual_flow_model(
        dataset=dataset,
        flow_model_class=MaskedAutoregressiveFlow,
        hidden_features=64,
        num_layers=5,
        num_blocks_per_layer=2,
        learning_rate=1e-3,
        batch_size=None,
        num_epochs=300,
        patience=100,
        noise_level=0.03,
        save_dir=os.path.join(save_dir, "reverse_model"),
        log_interval=10,
        direction='reverse',
        bidirectional_model=False
    )
    logger.info("Reverse model training complete")
    
    # Approach 2: Train a single model that handles both directions
    logger.info("Training unified bidirectional model (class 0 ↔ class 1)")
    bidirectional_model = train_counterfactual_flow_model(
        dataset=dataset,
        flow_model_class=MaskedAutoregressiveFlow,
        hidden_features=80,  # Slightly larger to handle both directions
        num_layers=6,
        num_blocks_per_layer=2,
        learning_rate=1e-3,
        batch_size=None,
        num_epochs=400,
        patience=150,
        noise_level=0.03,
        save_dir=os.path.join(save_dir, "unified_model"),
        log_interval=10,
        direction='both',
        bidirectional_model=True  # Enable bidirectional modeling
    )
    logger.info("Unified bidirectional model training complete")
    
    # Generate and visualize counterfactuals using forward model
    logger.info("Generating and visualizing counterfactuals using forward model")
    fwd_cf_vis_dir = os.path.join(save_dir, "counterfactual_visualization_forward")
    os.makedirs(fwd_cf_vis_dir, exist_ok=True)
    
    visualize_counterfactual_generation(
        model=forward_model,
        dataset=dataset,
        num_factual=6,
        num_samples=40,
        temperature=0.8,
        save_dir=fwd_cf_vis_dir,
        direction='forward',
        bidirectional_model=False
    )
    logger.info(f"Saved forward counterfactual visualizations to {fwd_cf_vis_dir}")
    
    # Generate and visualize counterfactuals using reverse model
    logger.info("Generating and visualizing counterfactuals using reverse model")
    rev_cf_vis_dir = os.path.join(save_dir, "counterfactual_visualization_reverse")
    os.makedirs(rev_cf_vis_dir, exist_ok=True)
    
    visualize_counterfactual_generation(
        model=reverse_model,
        dataset=dataset,
        num_factual=6,
        num_samples=40,
        temperature=0.8,
        save_dir=rev_cf_vis_dir,
        direction='reverse',
        bidirectional_model=False
    )
    logger.info(f"Saved reverse counterfactual visualizations to {rev_cf_vis_dir}")
    
    # Generate and visualize counterfactuals using unified bidirectional model
    logger.info("Generating and visualizing counterfactuals using unified bidirectional model")
    bi_cf_vis_dir = os.path.join(save_dir, "counterfactual_visualization_unified")
    os.makedirs(bi_cf_vis_dir, exist_ok=True)
    
    visualize_counterfactual_generation(
        model=bidirectional_model,
        dataset=dataset,
        num_factual=6,
        num_samples=40,
        temperature=0.8,
        save_dir=bi_cf_vis_dir,
        direction='both',  # Generate counterfactuals in both directions
        bidirectional_model=True
    )
    logger.info(f"Saved unified bidirectional counterfactual visualizations to {bi_cf_vis_dir}")
    
    return {
        'dataset': dataset,
        'forward_model': forward_model,
        'reverse_model': reverse_model,
        'bidirectional_model': bidirectional_model
    }


def train_circles_bidirectional():
    """
    Example using the circles dataset with bidirectional counterfactual generation
    (inner circle ↔ outer circle)
    """
    logger.info("Starting circles dataset example (bidirectional)")
    
    # Generate the circles dataset
    X, y = make_circles(n_samples=1000, noise=0.05, factor=0.5, random_state=42)
    logger.info(f"Generated circles dataset with {len(X)} samples")
    logger.info(f"Class distribution: Class 0: {np.sum(y == 0)}, Class 1: {np.sum(y == 1)}")
    
    # Set save directory
    save_dir = "results/circles_bidirectional"
    os.makedirs(save_dir, exist_ok=True)
    
    # Visualize the dataset
    visualize_dataset(
        X, y, 
        title="Circles Dataset (Bidirectional)",
        factual_class=0,  # Inner circle
        counterfactual_class=1,  # Outer circle
        save_path=os.path.join(save_dir, "circles_dataset.png")
    )
    logger.info(f"Saved dataset visualization to {os.path.join(save_dir, 'circles_dataset.png')}")
    
    # Create the counterfactual wrapper with bidirectional=True
    logger.info("Creating counterfactual dataset wrapper (bidirectional)")
    dataset = CounterfactualWrapper(
        X=X,
        y=y,
        factual_class=0,  # Inner circle
        counterfactual_class=1,  # Outer circle
        n_nearest=8,
        noise_level=0.02,  # Less noise for circles
        log_level='INFO',
        bidirectional=True  # Enable bidirectional mode
    )
    
    # Train a single unified bidirectional model
    logger.info("Training unified bidirectional model for circles")
    bidirectional_model = train_counterfactual_flow_model(
        dataset=dataset,
        flow_model_class=MaskedAutoregressiveFlow,
        hidden_features=80,
        num_layers=6,
        num_blocks_per_layer=2,
        learning_rate=1e-3,
        batch_size=None,
        num_epochs=400,
        patience=150,
        noise_level=0.02,
        save_dir=save_dir,
        log_interval=10,
        direction='both',  # Train in both directions
        bidirectional_model=True  # Enable bidirectional modeling
    )
    logger.info("Unified bidirectional model training complete")
    
    # Generate and visualize counterfactuals using unified bidirectional model
    logger.info("Generating and visualizing counterfactuals in both directions")
    cf_vis_dir = os.path.join(save_dir, "counterfactual_visualization")
    os.makedirs(cf_vis_dir, exist_ok=True)
    
    visualize_counterfactual_generation(
        model=bidirectional_model,
        dataset=dataset,
        num_factual=8,
        num_samples=50,
        temperature=0.8,
        save_dir=cf_vis_dir,
        direction='both',  # Generate counterfactuals in both directions
        bidirectional_model=True
    )
    logger.info(f"Saved bidirectional counterfactual visualizations to {cf_vis_dir}")
    
    return bidirectional_model, dataset


if __name__ == "__main__":
    # Parse command line arguments to determine which examples to run
    import argparse
    parser = argparse.ArgumentParser(description='Train counterfactual generative models')
    parser.add_argument('--all', action='store_true', help='Run all examples')
    parser.add_argument('--moons-uni', action='store_true', help='Run moons unidirectional example')
    parser.add_argument('--moons-bi', action='store_true', help='Run moons bidirectional example')
    parser.add_argument('--circles-bi', action='store_true', help='Run circles bidirectional example')
    args = parser.parse_args()
    
    # If no specific arguments provided, run all examples
    if not (args.moons_uni or args.moons_bi or args.circles_bi):
        args.all = True
    
    # Run the selected examples
    if args.all or args.moons_uni:
        logger.info("\n=== Starting Moons Unidirectional Example ===")
        moons_uni_model, moons_uni_dataset = train_moons_unidirectional()
        logger.info("Moons unidirectional example completed")
    
    if args.all or args.moons_bi:
        logger.info("\n=== Starting Moons Bidirectional Example ===")
        moons_bi_results = train_moons_bidirectional()
        logger.info("Moons bidirectional example completed")
    
    if args.all or args.circles_bi:
        logger.info("\n=== Starting Circles Bidirectional Example ===")
        circles_bi_model, circles_bi_dataset = train_circles_bidirectional()
        logger.info("Circles bidirectional example completed")
    
    logger.info("\nAll examples completed successfully!")
    logger.info("Results saved to:")
    if args.all or args.moons_uni:
        logger.info("  - results/moons_unidirectional")
    if args.all or args.moons_bi:
        logger.info("  - results/moons_bidirectional")
    if args.all or args.circles_bi:
        logger.info("  - results/circles_bidirectional") 