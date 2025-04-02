import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import logging
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from counterfactuals.datasets.blobs import BlobsDataset
from counterfactuals.datasets.generic_counterfactual import (
    CounterfactualWrapper, 
    train_counterfactual_flow_model,
    generate_counterfactuals,
    visualize_batch_distribution,
    visualize_counterfactual_generation
)
from counterfactuals.generative_models.maf import MaskedAutoregressiveFlow
from counterfactuals.metrics.metrics import CFMetrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('blobs_counterfactual_example')


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


def evaluate_counterfactuals(
    model, 
    dataset, 
    X, 
    y, 
    factual_indices, 
    generated_cfs,
    feature_names, 
    direction="forward",
    save_dir=None
):
    """
    Evaluate generated counterfactuals using both simple statistics and CFMetrics
    
    Args:
        model: The trained flow model
        dataset: CounterfactualWrapper instance
        X: Original feature data
        y: Labels 
        factual_indices: Indices of factual points
        generated_cfs: Generated counterfactuals
        feature_names: Names of features for display
        direction: 'forward' or 'reverse'
        save_dir: Directory to save results
    """
    # Create a discriminator model that simply returns the class labels
    class SimpleDiscriminator(torch.nn.Module):
        def __init__(self, threshold=0.5):
            super().__init__()
            self.threshold = threshold
            
        def forward(self, x):
            return x
            
        def predict(self, x):
            # Simple threshold-based classification on the first feature
            # Adjust as needed based on blobs dataset properties
            if isinstance(x, np.ndarray):
                return (x[:, 0] > self.threshold).astype(np.int64)
            else:
                return (x[:, 0] > self.threshold).long()
    
    # 1. Print basic statistics about the changes needed
    logger.info("\n=== Basic Counterfactual Statistics ===")
    
    # Transform generated counterfactuals back to original space
    generated_cfs_orig = []
    for cf_batch in generated_cfs:
        # Handle the 3D structure by reshaping before inverse_transform
        if cf_batch.ndim == 3:
            batch_size, n_samples, n_features = cf_batch.shape
            reshaped_batch = cf_batch.reshape(-1, n_features)
            transformed = dataset.feature_transformer.inverse_transform(reshaped_batch)
            transformed = transformed.reshape(batch_size, n_samples, n_features)
            generated_cfs_orig.append(transformed)
        else:
            generated_cfs_orig.append(dataset.feature_transformer.inverse_transform(cf_batch))
    
    # For each factual point, show the original values and generated counterfactuals
    all_changes = []
    for i, (factual_idx, cf_samples) in enumerate(zip(factual_indices, generated_cfs_orig)):
        factual_orig = X[factual_idx]
        
        logger.info(f"\nFactual example #{i+1} (Class {y[factual_idx]}):")
        for j, feature_name in enumerate(feature_names):
            logger.info(f"  {feature_name}: {factual_orig[j]:.3f}")
        
        if cf_samples.ndim == 3:
            cf_to_analyze = cf_samples[0]
        else:
            cf_to_analyze = cf_samples
            
        # Calculate average counterfactual values
        cf_means = np.mean(cf_to_analyze, axis=0)
        cf_stds = np.std(cf_to_analyze, axis=0)
        
        logger.info(f"Generated counterfactual (mean values):")
        for j, feature_name in enumerate(feature_names):
            logger.info(f"  {feature_name}: {cf_means[j]:.3f} ± {cf_stds[j]:.3f}")
        
        # Calculate average changes needed
        changes = cf_means - factual_orig
        logger.info(f"Changes needed:")
        for j, feature_name in enumerate(feature_names):
            logger.info(f"  {feature_name}: {changes[j]:+.3f}")
        
        all_changes.append(changes)
    
    # Calculate aggregate statistics
    all_changes = np.array(all_changes)
    mean_changes = np.mean(all_changes, axis=0)
    std_changes = np.std(all_changes, axis=0)
    
    logger.info("\n=== Aggregate Changes ===")
    for j, feature_name in enumerate(feature_names):
        logger.info(f"{feature_name}: {mean_changes[j]:+.3f} ± {std_changes[j]:.3f}")
    
    # 2. Calculate more sophisticated counterfactual metrics
    logger.info("\n=== Counterfactual Metrics ===")
    
    # Flatten the generated counterfactuals
    all_counterfactuals = []
    for cf_samples in generated_cfs:
        if cf_samples.ndim == 3:
            for batch in cf_samples:
                all_counterfactuals.extend(batch)
        else:
            all_counterfactuals.extend(cf_samples)
    
    X_cf = np.array(all_counterfactuals)
    
    # Calculate target labels (opposite of factual)
    y_factual = y[factual_indices]
    y_target = 1 - y_factual
    
    # Replicate y_target to match X_cf size
    cf_per_factual = X_cf.shape[0] // len(factual_indices)
    y_target_rep = np.repeat(y_target, cf_per_factual)
    
    # Get training data
    X_train = dataset.X_train
    y_train = dataset.y_train
    
    # Get factual points in the transformed space
    X_test = dataset.feature_transformer.transform(X[factual_indices])
    X_test_rep = np.repeat(X_test, cf_per_factual, axis=0)
    
    # Create a simplified discriminator model
    discriminator = SimpleDiscriminator(threshold=0.5)
    
    # For plausibility threshold, estimate from training data log likelihoods
    with torch.no_grad():
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        log_probs = model(X_train_tensor, y_train_tensor)
        threshold = torch.median(log_probs)
    
    # Create metrics calculator
    metrics = CFMetrics(
        X_cf=X_cf, 
        y_target=y_target_rep,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test_rep, 
        y_test=y_factual.repeat(cf_per_factual),
        gen_model=model,
        disc_model=discriminator,
        continuous_features=dataset.numerical_features,
        categorical_features=dataset.categorical_features,
        prob_plausibility_threshold=threshold
    )
    
    # Calculate all metrics
    metrics_results = metrics.calc_all_metrics()
    
    # Log metrics
    for metric_name, metric_value in metrics_results.items():
        logger.info(f"{metric_name}: {metric_value:.4f}")
    
    # Save metrics to file
    if save_dir:
        metrics_file = os.path.join(save_dir, f"metrics_{direction}.json")
        import json
        with open(metrics_file, 'w') as f:
            json.dump(metrics_results, f, indent=2)
    
    return metrics_results, generated_cfs_orig


def train_blobs_unidirectional():
    """
    Example using the blobs dataset with unidirectional counterfactual generation
    """
    logger.info("Starting blobs dataset example (unidirectional)")
    
    # Load the blobs dataset
    blobs_dataset = BlobsDataset(file_path="data/blobs.csv")
    X = np.vstack([blobs_dataset.X_train, blobs_dataset.X_test])
    y = np.concatenate([blobs_dataset.y_train, blobs_dataset.y_test])
    logger.info(f"Loaded blobs dataset with {len(X)} samples and {X.shape[1]} features")
    logger.info(f"Class distribution: Class 0: {np.sum(y == 0)}, Class 1: {np.sum(y == 1)}")
    
    # Set save directory
    save_dir = "results/blobs_unidirectional"
    os.makedirs(save_dir, exist_ok=True)
    
    # Define feature names
    feature_names = [f"Feature {i+1}" for i in range(X.shape[1])]
    
    # Visualize the dataset
    if X.shape[1] == 2:
        # If 2D, visualize directly
        plt = visualize_dataset(
            X, y, 
            title="Blobs Dataset", 
            factual_class=0,
            counterfactual_class=1,
            save_path=os.path.join(save_dir, "blobs_dataset.png")
        )
        logger.info(f"Saved dataset visualization to {os.path.join(save_dir, 'blobs_dataset.png')}")
    else:
        # If higher dimensional, use PCA for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        plt.figure(figsize=(10, 8))
        plt.scatter(
            X_pca[y == 0, 0], 
            X_pca[y == 0, 1], 
            color='blue', 
            alpha=0.6, 
            label=f'Class 0 (Factual)'
        )
        plt.scatter(
            X_pca[y == 1, 0], 
            X_pca[y == 1, 1], 
            color='red', 
            alpha=0.6, 
            label=f'Class 1 (Counterfactual)'
        )
        plt.title("Blobs Dataset (PCA Projection)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, "blobs_dataset_pca.png"))
        plt.close()
        logger.info(f"Saved PCA visualization to {os.path.join(save_dir, 'blobs_dataset_pca.png')}")
    
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
    
    # Visualize batch distribution
    logger.info("Visualizing batch distribution")
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
        num_epochs=300,
        patience=100,
        noise_level=0.03,
        save_dir=save_dir,
        log_interval=10,
        direction='forward',
        bidirectional_model=False
    )
    logger.info("Model training complete")
    
    # Generate counterfactuals for evaluation
    logger.info("Generating counterfactuals for evaluation")
    
    # Select class 0 examples to generate counterfactuals for
    factual_indices = np.where(y == 0)[0][:20]
    factual_points = dataset.feature_transformer.transform(X[factual_indices])
    
    generated_cfs = generate_counterfactuals(
        model=model,
        factual_points=factual_points,
        n_samples=10,
        temperature=0.8,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Evaluate the counterfactuals
    metrics_results, generated_cfs_orig = evaluate_counterfactuals(
        model=model,
        dataset=dataset,
        X=X,
        y=y,
        factual_indices=factual_indices,
        generated_cfs=generated_cfs,
        feature_names=feature_names,
        direction="forward",
        save_dir=save_dir
    )
    
    # Visualize generated counterfactuals
    cf_vis_dir = os.path.join(save_dir, "counterfactual_visualization")
    os.makedirs(cf_vis_dir, exist_ok=True)
    
    visualize_counterfactual_generation(
        model=model,
        dataset=dataset,
        num_factual=8,
        num_samples=20,
        temperature=0.8,
        save_dir=cf_vis_dir,
        direction='forward',
        bidirectional_model=False
    )
    logger.info(f"Saved counterfactual visualizations to {cf_vis_dir}")
    
    # Save generated counterfactuals for further analysis
    np.save(os.path.join(save_dir, "factual_points.npy"), X[factual_indices])
    
    # Save generated counterfactuals (need to convert to regular arrays first)
    cf_samples_array = []
    for cf_batch in generated_cfs_orig:
        if cf_batch.ndim == 3:
            cf_samples_array.append(cf_batch[0])
        else:
            cf_samples_array.append(cf_batch)
    
    np.save(os.path.join(save_dir, "generated_counterfactuals.npy"), np.array(cf_samples_array, dtype=object))
    
    return model, dataset, metrics_results


def train_blobs_bidirectional():
    """
    Example using the blobs dataset with bidirectional counterfactual generation
    """
    logger.info("Starting blobs dataset example (bidirectional)")
    
    # Load the blobs dataset
    blobs_dataset = BlobsDataset(file_path="data/blobs.csv")
    X = np.vstack([blobs_dataset.X_train, blobs_dataset.X_test])
    y = np.concatenate([blobs_dataset.y_train, blobs_dataset.y_test])
    logger.info(f"Loaded blobs dataset with {len(X)} samples and {X.shape[1]} features")
    logger.info(f"Class distribution: Class 0: {np.sum(y == 0)}, Class 1: {np.sum(y == 1)}")
    
    # Set save directory
    save_dir = "results/blobs_bidirectional"
    os.makedirs(save_dir, exist_ok=True)
    
    # Define feature names
    feature_names = [f"Feature {i+1}" for i in range(X.shape[1])]
    
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
    
    # Train a single unified bidirectional model
    logger.info("Training unified bidirectional model for blobs dataset")
    bidirectional_model = train_counterfactual_flow_model(
        dataset=dataset,
        flow_model_class=MaskedAutoregressiveFlow,
        hidden_features=80,
        num_layers=6,
        num_blocks_per_layer=2,
        learning_rate=1e-3,
        batch_size=None,
        num_epochs=300,
        patience=100,
        noise_level=0.03,
        save_dir=save_dir,
        log_interval=10,
        direction='both',  # Train in both directions
        bidirectional_model=True  # Enable bidirectional modeling
    )
    logger.info("Unified bidirectional model training complete")
    
    # Generate counterfactuals for evaluation in both directions
    logger.info("Generating counterfactuals in both directions")
    
    metrics_results = {}
    
    # 1. Class 0 → Class 1
    logger.info("\n--- Generating counterfactuals: Class 0 → Class 1 ---")
    factual_indices_0 = np.where(y == 0)[0][:20]
    factual_points_0 = dataset.feature_transformer.transform(X[factual_indices_0])
    
    generated_cfs_0_to_1 = generate_counterfactuals(
        model=bidirectional_model,
        factual_points=factual_points_0,
        n_samples=10,
        temperature=0.8,
        device="cuda" if torch.cuda.is_available() else "cpu",
        direction_indicator=0.0,  # Forward direction
        bidirectional_model=True
    )
    
    # Evaluate forward direction
    metrics_forward, cfs_orig_forward = evaluate_counterfactuals(
        model=bidirectional_model,
        dataset=dataset,
        X=X,
        y=y,
        factual_indices=factual_indices_0,
        generated_cfs=generated_cfs_0_to_1,
        feature_names=feature_names,
        direction="forward",
        save_dir=os.path.join(save_dir, "class0_to_class1")
    )
    metrics_results['forward'] = metrics_forward
    
    # 2. Class 1 → Class 0
    logger.info("\n--- Generating counterfactuals: Class 1 → Class 0 ---")
    factual_indices_1 = np.where(y == 1)[0][:20]
    factual_points_1 = dataset.feature_transformer.transform(X[factual_indices_1])
    
    generated_cfs_1_to_0 = generate_counterfactuals(
        model=bidirectional_model,
        factual_points=factual_points_1,
        n_samples=10,
        temperature=0.8,
        device="cuda" if torch.cuda.is_available() else "cpu",
        direction_indicator=1.0,  # Reverse direction
        bidirectional_model=True
    )
    
    # Evaluate reverse direction
    metrics_reverse, cfs_orig_reverse = evaluate_counterfactuals(
        model=bidirectional_model,
        dataset=dataset,
        X=X,
        y=y,
        factual_indices=factual_indices_1,
        generated_cfs=generated_cfs_1_to_0,
        feature_names=feature_names,
        direction="reverse",
        save_dir=os.path.join(save_dir, "class1_to_class0")
    )
    metrics_results['reverse'] = metrics_reverse
    
    # Compare metrics between directions
    logger.info("\n=== Metrics Comparison Between Directions ===")
    for metric in metrics_forward.keys():
        logger.info(f"{metric}: Forward={metrics_forward[metric]:.4f}, Reverse={metrics_reverse[metric]:.4f}, "
                   f"Diff={metrics_forward[metric]-metrics_reverse[metric]:+.4f}")
    
    # Visualize bidirectional counterfactuals
    cf_vis_dir = os.path.join(save_dir, "counterfactual_visualization")
    os.makedirs(cf_vis_dir, exist_ok=True)
    
    visualize_counterfactual_generation(
        model=bidirectional_model,
        dataset=dataset,
        num_factual=6,
        num_samples=20,
        temperature=0.8,
        save_dir=cf_vis_dir,
        direction='both',  # Generate counterfactuals in both directions
        bidirectional_model=True
    )
    logger.info(f"Saved bidirectional counterfactual visualizations to {cf_vis_dir}")
    
    # Save metrics comparison
    import json
    with open(os.path.join(save_dir, "metrics_comparison.json"), 'w') as f:
        comparison = {
            'forward': metrics_forward,
            'reverse': metrics_reverse,
            'diff': {k: metrics_forward[k] - metrics_reverse[k] for k in metrics_forward.keys()}
        }
        json.dump(comparison, f, indent=2)
    
    return bidirectional_model, dataset, metrics_results


def analyze_feature_importance(model, dataset, direction_indicator=None, bidirectional=False):
    """
    Analyze feature importance by measuring how much each feature needs to change
    to generate valid counterfactuals
    """
    logger.info("Analyzing feature importance")
    
    X = np.vstack([dataset.X_train, dataset.X_test])
    y = np.concatenate([dataset.y_train, dataset.y_test])
    
    # Pick samples from the factual class
    factual_class = dataset.factual_class
    if bidirectional and direction_indicator == 1.0:
        factual_class = dataset.counterfactual_class
    
    factual_indices = np.where(y == factual_class)[0][:50]  # Select more samples for better statistics
    factual_points = dataset.feature_transformer.transform(X[factual_indices])
    
    # Generate counterfactuals
    generated_cfs = generate_counterfactuals(
        model=model,
        factual_points=factual_points,
        n_samples=10,
        temperature=0.8,
        device="cuda" if torch.cuda.is_available() else "cpu",
        direction_indicator=direction_indicator,
        bidirectional_model=bidirectional
    )
    
    # Calculate changes needed for each feature
    all_changes = []
    
    for i, (factual_idx, cf_samples) in enumerate(zip(factual_indices, generated_cfs)):
        factual = factual_points[i]
        
        # Handle different shapes of generated_cfs
        if cf_samples.ndim == 3:
            cf_samples = cf_samples[0]
        
        # Calculate changes for each generated counterfactual
        for cf in cf_samples:
            changes = cf - factual
            all_changes.append(changes)
    
    # Convert to numpy array
    all_changes = np.array(all_changes)
    
    # Calculate mean absolute change per feature
    mean_abs_changes = np.mean(np.abs(all_changes), axis=0)
    
    # Calculate normalized importance (sum to 1)
    feature_importance = mean_abs_changes / np.sum(mean_abs_changes)
    
    # Calculate standard deviation of changes
    std_changes = np.std(all_changes, axis=0)
    
    # Calculate mean direction of changes
    mean_direction = np.mean(all_changes, axis=0)
    
    return feature_importance, mean_direction, std_changes, all_changes


if __name__ == "__main__":
    # Parse command line arguments to determine which examples to run
    import argparse
    parser = argparse.ArgumentParser(description='Train counterfactual generative models on Blobs Dataset')
    parser.add_argument('--all', action='store_true', help='Run all examples')
    parser.add_argument('--unidirectional', action='store_true', help='Run unidirectional example')
    parser.add_argument('--bidirectional', action='store_true', help='Run bidirectional example')
    args = parser.parse_args()
    
    # If no specific arguments provided, run all examples
    if not (args.unidirectional or args.bidirectional):
        args.all = True
    
    all_metrics = {}
    
    # Run the selected examples
    if args.all or args.unidirectional:
        logger.info("\n=== Starting Blobs Dataset Unidirectional Example ===")
        uni_model, uni_dataset, uni_metrics = train_blobs_unidirectional()
        all_metrics['unidirectional'] = uni_metrics
        logger.info("Blobs unidirectional example completed")
        
        # Analyze feature importance for unidirectional model
        if uni_model is not None:
            feature_names = [f"Feature {i+1}" for i in range(uni_dataset.X_train.shape[1])]
            feature_importance, mean_direction, std_changes, _ = analyze_feature_importance(
                uni_model, uni_dataset
            )
            
            logger.info("\n=== Feature Importance Analysis (Unidirectional) ===")
            for i, feature_name in enumerate(feature_names):
                logger.info(f"{feature_name}: {feature_importance[i]:.4f} importance, "
                            f"mean change: {mean_direction[i]:+.4f} ± {std_changes[i]:.4f}")
    
    if args.all or args.bidirectional:
        logger.info("\n=== Starting Blobs Dataset Bidirectional Example ===")
        bi_model, bi_dataset, bi_metrics = train_blobs_bidirectional()
        all_metrics['bidirectional'] = bi_metrics
        logger.info("Blobs bidirectional example completed")
        
        # Analyze feature importance for bidirectional model (both directions)
        if bi_model is not None:
            feature_names = [f"Feature {i+1}" for i in range(bi_dataset.X_train.shape[1])]
            
            # Analyze Class 0 → Class 1 direction
            logger.info("\n=== Feature Importance Analysis (Class 0 → Class 1) ===")
            feature_importance, mean_direction, std_changes, _ = analyze_feature_importance(
                bi_model, bi_dataset, direction_indicator=0.0, bidirectional=True
            )
            
            for i, feature_name in enumerate(feature_names):
                logger.info(f"{feature_name}: {feature_importance[i]:.4f} importance, "
                            f"mean change: {mean_direction[i]:+.4f} ± {std_changes[i]:.4f}")
            
            # Analyze Class 1 → Class 0 direction
            logger.info("\n=== Feature Importance Analysis (Class 1 → Class 0) ===")
            feature_importance, mean_direction, std_changes, _ = analyze_feature_importance(
                bi_model, bi_dataset, direction_indicator=1.0, bidirectional=True
            )
            
            for i, feature_name in enumerate(feature_names):
                logger.info(f"{feature_name}: {feature_importance[i]:.4f} importance, "
                            f"mean change: {mean_direction[i]:+.4f} ± {std_changes[i]:.4f}")
    
    # Output summary of all metrics
    if all_metrics:
        import json
        with open("results/blobs_all_metrics_summary.json", 'w') as f:
            json.dump(all_metrics, f, indent=2)
    
    logger.info("\nAll examples completed successfully!")
    logger.info("Results saved to:")
    if args.all or args.unidirectional:
        logger.info("  - results/blobs_unidirectional")
    if args.all or args.bidirectional:
        logger.info("  - results/blobs_bidirectional") 