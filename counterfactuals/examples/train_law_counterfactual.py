import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import logging
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from counterfactuals.datasets.law import LawDataset
from counterfactuals.datasets.generic_counterfactual import (
    CounterfactualWrapper, 
    train_counterfactual_flow_model,
    generate_counterfactuals
)
from counterfactuals.generative_models.maf import MaskedAutoregressiveFlow
from counterfactuals.discriminative_models.logistic_regression import LogisticRegression
from counterfactuals.metrics.metrics import CFMetrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('law_counterfactual_example')


def visualize_dataset_projection(X, y, title="Dataset Projection", factual_class=0, counterfactual_class=1, 
                                save_path=None, projection_method='pca'):
    """
    Visualize a dataset projection to 2D
    """
    plt.figure(figsize=(10, 8))
    
    # Apply dimensionality reduction if needed
    if X.shape[1] > 2:
        if projection_method == 'pca':
            projection = PCA(n_components=2)
        else:  # Use t-SNE
            projection = TSNE(n_components=2, random_state=42)
        
        X_projected = projection.fit_transform(X)
    else:
        X_projected = X
    
    plt.scatter(
        X_projected[y == factual_class, 0], 
        X_projected[y == factual_class, 1], 
        color='blue', 
        alpha=0.6, 
        label=f'Class {factual_class} (Factual)'
    )
    plt.scatter(
        X_projected[y == counterfactual_class, 0], 
        X_projected[y == counterfactual_class, 1], 
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
    
    return plt, X_projected


def evaluate_counterfactuals(
    cf_model, 
    disc_model,
    gen_model,
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
    # 1. Print basic statistics about the changes needed
    logger.info("\n=== Basic Counterfactual Statistics ===")
    
    # For each factual point, show the original values and generated counterfactuals
    all_changes = []
    for i, (factual_idx, cf_samples) in enumerate(zip(factual_indices, generated_cfs)):
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
    
    # For plausibility threshold, estimate from training data log likelihoods
    with torch.no_grad():
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        log_probs = gen_model(X_train_tensor, y_train_tensor)
        threshold = torch.median(log_probs)
    
    # Create metrics calculator
    metrics = CFMetrics(
        X_cf=X_cf, 
        y_target=y_target_rep,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test_rep, 
        y_test=y_factual.repeat(cf_per_factual),
        gen_model=gen_model,
        disc_model=disc_model,
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
    
    return metrics_results, generated_cfs


def train_law_unidirectional():
    """
    Example using the law dataset with unidirectional counterfactual generation
    (failed bar exam → passed bar exam)
    """
    logger.info("Starting law dataset example (unidirectional)")
    
    # Load the law dataset
    law_dataset = LawDataset()
    X = np.vstack([law_dataset.X_train, law_dataset.X_test])
    y = np.concatenate([law_dataset.y_train, law_dataset.y_test])
    logger.info(f"Loaded law dataset with {len(X)} samples and {X.shape[1]} features")
    logger.info(f"Class distribution: Failed bar: {np.sum(y == 0)}, Passed bar: {np.sum(y == 1)}")
    
    # Set save directory
    save_dir = "results/law_unidirectional"
    os.makedirs(save_dir, exist_ok=True)
    
    # Define feature names
    feature_names = ["LSAT", "GPA", "First Year GPA"]
    
    # Create dataset projections for basic visualization
    logger.info("Creating dataset projections")
    plt, X_projected = visualize_dataset_projection(
        X, y, 
        title="Law Dataset (PCA Projection)", 
        factual_class=0,  # Failed bar exam
        counterfactual_class=1,  # Passed bar exam
        save_path=os.path.join(save_dir, "law_dataset_pca.png"),
        projection_method='pca'
    )
    
    # Create the counterfactual wrapper
    logger.info("Creating counterfactual dataset wrapper (unidirectional)")
    dataset = CounterfactualWrapper(
        X=X,
        y=y,
        factual_class=0,  # Failed bar exam
        counterfactual_class=1,  # Passed bar exam
        n_nearest=10,
        noise_level=0.03,
        log_level='INFO',
        bidirectional=False
    )
    
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
        num_epochs=500,
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
    
    # Select failed bar examples to generate counterfactuals for
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


def train_law_bidirectional():
    """
    Example using the law dataset with bidirectional counterfactual generation
    (failed bar exam ↔ passed bar exam)
    """
    logger.info("Starting law dataset example (bidirectional)")
    
    # Load the law dataset
    law_dataset = LawDataset()
    disc_model = LogisticRegression(input_size=law_dataset.X_train.shape[1], target_size=1)
    disc_model.fit(law_dataset.train_dataloader(64, True), law_dataset.test_dataloader(64, False), epochs=1000, lr=0.001, patience=100)
    law_dataset.y_train = disc_model.predict(law_dataset.X_train).numpy()
    law_dataset.y_test = disc_model.predict(law_dataset.X_test).numpy()

    gen_model = MaskedAutoregressiveFlow(features=law_dataset.X_train.shape[1], hidden_features=16, num_layers=2, num_blocks_per_layer=2, context_features=1)
    gen_model.fit(law_dataset.train_dataloader(64, True, 0.03), law_dataset.test_dataloader(64, False), num_epochs=1000, learning_rate=0.001, patience=100)

    X = np.vstack([law_dataset.X_train, law_dataset.X_test])
    y = np.concatenate([law_dataset.y_train, law_dataset.y_test])
    logger.info(f"Loaded law dataset with {len(X)} samples and {X.shape[1]} features")
    
    # Set save directory
    save_dir = "results/law_bidirectional"
    os.makedirs(save_dir, exist_ok=True)
    
    # Define feature names
    feature_names = ["LSAT", "GPA", "First Year GPA"]
    
    # Create the counterfactual wrapper with bidirectional=True
    logger.info("Creating counterfactual dataset wrapper (bidirectional)")
    dataset = CounterfactualWrapper(
        X=X,
        y=y,
        factual_class=0,  # Failed bar exam
        counterfactual_class=1,  # Passed bar exam
        n_nearest=10,
        noise_level=0.03,
        log_level='INFO',
        bidirectional=True  # Enable bidirectional mode
    )
    
    # Train a single unified bidirectional model
    logger.info("Training unified bidirectional model for law dataset")
    bidirectional_model = train_counterfactual_flow_model(
        dataset=dataset,
        flow_model_class=MaskedAutoregressiveFlow,
        hidden_features=32,
        num_layers=2,
        num_blocks_per_layer=2,
        learning_rate=1e-3,
        batch_size=None,
        num_epochs=500,
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
    
    # 1. Failed → Passed
    logger.info("\n--- Generating counterfactuals: Failed → Passed ---")
    factual_indices_fail = np.where(y == 0)[0][:20]
    factual_points_fail = dataset.feature_transformer.transform(X[factual_indices_fail])
    
    generated_cfs_fail_to_pass = generate_counterfactuals(
        model=bidirectional_model,
        factual_points=factual_points_fail,
        n_samples=10,
        temperature=0.8,
        device="cuda" if torch.cuda.is_available() else "cpu",
        direction_indicator=0.0,  # Forward direction
        bidirectional_model=True
    )
    
    # Evaluate forward direction
    save_dir = os.path.join(save_dir, "fail_to_pass")
    os.makedirs(save_dir, exist_ok=True)
    metrics_forward, cfs_orig_forward = evaluate_counterfactuals(
        cf_model=bidirectional_model,
        gen_model=gen_model,
        disc_model=disc_model,
        dataset=dataset,
        X=X,
        y=y,
        factual_indices=factual_indices_fail,
        generated_cfs=generated_cfs_fail_to_pass,
        feature_names=feature_names,
        direction="forward",
        save_dir=save_dir
    )
    metrics_results['forward'] = metrics_forward
    
    # 2. Passed → Failed
    logger.info("\n--- Generating counterfactuals: Passed → Failed ---")
    factual_indices_pass = np.where(y == 1)[0][:20]
    factual_points_pass = dataset.feature_transformer.transform(X[factual_indices_pass])
    
    generated_cfs_pass_to_fail = generate_counterfactuals(
        model=bidirectional_model,
        factual_points=factual_points_pass,
        n_samples=10,
        temperature=0.8,
        device="cuda" if torch.cuda.is_available() else "cpu",
        direction_indicator=1.0,  # Reverse direction
        bidirectional_model=True
    )
    
    # Evaluate reverse direction
    save_dir = os.path.join(save_dir, "pass_to_fail")
    os.makedirs(save_dir, exist_ok=True)
    metrics_reverse, cfs_orig_reverse = evaluate_counterfactuals(
        cf_model=bidirectional_model,
        disc_model=disc_model,
        gen_model=gen_model,
        dataset=dataset,
        X=X,
        y=y,
        factual_indices=factual_indices_pass,
        generated_cfs=generated_cfs_pass_to_fail,
        feature_names=feature_names,
        direction="reverse",
        save_dir=save_dir
    )
    metrics_results['reverse'] = metrics_reverse
    
    # Compare metrics between directions
    logger.info("\n=== Metrics Comparison Between Directions ===")
    for metric in metrics_forward.keys():
        logger.info(f"{metric}: Forward={metrics_forward[metric]:.4f}, Reverse={metrics_reverse[metric]:.4f}, "
                   f"Diff={metrics_forward[metric]-metrics_reverse[metric]:+.4f}")
    
    # Save factual points and counterfactuals for further analysis
    np.save(os.path.join(save_dir, "factual_points_fail.npy"), X[factual_indices_fail])
    np.save(os.path.join(save_dir, "factual_points_pass.npy"), X[factual_indices_pass])
    
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
    parser = argparse.ArgumentParser(description='Train counterfactual generative models on Law Dataset')
    parser.add_argument('--all', action='store_true', help='Run all examples')
    parser.add_argument('--unidirectional', action='store_true', help='Run unidirectional example')
    parser.add_argument('--bidirectional', action='store_true', help='Run bidirectional example')
    args = parser.parse_args()
    
    # If no specific arguments provided, run all examples
    if not (args.unidirectional or args.bidirectional):
        args.all = True
    
    all_metrics = {}
    
    # Run the selected examples
    # if args.all or args.unidirectional:
    #     logger.info("\n=== Starting Law Dataset Unidirectional Example ===")
    #     uni_model, uni_dataset, uni_metrics = train_law_unidirectional()
    #     all_metrics['unidirectional'] = uni_metrics
    #     logger.info("Law unidirectional example completed")
        
    #     # Analyze feature importance for unidirectional model
    #     if uni_model is not None:
    #         feature_names = ["LSAT", "GPA", "First Year GPA"]
    #         feature_importance, mean_direction, std_changes, _ = analyze_feature_importance(
    #             uni_model, uni_dataset
    #         )
            
    #         logger.info("\n=== Feature Importance Analysis (Unidirectional) ===")
    #         for i, feature_name in enumerate(feature_names):
    #             logger.info(f"{feature_name}: {feature_importance[i]:.4f} importance, "
    #                         f"mean change: {mean_direction[i]:+.4f} ± {std_changes[i]:.4f}")
    
    if args.all or args.bidirectional:
        logger.info("\n=== Starting Law Dataset Bidirectional Example ===")
        bi_model, bi_dataset, bi_metrics = train_law_bidirectional()
        all_metrics['bidirectional'] = bi_metrics
        logger.info("Law bidirectional example completed")
        
        # Analyze feature importance for bidirectional model (both directions)
        if bi_model is not None:
            feature_names = ["LSAT", "GPA", "First Year GPA"]
            
            # Analyze Failed → Passed direction
            logger.info("\n=== Feature Importance Analysis (Failed → Passed) ===")
            feature_importance, mean_direction, std_changes, _ = analyze_feature_importance(
                bi_model, bi_dataset, direction_indicator=0.0, bidirectional=True
            )
            
            for i, feature_name in enumerate(feature_names):
                logger.info(f"{feature_name}: {feature_importance[i]:.4f} importance, "
                            f"mean change: {mean_direction[i]:+.4f} ± {std_changes[i]:.4f}")
            
            # Analyze Passed → Failed direction
            logger.info("\n=== Feature Importance Analysis (Passed → Failed) ===")
            feature_importance, mean_direction, std_changes, _ = analyze_feature_importance(
                bi_model, bi_dataset, direction_indicator=1.0, bidirectional=True
            )
            
            for i, feature_name in enumerate(feature_names):
                logger.info(f"{feature_name}: {feature_importance[i]:.4f} importance, "
                            f"mean change: {mean_direction[i]:+.4f} ± {std_changes[i]:.4f}")
    
    # Output summary of all metrics
    if all_metrics:
        import json
        with open("results/all_metrics_summary.json", 'w') as f:
            json.dump(all_metrics, f, indent=2)
    
    logger.info("\nAll examples completed successfully!")
    logger.info("Results saved to:")
    if args.all or args.unidirectional:
        logger.info("  - results/law_unidirectional")
    if args.all or args.bidirectional:
        logger.info("  - results/law_bidirectional") 