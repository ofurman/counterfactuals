import numpy as np
import torch
import os
import json
import matplotlib.pyplot as plt
import logging

from counterfactuals.metrics.metrics import CFMetrics


logger = logging.getLogger('counterfactual_example')

def evaluate_counterfactuals(
    disc_model,
    gen_model,
    dataset, 
    X, 
    y, 
    factual_indices, 
    generated_cfs,
    direction="forward",
    save_dir=None,
    *,
    p_value,
    target_class = None,
    mask
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
        direction: 'forward' or 'reverse'
        save_dir: Directory to save results
    """
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
    if target_class is not None:
        y_target = target_class * np.ones_like(y_factual)
    else:
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

    #action_mask = np.ones_like(mask, dtype=bool)
    #action_mask[mask]
    
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
        prob_plausibility_threshold=threshold,
        action_mask=mask
    )
    
    # Calculate all metrics
    metrics_results = metrics.calc_all_metrics()
    
    # Log metrics
    for metric_name, metric_value in metrics_results.items():
        logger.info(f"{metric_name}: {metric_value:.4f}")
    
    # Save metrics to file
    if save_dir:
        metrics_file = os.path.join(save_dir, f"metrics_{direction}_{p_value}_{mask}.json")
        import json
        with open(metrics_file, 'w') as f:
            # ignore on error
            json.dump(metrics_results, f, indent=2, default=str)
    
    return metrics_results, generated_cfs


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
