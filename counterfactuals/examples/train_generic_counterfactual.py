import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.datasets import make_moons, make_circles, make_classification
import os
import logging

from counterfactuals.datasets.generic_counterfactual import (
    MulticlassCounterfactualWrapper, 
    train_multiclass_counterfactual_flow_model,
    generate_multiclass_counterfactuals,
    visualize_multiclass_counterfactual_generation
)
from counterfactuals.generative_models.maf import MaskedAutoregressiveFlow
from counterfactuals.examples.utils import (
    visualize_dataset,
    evaluate_counterfactuals
)
from counterfactuals.datasets.law import LawDataset
from counterfactuals.discriminative_models.logistic_regression import LogisticRegression


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('counterfactual_example')


def train_moons_multiclass():
    """
    Example using the moons dataset with multiclass counterfactual generation
    """
    logger.info("Starting moons dataset example (multiclass)")
    
    # Generate the moons dataset
    X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)
    logger.info(f"Generated moons dataset with {len(X)} samples")
    logger.info(f"Class distribution: Class 0: {np.sum(y == 0)}, Class 1: {np.sum(y == 1)}")
    
    # Set save directory
    save_dir = "results/moons_multiclass"
    os.makedirs(save_dir, exist_ok=True)
    
    # Visualize the dataset
    visualize_dataset(
        X, y, 
        title="Moons Dataset (Multiclass)", 
        save_path=os.path.join(save_dir, "moons_dataset.png")
    )
    logger.info(f"Saved dataset visualization to {os.path.join(save_dir, 'moons_dataset.png')}")
    
    # Create the multiclass counterfactual wrapper
    logger.info("Creating multiclass counterfactual dataset wrapper")
    dataset = MulticlassCounterfactualWrapper(
        X=X,
        y=y,
        factual_classes=[0, 1],  # Use both classes as factual
        n_nearest=8,
        noise_level=0.03,
        log_level='INFO'
    )
    
    # Train a multiclass model
    logger.info("Training multiclass model")
    multiclass_model = train_multiclass_counterfactual_flow_model(
        dataset=dataset,
        flow_model_class=MaskedAutoregressiveFlow,
        hidden_features=32,
        num_layers=2,
        num_blocks_per_layer=2,
        learning_rate=1e-3,
        batch_size=None,
        num_epochs=10,
        patience=100,
        noise_level=0.03,
        save_dir=os.path.join(save_dir, "multiclass_model"),
        log_interval=10,
        balanced=True  # Ensure balanced representation of classes in batches
    )
    logger.info("Multiclass model training complete")
    
    # Generate and visualize counterfactuals using multiclass model
    logger.info("Generating and visualizing counterfactuals using multiclass model")
    cf_vis_dir = os.path.join(save_dir, "counterfactual_visualization")
    os.makedirs(cf_vis_dir, exist_ok=True)
    
    visualize_multiclass_counterfactual_generation(
        model=multiclass_model,
        dataset=dataset,
        num_factual=6,
        num_samples=40,
        temperature=0.8,
        save_dir=cf_vis_dir
    )
    logger.info(f"Saved multiclass counterfactual visualizations to {cf_vis_dir}")
    
    return multiclass_model, dataset


def train_three_class_example():
    """
    Example using a three-class dataset with multiclass counterfactual generation
    """
    logger.info("Starting three-class dataset example")
    
    # Generate a three-class dataset
    X, y = make_classification(
        n_samples=1500,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        n_classes=3,
        random_state=42
    )
    logger.info(f"Generated three-class dataset with {len(X)} samples")
    logger.info(f"Class distribution: Class 0: {np.sum(y == 0)}, Class 1: {np.sum(y == 1)}, Class 2: {np.sum(y == 2)}")
    
    # Set save directory
    save_dir = "results/three_class"
    os.makedirs(save_dir, exist_ok=True)
    
    # Visualize the dataset
    visualize_dataset(
        X, y, 
        title="Three-Class Dataset", 
        save_path=os.path.join(save_dir, "three_class_dataset.png")
    )
    logger.info(f"Saved dataset visualization to {os.path.join(save_dir, 'three_class_dataset.png')}")
    
    # Create the multiclass counterfactual wrapper
    logger.info("Creating multiclass counterfactual dataset wrapper")
    dataset = MulticlassCounterfactualWrapper(
        X=X,
        y=y,
        factual_classes=[0, 1, 2],  # Use all classes as factual
        n_nearest=8,
        noise_level=0.03,
        log_level='INFO'
    )
    
    # Train a multiclass model
    logger.info("Training multiclass model")
    multiclass_model = train_multiclass_counterfactual_flow_model(
        dataset=dataset,
        flow_model_class=MaskedAutoregressiveFlow,
        hidden_features=100,  # Larger model for more complex data
        num_layers=3,
        num_blocks_per_layer=2,
        learning_rate=1e-3,
        batch_size=None,
        num_epochs=500,
        patience=100,
        noise_level=0.03,
        save_dir=os.path.join(save_dir, "multiclass_model"),
        log_interval=10,
        balanced=True  # Ensure balanced representation of classes in batches
    )
    logger.info("Multiclass model training complete")
    
    # Generate and visualize counterfactuals using multiclass model
    logger.info("Generating and visualizing counterfactuals using multiclass model")
    cf_vis_dir = os.path.join(save_dir, "counterfactual_visualization")
    os.makedirs(cf_vis_dir, exist_ok=True)
    
    visualize_multiclass_counterfactual_generation(
        model=multiclass_model,
        dataset=dataset,
        num_factual=5,
        num_samples=30,
        temperature=0.8,
        save_dir=cf_vis_dir
    )
    logger.info(f"Saved multiclass counterfactual visualizations to {cf_vis_dir}")
    
    return multiclass_model, dataset


def train_law_multiclass():
    """
    Example using the law dataset with multiclass counterfactual generation
    """
    logger.info("Starting law dataset example (multiclass)")
    
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
    save_dir = "results/law_multiclass"
    os.makedirs(save_dir, exist_ok=True)
    
    # Define feature names
    feature_names = ["LSAT", "GPA", "First Year GPA"]
    
    # Create the multiclass counterfactual wrapper
    logger.info("Creating multiclass counterfactual dataset wrapper")
    dataset = MulticlassCounterfactualWrapper(
        X=X,
        y=y,
        factual_classes=[0, 1],  # Use both classes as factual
        n_nearest=10,
        noise_level=0.03,
        log_level='INFO',
    )
    
    # Train a multiclass model
    logger.info("Training multiclass model for law dataset")
    multiclass_model = train_multiclass_counterfactual_flow_model(
        dataset=dataset,
        flow_model_class=MaskedAutoregressiveFlow,
        hidden_features=32,
        num_layers=2,
        num_blocks_per_layer=2,
        learning_rate=1e-3,
        batch_size=None,
        num_epochs=10,
        patience=100,
        noise_level=0.03,
        save_dir=save_dir,
        log_interval=10,
        balanced=True  # Ensure balanced representation of classes in batches
    )
    logger.info("Multiclass model training complete")
    
    # Generate counterfactuals for evaluation
    logger.info("Generating counterfactuals for evaluation")
    
    metrics_results = {}
    
    # 1. Failed → Passed
    logger.info("\n--- Generating counterfactuals: Failed → Passed ---")
    factual_indices_fail = np.where(y == 0)[0][:20]
    factual_points_fail = dataset.feature_transformer.transform(X[factual_indices_fail])
    
    generated_cfs_fail_to_pass = generate_multiclass_counterfactuals(
        model=multiclass_model,
        factual_points=factual_points_fail,
        target_class=1,  # Target class (passed)
        n_samples=100,
        temperature=0.8,
        device="cuda" if torch.cuda.is_available() else "cpu",
        num_classes=len(dataset.classes)
    )
    
    # Evaluate forward direction
    save_dir_fail_to_pass = os.path.join(save_dir, "fail_to_pass")
    os.makedirs(save_dir_fail_to_pass, exist_ok=True)
    metrics_forward, cfs_orig_forward = evaluate_counterfactuals(
        disc_model=disc_model,
        gen_model=gen_model,
        dataset=dataset,
        X=X,
        y=y,
        factual_indices=factual_indices_fail,
        generated_cfs=generated_cfs_fail_to_pass,
        feature_names=feature_names,
        direction="forward",
        save_dir=save_dir_fail_to_pass
    )
    metrics_results['forward'] = metrics_forward
    
    # 2. Passed → Failed
    logger.info("\n--- Generating counterfactuals: Passed → Failed ---")
    factual_indices_pass = np.where(y == 1)[0][:20]
    factual_points_pass = dataset.feature_transformer.transform(X[factual_indices_pass])
    
    generated_cfs_pass_to_fail = generate_multiclass_counterfactuals(
        model=multiclass_model,
        factual_points=factual_points_pass,
        target_class=0,  # Target class (failed)
        n_samples=100,
        temperature=0.8,
        device="cuda" if torch.cuda.is_available() else "cpu",
        num_classes=len(dataset.classes)
    )
    
    # Evaluate reverse direction
    save_dir_pass_to_fail = os.path.join(save_dir, "pass_to_fail")
    os.makedirs(save_dir_pass_to_fail, exist_ok=True)
    metrics_reverse, cfs_orig_reverse = evaluate_counterfactuals(
        disc_model=disc_model,
        gen_model=gen_model,
        dataset=dataset,
        X=X,
        y=y,
        factual_indices=factual_indices_pass,
        generated_cfs=generated_cfs_pass_to_fail,
        feature_names=feature_names,
        direction="reverse",
        save_dir=save_dir_pass_to_fail
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
        json.dump(comparison, f, indent=2, default=str)
    
    return multiclass_model, dataset, metrics_results


if __name__ == "__main__":
    # Parse command line arguments to determine which examples to run
    import argparse
    parser = argparse.ArgumentParser(description='Train multiclass counterfactual generative models')
    parser.add_argument('--moons', action='store_true', help='Run moons multiclass example')
    parser.add_argument('--three-class', action='store_true', help='Run three-class example')
    parser.add_argument('--law', action='store_true', help='Run law multiclass example')
    args = parser.parse_args()
    
    # Run the selected examples
    if args.moons:
        logger.info("\n=== Starting Moons Multiclass Example ===")
        moons_model, moons_dataset = train_moons_multiclass()
        logger.info("Moons multiclass example completed")
    
    if args.three_class:
        logger.info("\n=== Starting Three-Class Example ===")
        three_class_model, three_class_dataset = train_three_class_example()
        logger.info("Three-class example completed")
    
    if args.law:
        logger.info("\n=== Starting Law Multiclass Example ===")
        law_model, law_dataset, law_metrics = train_law_multiclass()
        logger.info("Law multiclass example completed")
    
    logger.info("\nAll examples completed successfully!")
    logger.info("Results saved to:")
    if args.moons:
        logger.info("  - results/moons_multiclass")
    if args.three_class:
        logger.info("  - results/three_class")
    if args.law:
        logger.info("  - results/law_multiclass") 