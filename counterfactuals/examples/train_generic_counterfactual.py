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
from counterfactuals.discriminative_models.logistic_regression import (
    LogisticRegression, 
    MultinomialLogisticRegression,
)
from counterfactuals.discriminative_models.multilayer_perceptron import MultilayerPerceptron
from counterfactuals.datasets.heloc import HelocDataset
from counterfactuals.datasets.wine import WineDataset
from counterfactuals.datasets.moons import MoonsDataset
from counterfactuals.datasets.blobs import BlobsDataset
from counterfactuals.datasets.generic_counterfactual import AbstractDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('counterfactual_example')


def prepare_dataset_and_models(dataset_class: AbstractDataset):
    dataset = dataset_class()
    disc_model = MultilayerPerceptron(input_size=dataset.X_train.shape[1], hidden_layer_sizes=[256, 256], target_size=np.unique(dataset.y_train).shape[0])
    disc_model.fit(dataset.train_dataloader(64, True), dataset.test_dataloader(64, False), epochs=1000, lr=0.001, patience=100)
    disc_model = disc_model.eval()
    y_train = disc_model.predict(dataset.X_train).numpy().astype(int)
    y_test = disc_model.predict(dataset.X_test).numpy().astype(int)
    logger.info(f"Discriminator model accuracy: {np.sum(y_test == dataset.y_test) / len(dataset.y_test)}")
    

    dataset.y_train = y_train
    dataset.y_test = y_test

    gen_model = MaskedAutoregressiveFlow(features=dataset.X_train.shape[1], hidden_features=16, num_layers=2, num_blocks_per_layer=2, context_features=1)
    gen_model.fit(dataset.train_dataloader(64, True, 0.03), dataset.test_dataloader(64, False), num_epochs=1000, learning_rate=0.001, patience=100)
    return dataset, disc_model, gen_model


def train_moons_multiclass():
    """
    Example using the moons dataset with multiclass counterfactual generation
    """
    logger.info("Starting moons dataset example (multiclass)")
    
    dataset, disc_model, gen_model = prepare_dataset_and_models(MoonsDataset)

    # Set save directory
    save_dir = "results/moons_multiclass"
    os.makedirs(save_dir, exist_ok=True)
    
    # Visualize the dataset
    visualize_dataset(
        dataset.X_train, dataset.y_train, 
        title="Moons Dataset (Multiclass)", 
        save_path=os.path.join(save_dir, "moons_dataset.png")
    )
    logger.info(f"Saved dataset visualization to {os.path.join(save_dir, 'moons_dataset.png')}")
    
    # Create the multiclass counterfactual wrapper
    logger.info("Creating multiclass counterfactual dataset wrapper")
    dataset = MulticlassCounterfactualWrapper(
        X=dataset.X_train,
        y=dataset.y_train,
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
        num_epochs=200,
        patience=50,
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
        disc_model=disc_model,
        num_factual=6,
        num_samples=40,
        temperature=0.8,
        save_dir=cf_vis_dir
    )
    logger.info(f"Saved multiclass counterfactual visualizations to {cf_vis_dir}")

    # Generate counterfactuals for evaluation
    logger.info("Generating counterfactuals for evaluation")
    
    metrics_results = {}

    for factual_class in dataset.factual_classes:
        logger.info(f"Generating counterfactuals for factual class {factual_class}")
        factual_indices = np.where(dataset.y_train == factual_class)[0]
        factual_points = dataset.feature_transformer.transform(dataset.X_train[factual_indices])
        
        for target_class in dataset.classes:
            if target_class == factual_class:
                continue
            
            logger.info(f"Generating counterfactuals for factual class {factual_class} to target class {target_class}") 
            generated_cfs = generate_multiclass_counterfactuals(
                model=multiclass_model,
                factual_points=factual_points,
                target_class=target_class,
                n_samples=100,
                temperature=0.8,
                device="cuda" if torch.cuda.is_available() else "cpu",
                num_classes=len(dataset.classes)
            )

            metrics_forward, cfs_orig_forward = evaluate_counterfactuals(
                disc_model=disc_model,
                gen_model=gen_model,
                dataset=dataset,
                X=dataset.X_train,
                y=dataset.y_train,
                factual_indices=factual_indices,
                generated_cfs=generated_cfs,
                direction=f'class_{factual_class}_to_class_{target_class}',
                save_dir=save_dir
            )
            metrics_results[f'class_{factual_class}_to_class_{target_class}'] = metrics_forward
    
    return multiclass_model, dataset


def train_three_class_example():
    """
    Example using a three-class dataset with multiclass counterfactual generation
    """
    logger.info("Starting three-class dataset example")
    
    dataset, disc_model, gen_model = prepare_dataset_and_models(BlobsDataset)

    logger.info(f"Loaded three-class dataset with {len(dataset.X_train)} samples and {dataset.X_train.shape[1]} features")
    
    # Set save directory
    save_dir = "results/law_multiclass"
    os.makedirs(save_dir, exist_ok=True)
    
    # Set save directory
    save_dir = "results/three_class"
    os.makedirs(save_dir, exist_ok=True)
    
    # Visualize the dataset
    visualize_dataset(
        dataset.X_train, dataset.y_train, 
        title="Three-Class Dataset", 
        save_path=os.path.join(save_dir, "three_class_dataset.png")
    )
    logger.info(f"Saved dataset visualization to {os.path.join(save_dir, 'three_class_dataset.png')}")
    
    # Create the multiclass counterfactual wrapper
    logger.info("Creating multiclass counterfactual dataset wrapper")
    dataset = MulticlassCounterfactualWrapper(
        X=dataset.X_train,
        y=dataset.y_train,
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
        hidden_features=32,  # Larger model for more complex data
        num_layers=2,
        num_blocks_per_layer=2,
        learning_rate=1e-3,
        batch_size=None,
        num_epochs=200,
        patience=50,
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
        disc_model=disc_model,
        num_factual=5,
        num_samples=30,
        temperature=0.8,
        save_dir=cf_vis_dir
    )
    logger.info(f"Saved multiclass counterfactual visualizations to {cf_vis_dir}")


    # Generate counterfactuals for evaluation
    logger.info("Generating counterfactuals for evaluation")
    
    metrics_results = {}

    for factual_class in dataset.factual_classes:
        logger.info(f"Generating counterfactuals for factual class {factual_class}")
        factual_indices = np.where(dataset.y_train == factual_class)[0]
        factual_points = dataset.feature_transformer.transform(dataset.X_train[factual_indices])
        
        for target_class in dataset.classes:
            if target_class == factual_class:
                continue
            
            logger.info(f"Generating counterfactuals for factual class {factual_class} to target class {target_class}") 
            generated_cfs = generate_multiclass_counterfactuals(
                model=multiclass_model,
                factual_points=factual_points,
                target_class=target_class,
                n_samples=100,
                temperature=0.8,
                device="cuda" if torch.cuda.is_available() else "cpu",
                num_classes=len(dataset.classes)
            )

            metrics_forward, cfs_orig_forward = evaluate_counterfactuals(
                disc_model=disc_model,
                gen_model=gen_model,
                dataset=dataset,
                X=dataset.X_train,
                y=dataset.y_train,
                target_class=target_class,
                factual_indices=factual_indices,
                generated_cfs=generated_cfs,
                direction=f'class_{factual_class}_to_class_{target_class}',
                save_dir=save_dir
            )
            metrics_results[f'class_{factual_class}_to_class_{target_class}'] = metrics_forward
    
    return multiclass_model, dataset


def train_law_multiclass():
    """
    Example using the law dataset with multiclass counterfactual generation
    """
    logger.info("Starting law dataset example (multiclass)")
    
    dataset, disc_model, gen_model = prepare_dataset_and_models(LawDataset)
    # Set save directory
    save_dir = "results/law_multiclass"
    os.makedirs(save_dir, exist_ok=True)
    
    # Create the multiclass counterfactual wrapper
    logger.info("Creating multiclass counterfactual dataset wrapper")
    dataset = MulticlassCounterfactualWrapper(
        X=dataset.X_train,
        y=dataset.y_train,
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
        num_epochs=200,
        patience=50,
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
    factual_indices_fail = np.where(dataset.y_train == 0)[0][:20]
    factual_points_fail = dataset.feature_transformer.transform(dataset.X_train[factual_indices_fail])
    
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
        X=dataset.X_train,
        y=dataset.y_train,
        factual_indices=factual_indices_fail,
        generated_cfs=generated_cfs_fail_to_pass,
        direction="forward",
        save_dir=save_dir_fail_to_pass
    )
    metrics_results['forward'] = metrics_forward
    
    # 2. Passed → Failed
    logger.info("\n--- Generating counterfactuals: Passed → Failed ---")
    factual_indices_pass = np.where(dataset.y_train == 1)[0][:20]
    factual_points_pass = dataset.feature_transformer.transform(dataset.X_train[factual_indices_pass])
    
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
        X=dataset.X_train,
        y=dataset.y_train,
        factual_indices=factual_indices_pass,
        generated_cfs=generated_cfs_pass_to_fail,
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
    np.save(os.path.join(save_dir, "factual_points_fail.npy"), dataset.X_train[factual_indices_fail])
    np.save(os.path.join(save_dir, "factual_points_pass.npy"), dataset.X_train[factual_indices_pass])
    
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


def train_heloc_multiclass():
    """
    Example using the HELOC dataset with multiclass counterfactual generation
    """
    logger.info("Starting HELOC dataset example (multiclass)")
    
    dataset, disc_model, gen_model = prepare_dataset_and_models(HelocDataset)

    logger.info(f"Loaded HELOC dataset with {len(dataset.X_train)} samples and {dataset.X_train.shape[1]} features")
    
    # Set save directory
    save_dir = "results/heloc_multiclass"
    os.makedirs(save_dir, exist_ok=True)
    
    # Create the multiclass counterfactual wrapper
    logger.info("Creating multiclass counterfactual dataset wrapper")
    dataset = MulticlassCounterfactualWrapper(
        X=dataset.X_train,
        y=dataset.y_train,
        factual_classes=[0, 1],  # Use both classes as factual
        n_nearest=10,
        noise_level=0.03,
        log_level='INFO',
    )
    
    # Train a multiclass model
    logger.info("Training multiclass model for HELOC dataset")
    multiclass_model = train_multiclass_counterfactual_flow_model(
        dataset=dataset,
        flow_model_class=MaskedAutoregressiveFlow,
        hidden_features=32,
        num_layers=2,
        num_blocks_per_layer=2,
        learning_rate=1e-3,
        batch_size=None,
        num_epochs=200,
        patience=50,
        noise_level=0.03,
        save_dir=save_dir,
        log_interval=10,
        balanced=True  # Ensure balanced representation of classes in batches
    )
    logger.info("Multiclass model training complete")
    
    # Generate counterfactuals for evaluation
    logger.info("Generating counterfactuals for evaluation")
    
    metrics_results = {}
    
    # 1. Good Risk → Bad Risk
    logger.info("\n--- Generating counterfactuals: Good Risk → Bad Risk ---")
    factual_indices_good = np.where(dataset.y_train == 0)[0][:20]
    factual_points_good = dataset.feature_transformer.transform(dataset.X_train[factual_indices_good])
    
    generated_cfs_good_to_bad = generate_multiclass_counterfactuals(
        model=multiclass_model,
        factual_points=factual_points_good,
        target_class=1,  # Target class (bad risk)
        n_samples=100,
        temperature=0.8,
        device="cuda" if torch.cuda.is_available() else "cpu",
        num_classes=len(dataset.classes)
    )
    
    # Evaluate forward direction
    save_dir_good_to_bad = os.path.join(save_dir, "good_to_bad")
    os.makedirs(save_dir_good_to_bad, exist_ok=True)
    metrics_forward, cfs_orig_forward = evaluate_counterfactuals(
        disc_model=disc_model,
        gen_model=gen_model,
        dataset=dataset,
        X=dataset.X_train,
        y=dataset.y_train,
        factual_indices=factual_indices_good,
        generated_cfs=generated_cfs_good_to_bad,
        direction="forward",
        save_dir=save_dir_good_to_bad
    )
    metrics_results['forward'] = metrics_forward
    
    # 2. Bad Risk → Good Risk
    logger.info("\n--- Generating counterfactuals: Bad Risk → Good Risk ---")
    factual_indices_bad = np.where(dataset.y_train == 1)[0][:20]
    factual_points_bad = dataset.feature_transformer.transform(dataset.X_train[factual_indices_bad])
    
    generated_cfs_bad_to_good = generate_multiclass_counterfactuals(
        model=multiclass_model,
        factual_points=factual_points_bad,
        target_class=0,  # Target class (good risk)
        n_samples=100,
        temperature=0.8,
        device="cuda" if torch.cuda.is_available() else "cpu",
        num_classes=len(dataset.classes)
    )
    
    # Evaluate reverse direction
    save_dir_bad_to_good = os.path.join(save_dir, "bad_to_good")
    os.makedirs(save_dir_bad_to_good, exist_ok=True)
    metrics_reverse, cfs_orig_reverse = evaluate_counterfactuals(
        disc_model=disc_model,
        gen_model=gen_model,
        dataset=dataset,
        X=dataset.X_train,
        y=dataset.y_train,
        factual_indices=factual_indices_bad,
        generated_cfs=generated_cfs_bad_to_good,
        direction="reverse",
        save_dir=save_dir_bad_to_good
    )
    metrics_results['reverse'] = metrics_reverse
    
    # Compare metrics between directions
    logger.info("\n=== Metrics Comparison Between Directions ===")
    for metric in metrics_forward.keys():
        logger.info(f"{metric}: Forward={metrics_forward[metric]:.4f}, Reverse={metrics_reverse[metric]:.4f}, "
                   f"Diff={metrics_forward[metric]-metrics_reverse[metric]:+.4f}")
    
    # Save factual points and counterfactuals for further analysis
    np.save(os.path.join(save_dir, "factual_points_good.npy"), dataset.X_train[factual_indices_good])
    np.save(os.path.join(save_dir, "factual_points_bad.npy"), dataset.X_train[factual_indices_bad])
    
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


def train_wine_multiclass():
    """
    Example using the Wine dataset with multiclass counterfactual generation
    """
    logger.info("Starting Wine dataset example (multiclass)")
    
    dataset, disc_model, gen_model = prepare_dataset_and_models(WineDataset)
    
    logger.info(f"Loaded Wine dataset with {len(dataset.X_train)} samples and {dataset.X_train.shape[1]} features")
    logger.info(f"Class distribution: {np.unique(dataset.y_train, return_counts=True)}")
    
    # Set save directory
    save_dir = "results/wine_multiclass"
    os.makedirs(save_dir, exist_ok=True)
    
    # Create the multiclass counterfactual wrapper
    logger.info("Creating multiclass counterfactual dataset wrapper")
    dataset = MulticlassCounterfactualWrapper(
        X=dataset.X_train,
        y=dataset.y_train,
        factual_classes=list(np.unique(dataset.y_train)),  # Use all classes as factual
        n_nearest=10,
        noise_level=0.03,
        log_level='INFO',
    )
    
    # Train a multiclass model
    logger.info("Training multiclass model for Wine dataset")
    multiclass_model = train_multiclass_counterfactual_flow_model(
        dataset=dataset,
        flow_model_class=MaskedAutoregressiveFlow,
        hidden_features=32,
        num_layers=2,
        num_blocks_per_layer=2,
        learning_rate=1e-3,
        batch_size=None,
        num_epochs=200,
        patience=50,
        noise_level=0.03,
        save_dir=save_dir,
        log_interval=10,
        balanced=True  # Ensure balanced representation of classes in batches
    )
    logger.info("Multiclass model training complete")
    
    # Generate counterfactuals for evaluation
    logger.info("Generating counterfactuals for evaluation")
    
    metrics_results = {}
    
    # Generate counterfactuals for each class pair
    for factual_class in dataset.factual_classes:
        logger.info(f"\n--- Generating counterfactuals for factual class {factual_class} ---")
        factual_indices = np.where(dataset.y_train == factual_class)[0][:20]
        factual_points = dataset.feature_transformer.transform(dataset.X_train[factual_indices])
        
        for target_class in dataset.classes:
            if target_class == factual_class:
                continue
                
            logger.info(f"Generating counterfactuals: Class {factual_class} → Class {target_class}")
            generated_cfs = generate_multiclass_counterfactuals(
                model=multiclass_model,
                factual_points=factual_points,
                target_class=target_class,
                n_samples=100,
                temperature=0.8,
                device="cuda" if torch.cuda.is_available() else "cpu",
                num_classes=len(dataset.classes)
            )
            
            # Evaluate counterfactuals
            save_dir_class_pair = os.path.join(save_dir, f"class_{factual_class}_to_class_{target_class}")
            os.makedirs(save_dir_class_pair, exist_ok=True)
            metrics, cfs_orig = evaluate_counterfactuals(
                disc_model=disc_model,
                gen_model=gen_model,
                dataset=dataset,
                X=dataset.X_train,
                y=dataset.y_train,
                factual_indices=factual_indices,
                generated_cfs=generated_cfs,
                direction=f"class_{factual_class}_to_class_{target_class}",
                save_dir=save_dir_class_pair
            )
            metrics_results[f"class_{factual_class}_to_class_{target_class}"] = metrics
            
            # Save factual points and counterfactuals for further analysis
            np.save(os.path.join(save_dir_class_pair, f"factual_points_class_{factual_class}.npy"), dataset.X_train[factual_indices])
    
    # Save metrics comparison
    import json
    with open(os.path.join(save_dir, "metrics_comparison.json"), 'w') as f:
        json.dump(metrics_results, f, indent=2, default=str)
    
    return multiclass_model, dataset, metrics_results


if __name__ == "__main__":
    # Parse command line arguments to determine which examples to run
    import argparse
    parser = argparse.ArgumentParser(description='Train multiclass counterfactual generative models')
    parser.add_argument('--moons', action='store_true', help='Run moons multiclass example')
    parser.add_argument('--three-class', action='store_true', help='Run three-class example')
    parser.add_argument('--law', action='store_true', help='Run law multiclass example')
    parser.add_argument('--heloc', action='store_true', help='Run HELOC multiclass example')
    parser.add_argument('--wine', action='store_true', help='Run Wine multiclass example')
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
    
    if args.heloc:
        logger.info("\n=== Starting HELOC Multiclass Example ===")
        heloc_model, heloc_dataset, heloc_metrics = train_heloc_multiclass()
        logger.info("HELOC multiclass example completed")
    
    if args.wine:
        logger.info("\n=== Starting Wine Multiclass Example ===")
        wine_model, wine_dataset, wine_metrics = train_wine_multiclass()
        logger.info("Wine multiclass example completed")
    
    logger.info("\nAll examples completed successfully!")
    logger.info("Results saved to:")
    if args.moons:
        logger.info("  - results/moons_multiclass")
    if args.three_class:
        logger.info("  - results/three_class")
    if args.law:
        logger.info("  - results/law_multiclass")
    if args.heloc:
        logger.info("  - results/heloc_multiclass")
    if args.wine:
        logger.info("  - results/wine_multiclass") 