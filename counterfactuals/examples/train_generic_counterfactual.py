import sys
sys.path.append("/home/z1172691/counterfactuals")

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
    visualize_multiclass_counterfactual_generation,
    visualize_single_factual_counterfactuals
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
# from counterfactuals.datasets.DCEF.adult import AdultDataset
from counterfactuals.datasets.generic_counterfactual import AbstractDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('counterfactual_example')


def prepare_dataset_and_models(
        dataset_class: AbstractDataset,
        save_dir: str,
        load_from_save_dir: bool = False
    ):
    dataset = dataset_class()
    disc_model = MultilayerPerceptron(
        input_size=dataset.X_train.shape[1],
        hidden_layer_sizes=[256, 256],
        target_size=np.unique(dataset.y_train).shape[0]
    )
    if load_from_save_dir:
        disc_model.load_state_dict(torch.load(os.path.join(save_dir, "disc_model.pth")))
    else:
        disc_model.fit(
            dataset.train_dataloader(64, True),
            dataset.test_dataloader(64, False),
            epochs=10000,
            lr=0.001,
            patience=100,
            checkpoint_path=os.path.join(save_dir, "disc_model.pth")
    )
    disc_model = disc_model.eval()
    y_train = disc_model.predict(dataset.X_train).numpy().astype(int)
    y_test = disc_model.predict(dataset.X_test).numpy().astype(int)
    logger.info(f"Discriminator model accuracy: {np.sum(y_test == dataset.y_test) / len(dataset.y_test)}")
    

    dataset.y_train = y_train
    dataset.y_test = y_test

    gen_model = MaskedAutoregressiveFlow(
        features=dataset.X_train.shape[1],
        hidden_features=16,
        num_layers=2,
        num_blocks_per_layer=2,
        context_features=1
    )
    if load_from_save_dir:
        gen_model.load_state_dict(torch.load(os.path.join(save_dir, "gen_model.pth")))
    else:
        gen_model.fit(
            dataset.train_dataloader(64, True, 0.03), 
            dataset.test_dataloader(64, False), 
            num_epochs=10000, 
            learning_rate=0.001, 
            patience=100,
            checkpoint_path=os.path.join(save_dir, "gen_model.pth")
        )
    gen_model = gen_model.eval()
    return dataset, disc_model, gen_model


def train_method(
        dataset_class: AbstractDataset = MoonsDataset,
        dataset_name: str = "Moons",
        save_dir: str = "results/moons_multiclass",
        prob_threshold: float = 0.99,
        n_nearest: int = 16
):
    """
    Example using the moons dataset with multiclass counterfactual generation
    """
    logger.info("Starting moons dataset example (multiclass)")

    os.makedirs(save_dir, exist_ok=True)
    
    dataset, disc_model, gen_model = prepare_dataset_and_models(dataset_class, save_dir, load_from_save_dir=True)
    
    # Visualize the dataset
    visualize_dataset(
        dataset.X_train, dataset.y_train, 
        title=f"{dataset_name} Dataset (Multiclass)", 
        save_path=os.path.join(save_dir, f"{dataset_name}_dataset.png")
    )
    logger.info(f"Saved dataset visualization to {os.path.join(save_dir, f'{dataset_name}_dataset.png')}")

    # Select threshold for classifier as median of the posterior probabilities
    logger.info(f"Selected threshold for classifier: {prob_threshold}")
    
    # Create masks
    features = dataset.X_train.shape[1]
    masks = []
    for i in range(2):
        mask = np.ones(features) * 1e-2

        idx = np.random.choice(len(mask), size=int(features // 5) + 1, replace=False)
        mask[idx] = 1

        masks.append(mask)
    masks.append(np.ones(features) * 1e-2)
    masks = np.array(masks)

    p_values = [1e-2, 2.0]
    noise_level = 0.03

    # Create the multiclass counterfactual wrapper
    logger.info("Creating multiclass counterfactual dataset wrapper")
    dataset = MulticlassCounterfactualWrapper(
        X=dataset.X_train,
        y=dataset.y_train,
        factual_classes=np.unique(dataset.y_train),  # Use all classes as factual
        p_values=p_values,
        masks=masks,
        n_nearest=n_nearest,
        noise_level=noise_level,
        classifier=disc_model,
        prob_threshold=prob_threshold,
        log_level='INFO'
    )
    
    # Train a multiclass model
    logger.info("Training multiclass model")
    multiclass_model = train_multiclass_counterfactual_flow_model(
        dataset=dataset,
        flow_model_class=MaskedAutoregressiveFlow,
        hidden_features=64,
        num_layers=5,
        num_blocks_per_layer=2,
        learning_rate=1e-3,
        batch_size=256,
        num_epochs=10000,
        patience=50,
        noise_level=noise_level,
        save_dir=os.path.join(save_dir, "multiclass_model"),
        log_interval=10,
        balanced=True,  # Ensure balanced representation of classes in batches
        load_from_save_dir=True
    )
    logger.info("Multiclass model training complete")

    # visualise factual points
    mask = np.array([0.01, 0.01])
    p_value = 2.0
    visualize_single_factual_counterfactuals(
        model=multiclass_model,
        dataset=dataset,
        disc_model=disc_model,
        factual_point=dataset.X_train[0],
        factual_class=dataset.y_train[0],
        target_class=dataset.y_train[1],
        mask=mask,
        p_value=p_value,
        num_samples=20,
        temperature=0.8,
        save_path=os.path.join(save_dir, f"factual_point_class_{dataset.y_train[0]}_to_class_{dataset.y_train[1]}_mask_{mask}_p_{p_value}.pdf")
    )

    mask = np.array([0.01, 0.01])
    p_value = 0.01
    visualize_single_factual_counterfactuals(
        model=multiclass_model,
        dataset=dataset,
        disc_model=disc_model,
        factual_point=dataset.X_train[0],
        factual_class=dataset.y_train[0],
        target_class=dataset.y_train[1],
        mask=mask,
        p_value=p_value,
        num_samples=20,
        temperature=0.8,
        save_path=os.path.join(save_dir, f"factual_point_class_{dataset.y_train[0]}_to_class_{dataset.y_train[1]}_mask_{mask}_p_{p_value}.pdf")
    )

    mask = np.array([0.01, 1.0])
    p_value = 2.0
    visualize_single_factual_counterfactuals(
        model=multiclass_model,
        dataset=dataset,
        disc_model=disc_model,
        factual_point=dataset.X_train[0],
        factual_class=dataset.y_train[0],
        target_class=dataset.y_train[1],
        mask=mask,
        p_value=p_value,
        num_samples=20,
        temperature=0.8,
        save_path=os.path.join(save_dir, f"factual_point_class_{dataset.y_train[0]}_to_class_{dataset.y_train[1]}_mask_{mask}_p_{p_value}.pdf")
    )

    mask = np.array([1.0, 0.01])
    p_value = 2.0
    visualize_single_factual_counterfactuals(
        model=multiclass_model,
        dataset=dataset,
        disc_model=disc_model,
        factual_point=dataset.X_train[0],
        factual_class=dataset.y_train[0],
        target_class=dataset.y_train[1],
        mask=mask,
        p_value=p_value,
        num_samples=20,
        temperature=0.8,
        save_path=os.path.join(save_dir, f"factual_point_class_{dataset.y_train[0]}_to_class_{dataset.y_train[1]}_mask_{mask}_p_{p_value}.pdf")
    )
    return
    
    # Generate and visualize counterfactuals using multiclass model
    logger.info("Generating and visualizing counterfactuals using multiclass model")
    cf_vis_dir = os.path.join(save_dir, "counterfactual_visualization")
    os.makedirs(cf_vis_dir, exist_ok=True)

    if dataset.X.shape[1] == 2:
        for mask, p_value in zip(masks, p_values):
            visualize_multiclass_counterfactual_generation(
                model=multiclass_model,
                dataset=dataset,
                disc_model=disc_model,
                masks=masks,
                p_values=p_values,
                num_factual=6,
                num_samples=40,
                temperature=0.8,
                save_dir=cf_vis_dir
            )
        logger.info(f"Saved multiclass counterfactual visualizations to {cf_vis_dir}")

    # Generate counterfactuals for evaluation
    logger.info("Generating counterfactuals for evaluation")
    
    metrics_all = {}
    for mask_idx, mask in enumerate(masks):
        logger.info(f"Generating counterfactuals for mask {mask}")
        metrics_all[mask_idx] = {}
        for p_value in p_values:
            logger.info(f"Generating counterfactuals for p-norm {p_value}")
            metrics_all[mask_idx][p_value] = {}
            metrics_results = metrics_all[mask_idx][p_value]

            for factual_class in dataset.factual_classes:
                logger.info(f"Generating counterfactuals for factual class {factual_class}")
                factual_indices = np.where(dataset.y_train == factual_class)[0]
                factual_points = dataset.X_train[factual_indices]
                
                for target_class in dataset.classes:
                    if target_class == factual_class:
                        continue
                    
                    logger.info(f"Generating counterfactuals for factual class {factual_class} to target class {target_class}") 
                    generated_cfs = generate_multiclass_counterfactuals(
                        model=multiclass_model,
                        factual_points=factual_points,
                        target_class=target_class,
                        p_value=p_value,
                        mask=mask,
                        n_samples=100,
                        temperature=0.8,
                        device="cuda" if torch.cuda.is_available() else "cpu",
                        num_classes=len(dataset.classes)
                    )
                    np.save(
                        os.path.join(
                            save_dir,
                            f"generated_cfs_{factual_class}_to_class_{target_class}_p_{p_value}_mask_{mask}.npy"),
                            generated_cfs
                    )

                    action_mask = np.zeros_like(mask, dtype=bool)
                    action_mask[mask == 1.] = True
                    action_masks = []
                    
                    for gen_cf, fact_point in zip(generated_cfs, factual_points):
                        action_mask_cf =  np.broadcast_to(
                            action_mask[np.newaxis, :],
                            gen_cf.shape
                        )
                        broadcasted_fact = np.broadcast_to(
                            fact_point[np.newaxis, :], 
                            gen_cf.shape
                            )
                        gen_cf[action_mask_cf] = broadcasted_fact[action_mask_cf]

                        action_masks.append(action_mask_cf)
                    action_masks = np.concatenate(action_masks, axis=0)

                    metrics_forward, cfs_orig_forward = evaluate_counterfactuals(
                        disc_model=disc_model,
                        gen_model=gen_model,
                        dataset=dataset,
                        X=dataset.X_train,
                        y=dataset.y_train,
                        target_class=target_class,
                        factual_indices=factual_indices,
                        generated_cfs=generated_cfs,
                        p_value=p_value,
                        action_mask=action_masks,
                        direction=f'class_{factual_class}_to_class_{target_class}',
                        save_dir=save_dir
                    )
                    metrics_results[f'class_{factual_class}_to_class_{target_class}'] = metrics_forward

            # Save metrics comparison
            import json
            with open(os.path.join(save_dir, f"metrics_comparison_{p_value}_{action_mask}.json"), 'w') as f:
                json.dump(metrics_results, f, indent=2, default=str)
    
    return multiclass_model, dataset


if __name__ == "__main__":
    # Parse command line arguments to determine which examples to run
    import argparse
    parser = argparse.ArgumentParser(description='Train multiclass counterfactual generative models')
    parser.add_argument('--all', action='store_true', help='Run all examples')
    parser.add_argument('--moons', action='store_true', help='Run moons multiclass example')
    parser.add_argument('--three-class', action='store_true', help='Run three-class example')
    parser.add_argument('--law', action='store_true', help='Run law multiclass example')
    parser.add_argument('--heloc', action='store_true', help='Run HELOC multiclass example')
    parser.add_argument('--wine', action='store_true', help='Run Wine multiclass example')
    parser.add_argument('--adult', action='store_true', help='Run Adult multiclass example')
    parser.add_argument('--save_dir', default='results', help='Save directory path')
    args = parser.parse_args()
    
    # Run the selected examples
    if args.moons or args.all:
        logger.info("\n=== Starting Moons Multiclass Example ===")
        moons_model, moons_dataset = train_method(
            dataset_class=MoonsDataset,
            dataset_name="Moons",
            save_dir=f"{args.save_dir}/moons",
            prob_threshold=0.98,
            n_nearest=32
        )
        logger.info("Moons multiclass example completed")
        logger.info("Results saved to: results/moons")
    
    if args.three_class or args.all:
        logger.info("\n=== Starting Three-Class Example ===")
        three_class_model, three_class_dataset = train_method(
            dataset_class=BlobsDataset,
            dataset_name="Blobs",
            save_dir=f"{args.save_dir}/blobs",
            prob_threshold=0.98,
            n_nearest=32
        )
        logger.info("Three-class example completed")
    
    if args.law or args.all:
        logger.info("\n=== Starting Law Multiclass Example ===")
        law_model, law_dataset = train_method(
            dataset_class=LawDataset,
            dataset_name="Law",
            save_dir=f"{args.save_dir}/law",
            prob_threshold=0.55,
            n_nearest=32
        )
        logger.info("Law multiclass example completed")
    
    if args.heloc or args.all:
        logger.info("\n=== Starting HELOC Multiclass Example ===")
        heloc_model, heloc_dataset = train_method(
            dataset_class=HelocDataset,
            dataset_name="HELOC",
            save_dir=f"{args.save_dir}/heloc",
            prob_threshold=0.55,
            n_nearest=32
        )
        logger.info("HELOC multiclass example completed")
    
    if args.wine or args.all:
        logger.info("\n=== Starting Wine Multiclass Example ===")
        wine_model, wine_dataset = train_method(
            dataset_class=WineDataset,
            dataset_name="Wine",
            save_dir=f"{args.save_dir}/wine",
            prob_threshold=0.55,
            n_nearest=32
        )
        logger.info("Wine multiclass example completed")

    if args.adult or args.all:
        logger.info("\n=== Starting Adult Multiclass Example ===")
        adult_model, adult_dataset = train_method(
            dataset_class=AdultDataset,
            dataset_name="Adult",
            save_dir=f"{args.save_dir}/adult",
            prob_threshold=0.55,
            n_nearest=32
        )
        logger.info("Adult multiclass example completed")
    
    logger.info("\nAll examples completed successfully!")