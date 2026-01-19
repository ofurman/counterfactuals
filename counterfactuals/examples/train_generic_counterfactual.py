import sys

import pandas as pd

sys.path.append(r"C:\Users\marsz\Studies\ML-papers\DiCoFlex\counterfactuals")
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
from counterfactuals.datasets.DCENF.adult import AdultDataset
from counterfactuals.datasets import BankDataset, GMCDataset, LendingClubDataset, DefaultDataset
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
        D=dataset.X_train_clf.shape[1],
        weights_path=os.path.join(save_dir, "model.pt")
    )
    #if load_from_save_dir:
    #    disc_model.load_state_dict(torch.load(os.path.join(save_dir, "disc_model.pth")))
    #else:
    #    disc_model.fit(
    #        dataset.train_dataloader(64, True),
    #        dataset.test_dataloader(64, False),
    #        epochs=10000,
    #        lr=0.001,
    #        patience=100,
    #        checkpoint_path=os.path.join(save_dir, "disc_model.pth")
    #)
    disc_model.eval()
    y_train = disc_model.predict(dataset.X_train_clf).astype(int)
    y_test = disc_model.predict(dataset.X_test_clf).astype(int)
    logger.info(f"Discriminator model accuracy: {np.sum(y_test == dataset.y_test) / len(dataset.y_test)}")

    dataset.y_train = y_train
    dataset.y_test = y_test

    # Train set
    print("Train true counts:", np.bincount(y_train))
    #print("Train predicted counts:", np.bincount(y_pred_train))

    # Test set
    print("Test true counts:", np.bincount(y_test))
    #print("Test predicted counts:", np.bincount(y_pred_test))

    gen_model = MaskedAutoregressiveFlow(
        features=dataset.X_train.shape[1],
        hidden_features=16,
        num_layers=2,
        num_blocks_per_layer=2,
        context_features=1
    )
    #if load_from_save_dir:
    #    gen_model.load_state_dict(torch.load(os.path.join(save_dir, "gen_model.pth")))
    #else:
    #    gen_model.fit(
    #        dataset.train_dataloader(64, True, 0.03), 
    #        dataset.test_dataloader(64, False), 
    #        num_epochs=10000, 
    #        learning_rate=0.001, 
    #        patience=100,
    #        checkpoint_path=os.path.join(save_dir, "gen_model.pth")
    #    )
    gen_model = gen_model.eval()
    #return dataset, disc_model, gen_model

    return dataset, None, gen_model


def inverse_transform_data(data, dataset):
    data[:, dataset.categorical_features] = dataset.qt.inverse_transform(data[:, dataset.categorical_features])
    data_orig = np.empty((
        len(data), len(dataset.numerical_columns) + len(dataset.categorical_columns)
    ), dtype=object)

    numerical_pos = len(dataset.numerical_columns)
    numerical_indexes = [
        dataset.train_data.columns.get_loc(feat) for feat in dataset.feature_columns[:numerical_pos]
    ]
    data_orig[:, numerical_indexes] = (
        dataset.feature_transformer.named_transformers_["MinMaxScaler"].inverse_transform(
            data[:, dataset.numerical_features])
    )

    categorical_indexes = [
        dataset.train_data.columns.get_loc(feat) for feat in dataset.feature_columns[numerical_pos:]
    ]
    data_orig[:, categorical_indexes] = (
        dataset.feature_transformer.named_transformers_["OneHotEncoder"].inverse_transform(
            data[:, dataset.categorical_features]
        )
    )

    return data_orig

def train_method(
        dataset_class: AbstractDataset = MoonsDataset,
        dataset_name: str = "Moons",
        save_dir: str = "results/moons_multiclass",
        prob_threshold: float = 0.99,
        n_nearest: int = 16,
        data_dir = "data/moons_multiclass",
        load_from_save_dir: bool = False
):
    """
    Example using the moons dataset with multiclass counterfactual generation
    """
    logger.info("Starting moons dataset example (multiclass)")
    np.random.seed(0)

    os.makedirs(save_dir, exist_ok=True)
    
    dataset, disc_model, gen_model = prepare_dataset_and_models(dataset_class, data_dir, load_from_save_dir=False)
    
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
    masks.append(np.ones(features) * 1e-3)
    masks = np.array(masks)

    mask_features = len(masks)
    print(masks)

    p_values = [1e-2, 2.0]
    #p_values = [2.0]
    noise_level = 0.01

    # Create the multiclass counterfactual wrapper
    logger.info("Creating multiclass counterfactual dataset wrapper")
    dataset_cf = MulticlassCounterfactualWrapper(
        X=dataset.X_train,
        y=dataset.y_train,
        factual_classes=np.unique(dataset.y_train),  # Use all classes as factual
        p_values=p_values,
        masks=masks,
        n_nearest=n_nearest,
        noise_level=noise_level,
        classifier=disc_model,
        prob_threshold=prob_threshold,
        log_level='INFO',
        numerical_pos=len(dataset.numerical_features)
    )
    
    # Train a multiclass model
    logger.info("Training multiclass model")
    multiclass_model = train_multiclass_counterfactual_flow_model(
        dataset=dataset_cf,
        flow_model_class=MaskedAutoregressiveFlow,
        mask_features=mask_features,
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
        load_from_save_dir=False
    )
    logger.info("Multiclass model training complete")
    
    # Generate and visualize counterfactuals using multiclass model
    logger.info("Generating and visualizing counterfactuals using multiclass model")
    cf_vis_dir = os.path.join(save_dir, "counterfactual_visualization")
    os.makedirs(cf_vis_dir, exist_ok=True)

    if dataset_cf.X.shape[1] == 2:
        #for mask, p_value in zip(masks, p_values):
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

            for factual_class in dataset_cf.factual_classes:
                logger.info(f"Generating counterfactuals for factual class {factual_class}")
                factual_indices = np.where(dataset.y_test == factual_class)[0]
                factual_points = dataset.X_test[factual_indices]
                
                for target_class in dataset_cf.classes:
                    if target_class == factual_class:
                        continue
                    
                    logger.info(f"Generating counterfactuals for factual class {factual_class} to target class {target_class}")
                    mask_ohe = np.zeros(mask_features)
                    mask_ohe[mask_idx] = 1
                    print(mask_ohe)
                    generated_cfs, log_probs = generate_multiclass_counterfactuals(
                        model=multiclass_model,
                        factual_points=factual_points,
                        target_class=target_class,
                        p_value=p_value,
                        mask=mask_ohe,
                        n_samples=100,
                        temperature=0.8,
                        device="cuda" if torch.cuda.is_available() else "cpu",
                        num_classes=len(dataset_cf.classes)
                    )
                    print("Log probs shape: ", log_probs.shape)

                    action_mask = np.zeros_like(mask, dtype=bool)
                    action_mask[mask == 1.] = True
                    action_masks = []

                    # Replace counterfactuals containing NaNs with randomly chosen ones
                    new_generated_cfs = []
                    for gen_cf, fact_point in zip(generated_cfs, factual_points):
                        nan_mask = np.isnan(gen_cf)
                        if not np.all(nan_mask):    
                            while np.any(nan_mask):
                                nan_mask = np.isnan(gen_cf)
                                nan_mask_idx = np.any(nan_mask, axis=1)
                                idx_to_fix = np.where(nan_mask_idx)[0]
                                chosen_rand = np.random.choice(range(len(gen_cf)), size=len(idx_to_fix), replace=True)
                                gen_cf[nan_mask_idx] = gen_cf[chosen_rand]

                        action_mask_cf = np.broadcast_to(
                            action_mask[np.newaxis, :],
                            gen_cf.shape
                        )

                        action_masks.append(action_mask_cf)
                        new_generated_cfs.append(gen_cf)

                    action_masks = np.concatenate(action_masks, axis=0)
                    generated_cfs = np.array(new_generated_cfs)

                    print(generated_cfs.shape)
                    print(np.isnan(generated_cfs).sum() / generated_cfs.shape[2])

                    s, b, f = generated_cfs.shape
                    gen_rev = generated_cfs.copy().reshape(s*b, f)
                    gen_rev_orig = inverse_transform_data(gen_rev, dataset)

                    factual_points_orig = inverse_transform_data(factual_points.copy(), dataset)

                    true_columns = [col for col in dataset.train_data.columns if col in dataset.feature_columns]

                    path = f"generated_cfs_{factual_class}_to_class_{target_class}_p_{p_value}_mask_{mask_ohe}.csv"
                    path = path.replace('\n', '')
                    path = path.replace(' ', '')
                    np.savetxt(
                        os.path.join(
                            save_dir,
                            path),
                        gen_rev_orig,
                        delimiter=",",
                        header=','.join(true_columns),
                        comments="",
                        fmt="%s"
                    )
                    path = path.replace('generated_cfs', 'factual_points')
                    np.savetxt(
                        os.path.join(
                            save_dir,
                            path),
                        factual_points_orig,
                        delimiter=",",
                        header=','.join(true_columns),
                        comments="",
                        fmt="%s"
                    )

                    path = path.replace('factual_points', 'log_probs')
                    np.save(
                        os.path.join(
                            save_dir,
                            path),
                        log_probs
                    )

                    #np.save(
                    #    os.path.join(
                    #        save_dir,
                    #        path),
                    #        gen_rev_orig
                    #)

                    #metrics_forward, cfs_orig_forward = evaluate_counterfactuals(
                    #    disc_model=disc_model,
                    #    gen_model=gen_model,
                    #    dataset=dataset,
                    #    X=dataset.X_test,
                    #    y=dataset.y_test,
                    #    target_class=target_class,
                    #    factual_indices=factual_indices,
                    #    generated_cfs=generated_cfs,
                    #    p_value=p_value,
                    #    action_mask=action_masks,
                    #    direction=f'class_{factual_class}_to_class_{target_class}',
                    #)
                    #metrics_results[f'class_{factual_class}_to_class_{target_class}'] = metrics_forward

            # Save metrics comparison
            import json
            metric_path = f"metrics_comparison_{p_value}_{mask_ohe}.json"
            metric_path = metric_path.replace('\n', '')
            metric_path = metric_path.replace(' ', '')
            with open(os.path.join(save_dir, metric_path), 'w') as f:
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
    parser.add_argument('--bank', action='store_true', help='Run Bank multiclass example')
    parser.add_argument('--gmc', action='store_true', help='Run GMC multiclass example')
    parser.add_argument('--lending', action='store_true', help='Run Lending Club multiclass example')
    parser.add_argument('--default', action='store_true', help='Run Default multiclass example')
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
            data_dir=f"data/adult",
            prob_threshold=0.55,
            n_nearest=32
        )
        logger.info("Adult multiclass example completed")

    if args.bank or args.all:
        logger.info("\n=== Starting Bank Multiclass Example ===")
        bank_model, bank_dataset = train_method(
            dataset_class=BankDataset,
            dataset_name="Bank",
            save_dir=f"{args.save_dir}/bank",
            data_dir=f"data/bank",
            prob_threshold=0.55,
            n_nearest=32
        )
        logger.info("Bank multiclass example completed")

    if args.gmc or args.all:
        logger.info("\n=== Starting GMC Multiclass Example ===")
        gmc_model, gmc_dataset = train_method(
            dataset_class=GMCDataset,
            dataset_name="GMC",
            save_dir=f"{args.save_dir}/gmc",
            data_dir=f"data/gmc",
            prob_threshold=0.55,
            n_nearest=32
        )
        logger.info("GMC multiclass example completed")

    if args.lending or args.all:
        logger.info("\n=== Starting LendingClub Multiclass Example ===")
        gmc_model, gmc_dataset = train_method(
            dataset_class=LendingClubDataset,
            dataset_name="LendingClub",
            save_dir=f"{args.save_dir}/lending-club",
            data_dir=f"data/lending-club",
            prob_threshold=0.55,
            n_nearest=32
        )
        logger.info("LendingClub multiclass example completed")

    if args.default or args.all:
        logger.info("\n=== Starting Default Multiclass Example ===")
        gmc_model, gmc_dataset = train_method(
            dataset_class=DefaultDataset,
            dataset_name="Default",
            save_dir=f"{args.save_dir}/default",
            data_dir=f"data/default",
            prob_threshold=0.55,
            n_nearest=32
        )
        logger.info("Default multiclass example completed")
    
    logger.info("\nAll examples completed successfully!")
