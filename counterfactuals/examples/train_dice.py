import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import pandas as pd
import dice_ml
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
from counterfactuals.datasets.adult import AdultDataset
from counterfactuals.datasets.generic_counterfactual import AbstractDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('counterfactual_example')


def prepare_dataset_and_models(dataset_class: AbstractDataset, save_dir: str):
    dataset = dataset_class()
    try:
        dataset.add_noise_to_categorical_features()
    except:
        pass
    disc_model = MultilayerPerceptron(
        input_size=dataset.X_train.shape[1],
        hidden_layer_sizes=[256, 256],
        target_size=np.unique(dataset.y_train).shape[0]
    )
    disc_model.fit(
        dataset.train_dataloader(64, True),
        dataset.test_dataloader(64, False),
        epochs=1000,
        lr=0.001,
        patience=50,
        checkpoint_path=os.path.join(save_dir, "disc_model.pth")
    )
    disc_model = disc_model.eval()
    y_train = disc_model.predict(dataset.X_train).numpy().astype(int)
    y_test = disc_model.predict(dataset.X_test).numpy().astype(int)
    logger.info(f"Discriminator model accuracy: {np.sum(y_test == dataset.y_test) / len(dataset.y_test)}")
    dataset.y_train = y_train
    dataset.y_test = y_test

    gen_model = MaskedAutoregressiveFlow(features=dataset.X_train.shape[1], hidden_features=16, num_layers=2, num_blocks_per_layer=2, context_features=1)
    gen_model.fit(
        dataset.train_dataloader(64, True, 0.03), 
        dataset.test_dataloader(64, False), 
        num_epochs=1000, 
        learning_rate=0.001, 
        patience=50,
        checkpoint_path=os.path.join(save_dir, "gen_model.pth")
    )
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
    
    dataset, disc_model, gen_model = prepare_dataset_and_models(dataset_class, save_dir)
    
    # Visualize the dataset
    if dataset.X.shape[1] == 2:
        visualize_dataset(
            dataset.X_train, dataset.y_train, 
            title=f"{dataset_name} Dataset (Multiclass)", 
            save_path=os.path.join(save_dir, f"{dataset_name}_dataset.png")
        )
        logger.info(f"Saved dataset visualization to {os.path.join(save_dir, f'{dataset_name}_dataset.png')}")

    # Select threshold for classifier as median of the posterior probabilities
    logger.info(f"Selected threshold for classifier: {prob_threshold}")
    
    # Create the multiclass counterfactual wrapper
    logger.info("Creating multiclass counterfactual dataset wrapper")
    train_dataset = pd.DataFrame(dataset.X_train, columns=dataset.feature_columns)
    train_dataset['target'] = dataset.y_train
    # Dataset for training an ML model
    print(train_dataset.columns)
    print(dataset.numerical_feature_columns)
    d = dice_ml.Data(
        dataframe=train_dataset,
        continuous_features=dataset.numerical_feature_columns,
        outcome_name='target'
    )

    # Pre-trained ML model
    m = dice_ml.Model(model=disc_model, backend='PYT', func=None, model_type='classifier')
    # DiCE explanation instance
    exp = dice_ml.Dice(d,m)
    logger.info("Multiclass model training complete")
    
    # Generate and visualize counterfactuals using multiclass model
    logger.info("Generating and visualizing counterfactuals using multiclass model")
    cf_vis_dir = os.path.join(save_dir, "counterfactual_visualization")
    os.makedirs(cf_vis_dir, exist_ok=True)

    # if dataset.X.shape[1] == 2:
    #     visualize_multiclass_counterfactual_generation(
    #         model=multiclass_model,
    #         dataset=dataset,
    #         disc_model=disc_model,
    #         num_factual=6,
    #         num_samples=40,
    #         temperature=0.8,
    #         save_dir=cf_vis_dir
    #     )
    #     logger.info(f"Saved multiclass counterfactual visualizations to {cf_vis_dir}")

    # Generate counterfactuals for evaluation
    logger.info("Generating counterfactuals for evaluation")
    
    metrics_results = {}

    for factual_class in np.unique(dataset.y_train):
        logger.info(f"Generating counterfactuals for factual class {factual_class}")
        factual_indices = np.where(dataset.y_test == factual_class)[0]
        factual_points = dataset.X_test[factual_indices]
        
        for target_class in np.unique(dataset.y_train):
            if target_class == factual_class:
                continue
            
            logger.info(f"Generating counterfactuals for factual class {factual_class} to target class {target_class}")
            generated_cfs = []
            factual_points = pd.DataFrame(dataset.X_test[factual_indices], columns=dataset.feature_columns)
            dice_exp = exp.generate_counterfactuals(factual_points, total_CFs=100, desired_class=int(target_class), verbose=True)
            for generated_cf in dice_exp.cf_examples_list:
                generated_cfs.append(generated_cf.final_cfs_df.to_numpy()[:, :-1])
            generated_cfs = np.array(generated_cfs)

            metrics_forward, cfs_orig_forward = evaluate_counterfactuals(
                disc_model=disc_model,
                gen_model=gen_model,
                dataset=dataset,
                X=dataset.X_test,
                y=dataset.y_test,
                target_class=target_class,
                factual_indices=factual_indices,
                generated_cfs=generated_cfs,
                direction=f'class_{factual_class}_to_class_{target_class}',
                save_dir=save_dir
            )
            metrics_results[f'class_{factual_class}_to_class_{target_class}'] = metrics_forward
    
    return exp, dataset


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
    args = parser.parse_args()
    
    # Run the selected examples
    if args.moons or args.all:
        logger.info("\n=== Starting Moons Multiclass Example ===")
        moons_model, moons_dataset = train_method(
            dataset_class=MoonsDataset,
            dataset_name="Moons",
            save_dir="results_dice/moons",
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
            save_dir="results_dice/blobs",
            prob_threshold=0.98,
            n_nearest=32
        )
        logger.info("Three-class example completed")
    
    if args.law or args.all:
        logger.info("\n=== Starting Law Multiclass Example ===")
        law_model, law_dataset = train_method(
            dataset_class=LawDataset,
            dataset_name="Law",
            save_dir="results_dice/law",
            prob_threshold=0.55,
            n_nearest=32
        )
        logger.info("Law multiclass example completed")
    
    if args.heloc or args.all:
        logger.info("\n=== Starting HELOC Multiclass Example ===")
        heloc_model, heloc_dataset = train_method(
            dataset_class=HelocDataset,
            dataset_name="HELOC",
            save_dir="results_dice/heloc",
            prob_threshold=0.55,
            n_nearest=32
        )
        logger.info("HELOC multiclass example completed")
    
    if args.wine or args.all:
        logger.info("\n=== Starting Wine Multiclass Example ===")
        wine_model, wine_dataset = train_method(
            dataset_class=WineDataset,
            dataset_name="Wine",
            save_dir="results_dice/wine",
            prob_threshold=0.55,
            n_nearest=32
        )
        logger.info("Wine multiclass example completed")

    if args.adult or args.all:
        logger.info("\n=== Starting Adult Multiclass Example ===")
        adult_model, adult_dataset = train_method(
            dataset_class=AdultDataset,
            dataset_name="Adult",
            save_dir="results_dice/adult",
            prob_threshold=0.55,
            n_nearest=32
        )
        logger.info("Adult multiclass example completed")
    
    logger.info("\nAll examples completed successfully!")