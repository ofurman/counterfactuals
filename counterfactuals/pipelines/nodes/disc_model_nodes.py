import logging
import os

import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from sklearn.metrics import classification_report, r2_score

logger = logging.getLogger(__name__)


def isntantiate_disc_model(cfg: DictConfig, dataset: DictConfig) -> torch.nn.Module:
    """
    Create a discriminative model instance based on configuration and dataset.

    Automatically determines the number of output classes based on the dataset type
    and creates an appropriate discriminative model (classification or regression).

    Args:
        cfg: Hydra configuration containing model parameters
        dataset: Dataset instance containing training data

    Returns:
        torch.nn.Module: Instantiated discriminative model
    """
    logger.info("Creating discriminator model")
    binary_datasets = [
        # "MoonsDataset",
        # "LawDataset",
        # "HelocDataset",
        # "AuditDataset",
        # "ToyRegressionDataset",
        # "ConcreteDataset",
        # "DiabetesDataset",
        # "YachtDataset",
        # "WineQualityDataset",
        # "WineDataset",
    ]
    dataset_name = cfg.dataset._target_.split(".")[-1]
    num_classes = (
        1 if dataset_name in binary_datasets else len(np.unique(dataset.y_train))
    )
    num_classes = 20 if dataset_name == "Scm20dDataset" else num_classes

    disc_model = instantiate(
        cfg.disc_model.model,
        input_size=dataset.X_train.shape[1],
        target_size=num_classes,
    )
    return disc_model


def train_disc_model(
    disc_model: torch.nn.Module,
    dataset: DictConfig,
    disc_model_path: str,
    cfg: DictConfig,
) -> torch.nn.Module:
    """
    Train a discriminative model on the provided dataset.

    Trains the model using configured parameters, saves checkpoints during training,
    and saves the final model to the specified path.

    Args:
        disc_model: Instantiated discriminative model to train
        dataset: Dataset instance containing training and test data
        disc_model_path: File path where the trained model will be saved
        cfg: Hydra configuration containing training parameters

    Returns:
        torch.nn.Module: Trained discriminative model
    """
    logger.info("Training discriminator model")
    train_dataloader = dataset.train_dataloader(
        batch_size=cfg.disc_model.batch_size, shuffle=True, noise_lvl=0
    )
    test_dataloader = dataset.test_dataloader(
        batch_size=cfg.disc_model.batch_size, shuffle=False
    )
    disc_model.fit(
        train_dataloader,
        test_dataloader,
        epochs=cfg.disc_model.epochs,
        lr=cfg.disc_model.lr,
        patience=cfg.disc_model.patience,
        checkpoint_path=disc_model_path,
    )
    disc_model.save(disc_model_path)
    return disc_model


def evaluate_disc_model(disc_model: torch.nn.Module, dataset: DictConfig) -> dict:
    """
    Evaluate a discriminative model's performance on test data.

    Automatically determines evaluation metrics based on the model type:
    - Classification models: Uses classification report with precision, recall, F1-score
    - Regression models: Uses R² score

    Args:
        disc_model: Trained discriminative model to evaluate
        dataset: Dataset instance containing test data and labels

    Returns:
        dict: Dictionary containing evaluation metrics
    """
    logger.info("Evaluating discriminator model")
    try:
        print(classification_report(dataset.y_test, disc_model.predict(dataset.X_test)))
        report = classification_report(
            dataset.y_test, disc_model.predict(dataset.X_test), output_dict=True
        )
    except ValueError:
        # evaluate regression model on R1 score
        report = [
            {"r2_score": r2_score(dataset.y_test, disc_model.predict(dataset.X_test))}
        ]
        print(report)

    return report


def create_disc_model(
    cfg: DictConfig,
    dataset: DictConfig,
    disc_model_path: str,
    save_folder: str,
) -> torch.nn.Module:
    """
    Create, train, and evaluate a discriminative model.

    Main orchestration function that handles the complete discriminative model pipeline:
    model instantiation, training (if enabled), loading (if pre-trained), evaluation,
    and results saving.

    Args:
        cfg: Hydra configuration containing all model and training parameters
        dataset: Dataset instance containing training and test data
        disc_model_path: File path for saving/loading the model
        save_folder: Directory path for saving evaluation results

    Returns:
        torch.nn.Module: Trained and evaluated discriminative model in evaluation mode
    """
    disc_model_name = cfg.disc_model.model._target_.split(".")[-1]
    disc_model = isntantiate_disc_model(cfg, dataset)
    print(disc_model_path)

    if cfg.disc_model.train_model:
        disc_model = train_disc_model(disc_model, dataset, disc_model_path, cfg)
    else:
        logger.info("Loading discriminator model")
        disc_model.load(disc_model_path)

    disc_model.eval()
    report = evaluate_disc_model(disc_model, dataset)
    pd.DataFrame(report).transpose().to_csv(
        os.path.join(save_folder, f"eval_disc_model_{disc_model_name}.csv")
    )
    logger.info(f"Discriminator model evaluation results:\n {report}")
    return disc_model
