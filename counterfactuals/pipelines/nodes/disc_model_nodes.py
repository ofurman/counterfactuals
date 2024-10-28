import logging
import os
import numpy as np
import torch
import neptune
from neptune.utils import stringify_unsupported
from hydra.utils import instantiate
from omegaconf import DictConfig
from sklearn.metrics import classification_report
from sklearn.metrics import r2_score
import torch.utils
import pandas as pd

logger = logging.getLogger(__name__)


def isntantiate_disc_model(cfg: DictConfig, dataset: DictConfig) -> torch.nn.Module:
    """
    Create a discriminator model
    """
    logger.info("Creating discriminator model")
    binary_datasets = [
        "MoonsDataset",
        "LawDataset",
        "HelocDataset",
        "AuditDataset",
        "ToyRegressionDataset",
        "ConcreteDataset",
        "DiabetesDataset",
        "YachtDataset",
        "WineQualityDataset",
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
    run: neptune.Run,
) -> torch.nn.Module:
    """
    Train a discriminator model
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
    Evaluate a discriminator model
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
    run: neptune.Run,
) -> torch.nn.Module:
    """
    Create and train a discriminator model
    """
    disc_model_name = cfg.disc_model.model._target_.split(".")[-1]
    disc_model = isntantiate_disc_model(cfg, dataset)
    print(disc_model_path)

    if cfg.disc_model.train_model:
        disc_model = train_disc_model(disc_model, dataset, disc_model_path, cfg, run)
    else:
        logger.info("Loading discriminator model")
        disc_model.load(disc_model_path)

    disc_model.eval()
    report = evaluate_disc_model(disc_model, dataset)
    pd.DataFrame(report).transpose().to_csv(
        os.path.join(save_folder, f"eval_disc_model_{disc_model_name}.csv")
    )
    run["metrics/disc_model"] = stringify_unsupported(report)
    logger.info(
        f"Discriminator model evaluation results:\n {stringify_unsupported(report)}"
    )
    return disc_model
