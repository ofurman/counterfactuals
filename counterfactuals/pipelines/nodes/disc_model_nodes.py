import json
import logging
import os

import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
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
    num_classes = 1 if dataset_name in binary_datasets else len(np.unique(dataset.y_train))
    num_classes = 20 if dataset_name == "Scm20dDataset" else num_classes

    model_config = OmegaConf.to_container(cfg.disc_model.model)
    del model_config["_target_"]

    disc_model = instantiate(
        cfg.disc_model.model,
        num_inputs=dataset.X_train.shape[1],
        num_targets=num_classes,
        **model_config,
    )
    return disc_model


def build_early_stopping_loaders(
    dataset: DictConfig,
    batch_size: int,
    validation_source: str,
    seed: int,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Build the (train, early-stopping) loader pair for discriminator training.

    Args:
        dataset: Dataset instance exposing train/test/val loaders.
        batch_size: Batch size for both loaders.
        validation_source: Which split early stopping selects on.
            "test" keeps the historical behaviour of monitoring the test split.
            "val" uses the dataset's validation split, falling back to a held-out
            80/20 slice of train when the dataset ships no val.csv.
        seed: Seed for the 80/20 fallback split, so the held-out rows are
            reproducible and vary with the experiment seed.

    Returns:
        Tuple of (train_dataloader, early_stopping_dataloader).

    Raises:
        ValueError: If validation_source is not "test" or "val".
    """
    if validation_source == "test":
        return (
            dataset.train_dataloader(batch_size=batch_size, shuffle=True, noise_lvl=0),
            dataset.test_dataloader(batch_size=batch_size, shuffle=False),
        )
    if validation_source != "val":
        raise ValueError(
            f"Unknown validation_source '{validation_source}', expected 'test' or 'val'"
        )

    val_dataloader = dataset.val_dataloader(batch_size=batch_size, shuffle=False)
    if val_dataloader is not None:
        logger.info("Early stopping on the dataset's validation split")
        return (
            dataset.train_dataloader(batch_size=batch_size, shuffle=True, noise_lvl=0),
            val_dataloader,
        )

    logger.info("No validation split available; holding out 20%% of train for early stopping")
    n_train = len(dataset.X_train)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_train)
    n_holdout = max(1, int(round(0.2 * n_train)))
    holdout_idx, fit_idx = perm[:n_holdout], perm[n_holdout:]

    X_fit = torch.from_numpy(dataset.X_train[fit_idx])
    y_fit = torch.from_numpy(dataset.y_train[fit_idx])
    X_holdout = torch.from_numpy(dataset.X_train[holdout_idx])
    y_holdout = torch.from_numpy(dataset.y_train[holdout_idx])

    train_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_fit, y_fit), batch_size=batch_size, shuffle=True
    )
    holdout_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_holdout, y_holdout), batch_size=batch_size, shuffle=False
    )
    return train_dataloader, holdout_dataloader


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
    train_dataloader, early_stopping_dataloader = build_early_stopping_loaders(
        dataset,
        batch_size=cfg.disc_model.batch_size,
        validation_source=cfg.disc_model.get("validation_source", "test"),
        seed=cfg.experiment.get("seed", 42),
    )
    disc_model.fit(
        train_dataloader,
        early_stopping_dataloader,
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
        report = [{"r2_score": r2_score(dataset.y_test, disc_model.predict(dataset.X_test))}]
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

    # The model space this classifier reads. A checkpoint silently loaded into a
    # different space produces garbage predictions (the DiCoFlex minmax_qt runs
    # once loaded a standard-space classifier this way), so the space is recorded
    # next to the checkpoint at train time and verified at load time.
    disc_space = cfg.experiment.get("disc_model_space_scaler") or cfg.experiment.get(
        "model_space_scaler", "minmax"
    )
    space_sidecar = disc_model_path + ".space.json"
    if cfg.disc_model.train_model:
        disc_model = train_disc_model(disc_model, dataset, disc_model_path, cfg)
        with open(space_sidecar, "w") as f:
            json.dump({"model_space_scaler": disc_space}, f)
    else:
        logger.info("Loading discriminator model")
        if os.path.exists(space_sidecar):
            with open(space_sidecar) as f:
                recorded_space = json.load(f)["model_space_scaler"]
            if recorded_space != disc_space:
                raise RuntimeError(
                    f"Classifier checkpoint {disc_model_path} was trained in the "
                    f"'{recorded_space}' model space but this run feeds it "
                    f"'{disc_space}'. Set experiment.disc_model_space_scaler="
                    f"{recorded_space} (or retrain) instead of loading it into a "
                    "space it never saw."
                )
        disc_model.load(disc_model_path)

    disc_model.eval()
    report = evaluate_disc_model(disc_model, dataset)
    pd.DataFrame(report).transpose().to_csv(
        os.path.join(save_folder, f"eval_disc_model_{disc_model_name}.csv")
    )
    logger.info(f"Discriminator model evaluation results:\n {report}")
    return disc_model
