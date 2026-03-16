import logging

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def instantiate_gen_model(cfg: DictConfig, dataset: DictConfig) -> torch.nn.Module:
    """
    Create a generative model instance based on configuration and dataset.

    Instantiates a generative model with the appropriate input dimensions based on
    the dataset features and configures it for conditional generation.

    Args:
        cfg: Hydra configuration containing model parameters
        dataset: Dataset instance containing training data for feature dimension

    Returns:
        torch.nn.Module: Instantiated generative model

    Raises:
        ImportError: If model class cannot be imported.
        TypeError: If model configuration is invalid.
        ValueError: If model instantiation fails.
    """
    logger.info("Creating generative model")
    try:
        gen_model = instantiate(
            cfg.gen_model.model, features=dataset.X_train.shape[1], context_features=1
        )
    except (ImportError, TypeError, ValueError) as e:
        model_target = cfg.gen_model.model.get("_target_", "unknown")
        logger.error(
            f"Failed to instantiate generative model '{model_target}': {e}. "
            f"Features: {dataset.X_train.shape[1]}, Context features: 1"
        )
        raise
    return gen_model


def train_gen_model(
    gen_model: torch.nn.Module,
    dataset: DictConfig,
    gen_model_path: str,
    cfg: DictConfig,
    dequantizer: object | None = None,
) -> torch.nn.Module:
    """
    Train a generative model on the provided dataset.

    Trains the model using configured parameters with optional noise augmentation,
    saves checkpoints during training, and saves the final model to the specified path.

    Args:
        gen_model: Instantiated generative model to train
        dataset: Dataset instance containing training and test data
        gen_model_path: File path where the trained model will be saved
        cfg: Hydra configuration containing training parameters
        dequantizer: Optional dequantizer for data preprocessing

    Returns:
        torch.nn.Module: Trained generative model

    Raises:
        RuntimeError: If training fails (e.g., CUDA OOM, NaN losses).
        OSError: If model cannot be saved to disk.
    """
    train_dataloader = dataset.train_dataloader(
        batch_size=cfg.gen_model.batch_size,
        shuffle=True,
        noise_lvl=cfg.gen_model.noise_lvl,
    )
    test_dataloader = dataset.test_dataloader(batch_size=cfg.gen_model.batch_size, shuffle=False)
    logger.info("Training generative model")
    try:
        gen_model.fit(
            train_loader=train_dataloader,
            test_loader=test_dataloader,
            epochs=cfg.gen_model.epochs,
            patience=cfg.gen_model.patience,
            lr=cfg.gen_model.lr,
            checkpoint_path=gen_model_path,
            dequantizer=dequantizer,
        )
        gen_model.save(gen_model_path)
    except RuntimeError as e:
        logger.error(f"Generative model training failed at path '{gen_model_path}': {e}")
        raise
    except OSError as e:
        logger.error(f"Failed to save generative model to '{gen_model_path}': {e}")
        raise
    return gen_model


def evaluate_gen_model(
    cfg: DictConfig,
    gen_model: torch.nn.Module,
    dataset: DictConfig,
) -> None:
    """
    Evaluate a generative model's performance using log-likelihood.

    Computes and logs the average log-likelihood on both training and test datasets
    to assess the model's ability to capture the data distribution.

    Args:
        cfg: Hydra configuration containing evaluation parameters
        gen_model: Trained generative model to evaluate
        dataset: Dataset instance containing training and test data

    Returns:
        None: Results are logged but not returned
    """
    try:
        train_dataloader = dataset.train_dataloader(
            batch_size=cfg.gen_model.batch_size,
            shuffle=True,
            noise_lvl=cfg.gen_model.noise_lvl,
        )
        test_dataloader = dataset.test_dataloader(
            batch_size=cfg.gen_model.batch_size, shuffle=False
        )
        gen_model.eval()
        logger.info("Evaluating generative model")
        train_ll = gen_model.predict_log_prob(train_dataloader).mean().item()
        test_ll = gen_model.predict_log_prob(test_dataloader).mean().item()
        logger.info(
            f"Generative model evaluation results:\n train_ll: {train_ll:.4f}, test_ll: {test_ll:.4f}"
        )
    except Exception as e:
        logger.warning(f"Generative model evaluation failed (non-fatal): {e}")
        logger.warning("Continuing without evaluation results")


def create_gen_model(
    cfg: DictConfig,
    dataset: DictConfig,
    gen_model_path: str,
    dequantizer: object | None = None,
) -> torch.nn.Module:
    """
    Create, train, and evaluate a generative model.

    Main orchestration function that handles the complete generative model pipeline:
    model instantiation, training (if enabled), loading (if pre-trained), and
    evaluation using log-likelihood metrics.

    Args:
        cfg: Hydra configuration containing all model and training parameters
        dataset: Dataset instance containing training and test data
        gen_model_path: File path for saving/loading the model
        dequantizer: Optional dequantizer for data preprocessing

    Returns:
        torch.nn.Module: Trained and evaluated generative model in evaluation mode

    Raises:
        FileNotFoundError: If model checkpoint file does not exist when loading.
        RuntimeError: If model checkpoint is corrupted or loading fails.
    """
    gen_model = instantiate_gen_model(cfg, dataset)
    if cfg.gen_model.train_model:
        gen_model = train_gen_model(gen_model, dataset, gen_model_path, cfg, dequantizer)
    else:
        logger.info("Loading generative model")
        try:
            gen_model.load(gen_model_path)
        except FileNotFoundError as e:
            logger.error(f"Generative model checkpoint not found at '{gen_model_path}': {e}")
            raise
        except RuntimeError as e:
            logger.error(f"Failed to load generative model from '{gen_model_path}': {e}")
            raise

    gen_model.eval()
    logger.info("Evaluating generative model")
    evaluate_gen_model(cfg, gen_model, dataset)
    return gen_model
