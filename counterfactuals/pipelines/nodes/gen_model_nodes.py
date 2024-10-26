import logging
import torch
import neptune
from hydra.utils import instantiate
from omegaconf import DictConfig
import torch.utils


logger = logging.getLogger(__name__)


def instantiate_gen_model(cfg: DictConfig, dataset: DictConfig) -> torch.nn.Module:
    """
    Create a generative model
    """
    logger.info("Creating generative model")
    gen_model = instantiate(
        cfg.gen_model.model, features=dataset.X_train.shape[1], context_features=1
    )
    return gen_model


def train_gen_model(
    gen_model: torch.nn.Module,
    dataset: DictConfig,
    gen_model_path: str,
    cfg: DictConfig,
    run: neptune.Run,
) -> torch.nn.Module:
    """
    Train a generative model
    """
    train_dataloader = dataset.train_dataloader(
        batch_size=cfg.gen_model.batch_size,
        shuffle=True,
        noise_lvl=cfg.gen_model.noise_lvl,
    )
    test_dataloader = dataset.test_dataloader(
        batch_size=cfg.gen_model.batch_size, shuffle=False
    )
    logger.info("Training generative model")
    gen_model.fit(
        train_loader=train_dataloader,
        test_loader=test_dataloader,
        num_epochs=cfg.gen_model.epochs,
        patience=cfg.gen_model.patience,
        learning_rate=cfg.gen_model.lr,
        checkpoint_path=gen_model_path,
        neptune_run=run,
    )
    gen_model.save(gen_model_path)
    return gen_model


def evaluate_gen_model(
    cfg: DictConfig, gen_model: torch.nn.Module, dataset: DictConfig, run: neptune.Run
) -> None:
    """
    Evaluate a generative model
    """
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
    run["metrics/gen_model"] = {"train_ll": train_ll, "test_ll": test_ll}
    logger.info(
        f"Generative model evaluation results:\n train_ll: {train_ll:.4f}, test_ll: {test_ll:.4f}"
    )


def create_gen_model(
    cfg: DictConfig, dataset: DictConfig, gen_model_path: str, run: neptune.Run
) -> torch.nn.Module:
    """
    Create and train a generative model
    """
    gen_model = instantiate_gen_model(cfg, dataset)
    if cfg.gen_model.train_model:
        gen_model = train_gen_model(gen_model, dataset, gen_model_path, cfg, run)
    else:
        logger.info("Loading generative model")
        gen_model.load(gen_model_path)

    gen_model.eval()
    logger.info("Evaluating generative model")
    evaluate_gen_model(cfg, gen_model, dataset, run)
    return gen_model
