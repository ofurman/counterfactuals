import logging
import os
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend

import hydra
import torch
import torch.utils
from hydra.utils import instantiate
from omegaconf import DictConfig

from counterfactuals.dequantization.dequantizer import GroupDequantizer
from counterfactuals.pipelines.nodes.disc_model_nodes import create_disc_model
from counterfactuals.pipelines.nodes.gen_model_nodes import create_gen_model
from counterfactuals.pipelines.nodes.helper_nodes import set_model_paths

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def get_categorical_intervals(
    use_categorical: bool, categorical_features_lists: List[List[int]]
) -> Optional[List[List[int]]]:
    """
    Get categorical feature intervals based on configuration.

    Returns the categorical features lists if categorical processing is enabled,
    otherwise returns None.

    Args:
        use_categorical: Whether to use categorical feature processing
        categorical_features_lists: List of lists containing categorical feature indices

    Returns:
        List of categorical feature intervals if use_categorical is True, None otherwise
    """
    return categorical_features_lists if use_categorical else None


@hydra.main(config_path="./conf", config_name="ppcef_config", version_base="1.2")
def main(cfg: DictConfig):
    torch.manual_seed(0)

    # Allow CUDA usage - remove the forced CPU-only setting
    # Only disable CUDA if explicitly configured to do so
    if hasattr(cfg, "use_cuda") and not cfg.use_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        logger.info("CUDA disabled by configuration")
    elif torch.cuda.is_available():
        logger.info(f"CUDA available: Using GPU {torch.cuda.get_device_name()}")
    else:
        logger.info("CUDA not available: Using CPU")

    logger.info("Initializing pipeline")

    disc_model_path, gen_model_path, save_folder = set_model_paths(cfg)

    logger.info("Loading dataset")
    dataset = instantiate(cfg.dataset)
    dequantizer = GroupDequantizer(dataset.categorical_features_lists)
    for fold_n, _ in enumerate(dataset.get_cv_splits(5)):
        disc_model_path, gen_model_path, save_folder = set_model_paths(cfg, fold=fold_n)
        disc_model = create_disc_model(cfg, dataset, disc_model_path, save_folder)

        if cfg.experiment.relabel_with_disc_model:
            dataset.y_train = disc_model.predict(dataset.X_train).detach().numpy()
            dataset.y_test = disc_model.predict(dataset.X_test).detach().numpy()

        dequantizer.fit(dataset.X_train)
        create_gen_model(cfg, dataset, gen_model_path, dequantizer)


if __name__ == "__main__":
    main()
