"""Train the classifier once per (dataset, seed), for every method to share.

Each method pipeline would otherwise train its own classifier into the same
path inside a seed root, so the three baselines explain three separately-trained
models that merely happen to share a seed — and, when the sweep runs methods
concurrently, race to write the same file. Training it once here and running the
pipelines with `disc_model.train_model=false` makes "the same model under
explanation" a property of the setup rather than a hope about determinism.

Usage:
    uv run python -m scripts.train_shared_disc_model \
        --config-name=dictum_dice_config \
        experiment.seed=42 \
        experiment.output_folder=results/dictum/seed_42 \
        dataset.config_path=config/datasets/adult_split.yaml \
        dataset.train_data_path=data_train_test_val/adult/train.csv \
        dataset.test_data_path=data_train_test_val/adult/test.csv \
        dataset.val_data_path=data_train_test_val/adult/val.csv
"""

import logging
import os

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from counterfactuals.datasets.method_dataset import MethodDataset
from counterfactuals.pipelines.nodes.disc_model_nodes import create_disc_model
from counterfactuals.pipelines.nodes.helper_nodes import set_model_paths
from counterfactuals.pipelines.nodes.seeding import set_global_seed
from counterfactuals.preprocessing import build_model_space_pipeline

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


@hydra.main(config_path="../counterfactuals/pipelines/conf", config_name=None, version_base="1.2")
def main(cfg: DictConfig) -> None:
    """Train and save the shared classifier for one dataset and seed."""
    set_global_seed(cfg.experiment.get("seed", 42))
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    preprocessing_pipeline = build_model_space_pipeline(
        cfg.experiment.get("model_space_scaler", "minmax")
    )
    dataset = MethodDataset(instantiate(cfg.dataset), preprocessing_pipeline)

    disc_model_path, _, save_folder = set_model_paths(cfg, fold=0)
    # `create_disc_model` honours cfg.disc_model.train_model, which the caller
    # must leave true here; the method runs then flip it to false.
    create_disc_model(cfg, dataset, disc_model_path, save_folder)
    logger.info("Shared classifier written to %s", disc_model_path)


if __name__ == "__main__":
    main()
