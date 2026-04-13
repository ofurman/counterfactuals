"""Entry point for the TabDCE pipeline (single-CF variant)."""

import logging

import hydra
import torch
from omegaconf import DictConfig

from counterfactuals.pipelines.runners.tabdce_runner import (  # noqa: F401
    TabDCEPipelineRunner,
    create_diffusion_model,
    prepare_tabular_dataset,
    train_tabdce_diffusion,
)

logger = logging.getLogger(__name__)


@hydra.main(config_path="./conf", config_name="tabdce_config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """Run TabDCE with the standard full pipeline interface."""
    seed = cfg.experiment.get("seed", 0)
    torch.manual_seed(seed)
    runner = TabDCEPipelineRunner(cfg, logger, TabDCEPipelineRunner.default_preprocessing())
    runner.run()


if __name__ == "__main__":
    main()
