"""Entry point for the TabDCE pairwise pipeline (multiple CFs per instance)."""

import logging

import hydra
import torch
from omegaconf import DictConfig

from counterfactuals.pipelines.runners.tabdce_pairwise_runner import TabDCEPairwisePipelineRunner

logger = logging.getLogger(__name__)


@hydra.main(config_path="./conf", config_name="tabdce_config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """Run TabDCE pairwise pipeline."""
    seed = cfg.experiment.get("seed", 0)
    torch.manual_seed(seed)
    runner = TabDCEPairwisePipelineRunner(
        cfg, logger, TabDCEPairwisePipelineRunner.default_preprocessing()
    )
    runner.run()


if __name__ == "__main__":
    main()
