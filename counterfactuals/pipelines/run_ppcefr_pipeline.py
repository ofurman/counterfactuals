import logging

import hydra
import torch
from omegaconf import DictConfig

from counterfactuals.pipelines.runners.ppcefr_runner import PPCEFRPipelineRunner

logger = logging.getLogger(__name__)


@hydra.main(config_path="./conf", config_name="ppcefr_config", version_base="1.2")
def main(cfg: DictConfig):
    torch.manual_seed(0)
    runner = PPCEFRPipelineRunner(cfg, logger, None)
    runner.run()


if __name__ == "__main__":
    main()
