import logging

import hydra
import torch
from omegaconf import DictConfig

from counterfactuals.pipelines.runners.group_globe_ce_runner import GroupGLOBECEPipelineRunner

logger = logging.getLogger(__name__)


@hydra.main(config_path="./conf", config_name="group_globe_ce_config", version_base="1.2")
def main(cfg: DictConfig):
    torch.manual_seed(0)
    runner = GroupGLOBECEPipelineRunner(cfg, logger, None)
    runner.run()


if __name__ == "__main__":
    main()
