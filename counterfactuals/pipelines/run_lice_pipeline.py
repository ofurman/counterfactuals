import logging

import hydra
import torch
from omegaconf import DictConfig

from counterfactuals.pipelines.runners.lice_runner import LiCEPipelineRunner

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


@hydra.main(config_path="./conf", config_name="globe_ce_config", version_base="1.2")
def main(cfg: DictConfig):
    torch.manual_seed(0)
    runner = LiCEPipelineRunner(cfg, logger, None)
    runner.run()


if __name__ == "__main__":
    main()
