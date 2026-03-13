import logging

import hydra
import torch
from omegaconf import DictConfig

from counterfactuals.pipelines.run_tabdce_pipeline import build_preprocessing_pipeline
from counterfactuals.pipelines.runners.tabdce_pairwise_runner import TabDCEPairwisePipelineRunner

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


@hydra.main(config_path="./conf", config_name="tabdce_config", version_base="1.2")
def main(cfg: DictConfig):
    seed = cfg.experiment.get("seed", 0)
    torch.manual_seed(seed)
    preprocessing_pipeline = build_preprocessing_pipeline()
    runner = TabDCEPairwisePipelineRunner(cfg, logger, preprocessing_pipeline)
    runner.run()


if __name__ == "__main__":
    main()
