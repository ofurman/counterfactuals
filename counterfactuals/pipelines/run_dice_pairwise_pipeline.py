import logging

import hydra
from omegaconf import DictConfig

from counterfactuals.pipelines.runners.dice_pairwise_runner import DiCEPairwisePipelineRunner
from counterfactuals.preprocessing import (
    MinMaxScalingStep,
    PreprocessingPipeline,
    TorchDataTypeStep,
)

warnings = __import__("warnings")
warnings.filterwarnings("ignore", category=FutureWarning, module="dice_ml")

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


@hydra.main(config_path="./conf", config_name="dice_config", version_base="1.2")
def main(cfg: DictConfig):
    import torch

    torch.manual_seed(0)
    preprocessing_pipeline = PreprocessingPipeline(
        [
            ("minmax", MinMaxScalingStep()),
            ("torch_dtype", TorchDataTypeStep()),
        ]
    )
    runner = DiCEPairwisePipelineRunner(cfg, logger, preprocessing_pipeline)
    runner.run()


if __name__ == "__main__":
    main()
