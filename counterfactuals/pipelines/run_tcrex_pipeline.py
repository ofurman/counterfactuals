import logging

import hydra
import torch
from omegaconf import DictConfig

from counterfactuals.pipelines.runners.tcrex_runner import TCRExPipelineRunner
from counterfactuals.preprocessing import (
    MinMaxScalingStep,
    PreprocessingPipeline,
    TorchDataTypeStep,
)

logger = logging.getLogger(__name__)


@hydra.main(config_path="./conf", config_name="tcrex_config", version_base="1.2")
def main(cfg: DictConfig):
    torch.manual_seed(0)
    preprocessing_pipeline = PreprocessingPipeline(
        [
            ("minmax", MinMaxScalingStep()),
            ("torch_dtype", TorchDataTypeStep()),
        ]
    )
    runner = TCRExPipelineRunner(cfg, logger, preprocessing_pipeline)
    runner.run()


if __name__ == "__main__":
    main()
