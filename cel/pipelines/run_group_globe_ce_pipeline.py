import logging

import hydra
import torch
from omegaconf import DictConfig

from cel.pipelines.runners.group_globe_ce_runner import GroupGLOBECEPipelineRunner
from cel.preprocessing import (
    MinMaxScalingStep,
    PreprocessingPipeline,
    TorchDataTypeStep,
)

logger = logging.getLogger(__name__)


@hydra.main(config_path="./conf", config_name="group_globe_ce_config", version_base="1.2")
def main(cfg: DictConfig):
    torch.manual_seed(0)
    preprocessing_pipeline = PreprocessingPipeline(
        [
            ("torch_dtype", TorchDataTypeStep()),
            ("minmax", MinMaxScalingStep()),
        ]
    )
    runner = GroupGLOBECEPipelineRunner(cfg, logger, preprocessing_pipeline)
    runner.run()


if __name__ == "__main__":
    main()
