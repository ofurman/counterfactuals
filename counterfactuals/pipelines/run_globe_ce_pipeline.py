import logging

import hydra
import torch
from omegaconf import DictConfig

from counterfactuals.pipelines.runners.globe_ce_runner import GLOBECEPipelineRunner
from counterfactuals.preprocessing import (
    MinMaxScalingStep,
    PreprocessingPipeline,
    TorchDataTypeStep,
)

logger = logging.getLogger(__name__)


@hydra.main(config_path="./conf", config_name="globe_ce_config", version_base="1.2")
def main(cfg: DictConfig):
    torch.manual_seed(0)
    # GLOBE-CE uses torch_dtype first, then minmax (order matters for scaling)
    preprocessing_pipeline = PreprocessingPipeline(
        [
            ("torch_dtype", TorchDataTypeStep()),
            ("minmax", MinMaxScalingStep()),
        ]
    )
    runner = GLOBECEPipelineRunner(cfg, logger, preprocessing_pipeline)
    runner.run()


if __name__ == "__main__":
    main()
