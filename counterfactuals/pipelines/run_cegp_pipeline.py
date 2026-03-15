import logging

import hydra
from omegaconf import DictConfig

from counterfactuals.pipelines.runners.cegp_runner import CEGPPipelineRunner
from counterfactuals.preprocessing import (
    MinMaxScalingStep,
    PreprocessingPipeline,
    TorchDataTypeStep,
)

logger = logging.getLogger(__name__)


@hydra.main(config_path="./conf", config_name="cegp_config", version_base="1.2")
def main(cfg: DictConfig):
    preprocessing_pipeline = PreprocessingPipeline(
        [
            ("minmax", MinMaxScalingStep()),
            ("torch_dtype", TorchDataTypeStep()),
        ]
    )
    runner = CEGPPipelineRunner(cfg, logger, preprocessing_pipeline)
    runner.run()


if __name__ == "__main__":
    main()
