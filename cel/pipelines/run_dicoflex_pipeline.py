import logging

import hydra
from omegaconf import DictConfig

from cel.pipelines.nodes.seeding import set_global_seed
from cel.pipelines.runners.dicoflex_runner import DiCoFlexPipelineRunner
from cel.preprocessing import (
    MinMaxScalingStep,
    PreprocessingPipeline,
    TorchDataTypeStep,
)

logger = logging.getLogger(__name__)


@hydra.main(config_path="./conf", config_name="dicoflex_config", version_base="1.2")
def main(cfg: DictConfig):
    set_global_seed(cfg.experiment.get("seed", 42))
    preprocessing_pipeline = PreprocessingPipeline(
        [
            ("minmax", MinMaxScalingStep()),
            ("torch_dtype", TorchDataTypeStep()),
        ]
    )
    runner = DiCoFlexPipelineRunner(cfg, logger, preprocessing_pipeline)
    runner.run()


if __name__ == "__main__":
    main()
