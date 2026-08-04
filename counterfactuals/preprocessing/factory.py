from counterfactuals.preprocessing.pipeline import PreprocessingPipeline
from counterfactuals.preprocessing.scalers import MinMaxScalingStep, StandardScalingStep
from counterfactuals.preprocessing.torch_dtype import TorchDataTypeStep

_SCALER_STEPS = {
    "minmax": MinMaxScalingStep,
    "standard": StandardScalingStep,
}


def build_model_space_pipeline(scaler: str = "minmax") -> PreprocessingPipeline:
    """Build the continuous-feature-scaling + torch-dtype pipeline shared by the
    traintest baselines (DiCE, CCHVAE, DiCoFlex).

    Args:
        scaler: "minmax" (default, unchanged behavior) for MinMaxScaler(0, 1),
            or "standard" for StandardScaler (z-score) — the space DICTUM's
            evaluation protocol uses.

    Returns:
        A PreprocessingPipeline with the scaler registered under the "minmax"
        step name for backward compatibility with code that does
        `pipeline.get_step("minmax")` regardless of which scaler is active.
    """
    try:
        scaling_step_cls = _SCALER_STEPS[scaler]
    except KeyError:
        raise ValueError(f"Unknown scaler '{scaler}', expected one of {list(_SCALER_STEPS)}")
    return PreprocessingPipeline(
        [
            ("minmax", scaling_step_cls()),
            ("torch_dtype", TorchDataTypeStep()),
        ]
    )
