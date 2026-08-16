from counterfactuals.preprocessing.pipeline import PreprocessingPipeline
from counterfactuals.preprocessing.scalers import (
    MinMaxScalingStep,
    QuantileTransformCategoricalStep,
    StandardScalingStep,
)
from counterfactuals.preprocessing.torch_dtype import TorchDataTypeStep

_SCALER_STEPS = {
    "minmax": MinMaxScalingStep,
    "standard": StandardScalingStep,
}


def build_model_space_pipeline(scaler: str = "minmax") -> PreprocessingPipeline:
    """Build the continuous-feature-scaling + torch-dtype pipeline shared by the
    traintest baselines (DiCE, CCHVAE, DiCoFlex).

    Args:
        scaler: one of
            * "minmax"    -- MinMaxScaler(0, 1) on continuous features (default).
            * "standard"  -- StandardScaler (z-score) on continuous features; the
              space DICTUM's evaluation protocol reports metrics in.
            * "minmax_qt" -- the ORIGINAL DiCoFlex generation space
              (``ofurman/DiCoFlex``): MinMax on continuous features AND a
              QuantileTransformer on the one-hot categorical columns. Use this to
              GENERATE counterfactuals; report metrics with ``standard``.

    Returns:
        A PreprocessingPipeline with the continuous scaler registered under the
        "minmax" step name for backward compatibility with code that does
        `pipeline.get_step("minmax")` regardless of which scaler is active.
    """
    if scaler == "minmax_qt":
        return PreprocessingPipeline(
            [
                ("minmax", MinMaxScalingStep()),
                ("quantile_categorical", QuantileTransformCategoricalStep()),
                ("torch_dtype", TorchDataTypeStep()),
            ]
        )
    try:
        scaling_step_cls = _SCALER_STEPS[scaler]
    except KeyError:
        raise ValueError(
            f"Unknown scaler '{scaler}', expected one of {list(_SCALER_STEPS) + ['minmax_qt']}"
        )
    return PreprocessingPipeline(
        [
            ("minmax", scaling_step_cls()),
            ("torch_dtype", TorchDataTypeStep()),
        ]
    )
