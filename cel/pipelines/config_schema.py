"""Structured config schemas for pipeline runners.

These dataclasses document and enforce the fields that every pipeline config
must provide.  They are used in :meth:`PipelineRunner._validate_cfg` to catch
misconfigured runs before any model loading or training begins.

Usage (validation only — no need to register with Hydra's ConfigStore)::

    from cel.pipelines.config_schema import REQUIRED_CFG_KEYS

Only the keys in ``REQUIRED_CFG_KEYS`` are validated at runtime; method-specific
keys (e.g. ``counterfactuals_params.hyperparams``) are left unchecked so that
runner subclasses remain free to extend the config.
"""

from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class ExperimentConfig:
    """Experiment-level settings present in every pipeline config."""

    output_folder: str = MISSING
    relabel_with_disc_model: bool = MISSING


@dataclass
class DiscModelTrainingConfig:
    """Training hyperparameters for the discriminative model."""

    train_model: bool = MISSING
    epochs: int = MISSING
    batch_size: int = MISSING
    patience: int = MISSING
    lr: float = MISSING


@dataclass
class GenModelTrainingConfig:
    """Training hyperparameters for the generative model."""

    train_model: bool = MISSING
    epochs: int = MISSING
    batch_size: int = MISSING
    patience: int = MISSING
    lr: float = MISSING


@dataclass
class CounterfactualsParamsConfig:
    """Parameters common to every counterfactual search."""

    log_prob_quantile: float = MISSING
    target_class: int = MISSING
    batch_size: int = MISSING


# Flat list of dot-separated paths validated in PipelineRunner._validate_cfg.
# Derived from the dataclasses above so there is a single source of truth.
REQUIRED_CFG_KEYS: tuple[str, ...] = (
    "experiment.output_folder",
    "experiment.relabel_with_disc_model",
    "disc_model.train_model",
    "disc_model.epochs",
    "disc_model.batch_size",
    "disc_model.patience",
    "disc_model.lr",
    "gen_model.train_model",
    "gen_model.epochs",
    "gen_model.batch_size",
    "gen_model.patience",
    "gen_model.lr",
    "counterfactuals_params.log_prob_quantile",
    "counterfactuals_params.target_class",
    "counterfactuals_params.batch_size",
)
