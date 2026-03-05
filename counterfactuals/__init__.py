"""CEL: Counterfactual Explanations Library.

A Python framework for generating and evaluating counterfactual explanations
in machine learning models. The main contribution is PPCEF (Probabilistically
Plausible Counterfactual Explanations using Normalizing Flows).
"""

__version__ = "0.1.0"

from counterfactuals.cf_methods import (
    PPCEF,
    BaseCounterfactualMethod,
    ExplanationResult,
)
from counterfactuals.datasets import MethodDataset
from counterfactuals.losses import BinaryDiscLoss, MulticlassDiscLoss
from counterfactuals.metrics import CFMetrics, evaluate_cf
from counterfactuals.models import (
    MaskedAutoregressiveFlow,
    MLPClassifier,
)

__all__ = [
    # Version
    "__version__",
    # CF Methods
    "PPCEF",
    "BaseCounterfactualMethod",
    "ExplanationResult",
    # Datasets
    "MethodDataset",
    # Losses
    "BinaryDiscLoss",
    "MulticlassDiscLoss",
    # Metrics
    "CFMetrics",
    "evaluate_cf",
    # Models
    "MaskedAutoregressiveFlow",
    "MLPClassifier",
]
