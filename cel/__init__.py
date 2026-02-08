"""CEL: Counterfactual Explanations Library.

A Python framework for generating and evaluating counterfactual explanations
in machine learning models. The main contribution is PPCEF (Probabilistically
Plausible Counterfactual Explanations using Normalizing Flows).
"""

__version__ = "0.1.0"

from cel.cf_methods import (
    PPCEF,
    BaseCounterfactualMethod,
    ExplanationResult,
)
from cel.datasets import MethodDataset
from cel.losses import BinaryDiscLoss, MulticlassDiscLoss
from cel.metrics import CFMetrics, evaluate_cf
from cel.models import (
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
