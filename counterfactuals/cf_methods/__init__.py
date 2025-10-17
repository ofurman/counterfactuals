# Import base classes and mixins
# Import global methods using importlib to avoid keyword conflict
import importlib

from .counterfactual_base import BaseCounterfactualMethod, ExplanationResult
from .global_counterfactual_mixin import GlobalCounterfactualMixin
from .group_counterfactual_mixin import GroupCounterfactualMixin

# Import from subpackages
from .local import CCHVAE, DICE, PPCEF, PPCEFR, WACH, WACH_OURS, Artelt
from .local_counterfactual_mixin import LocalCounterfactualMixin

__all__ = [
    # Base classes
    "BaseCounterfactualMethod",
    "ExplanationResult",
    "LocalCounterfactualMixin",
    "GlobalCounterfactualMixin",
    "GroupCounterfactualMixin",
    # Local methods
    "PPCEF",
    "DICE",
    "WACH",
    "WACH_OURS",
    "Artelt",
    "PPCEFR",
    # Global methods
    "CCHVAE",
    "GlobalGLANCE",
    "GLOBE_CE",
]
