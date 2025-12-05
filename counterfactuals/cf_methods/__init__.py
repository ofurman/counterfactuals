# Import base classes and mixins
# Import global methods using importlib to avoid keyword conflict
from .counterfactual_base import BaseCounterfactualMethod, ExplanationResult
from .global_counterfactual_mixin import GlobalCounterfactualMixin
from .global_methods import GLOBE_CE, AReS
from .group_counterfactual_mixin import GroupCounterfactualMixin
from .group_methods import GLANCE, RPPCEF
from .group_methods.tcrex import TCREx
from .local_counterfactual_mixin import LocalCounterfactualMixin

# Import from subpackages
from .local_methods import (
    CCHVAE,
    DICE,
    PPCEF,
    PPCEFR,
    WACH,
    WACH_OURS,
    Artelt,
    DiCoFlex,
)

# Import from subpackages

__all__ = [
    # Base classes
    "BaseCounterfactualMethod",
    "ExplanationResult",
    "LocalCounterfactualMixin",
    "GlobalCounterfactualMixin",
    "GroupCounterfactualMixin",
    # Local methods
    "PPCEF",
    "DiCoFlex",
    "DICE",
    "WACH",
    "WACH_OURS",
    "Artelt",
    "PPCEFR",
    "CCHVAE",
    # Global methods
    "AReS",
    "GLOBE_CE",
    # Group methods
    "RPPCEF",
    "GLANCE",
    "TCREx",
]
