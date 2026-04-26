# Import base classes and mixins
# Import global methods using importlib to avoid keyword conflict
from .counterfactual_base import BaseCounterfactualMethod, ExplanationResult
from .global_counterfactual_mixin import GlobalCounterfactualMixin
from .global_methods import GLOBE_CE, AReS
from .group_counterfactual_mixin import GroupCounterfactualMixin
from .group_methods import GLANCE
from .group_methods.tcrex import TCREx
from .local_counterfactual_mixin import LocalCounterfactualMixin

# Import from subpackages
from .local_methods import (
    CADEX,
    CCHVAE,
    CEARM,
    CEGP,
    CEM_CF,
    DICE,
    PPCEF,
    SACE,
    WACH,
    Artelt,
)

try:
    from .local_methods import CeFlow  # type: ignore[attr-defined]
except ImportError:
    CeFlow = None
try:
    from .local_methods import DiCoFlex  # type: ignore[attr-defined]
except ImportError:
    DiCoFlex = None
try:
    from .local_methods import TabDCE  # type: ignore[attr-defined]
except ImportError:
    TabDCE = None

# Backward compatibility alias for legacy Hydra targets.
RPPCEF = PPCEF

# Import from subpackages

__all__ = [
    # Base classes
    "BaseCounterfactualMethod",
    "ExplanationResult",
    "LocalCounterfactualMixin",
    "GlobalCounterfactualMixin",
    "GroupCounterfactualMixin",
    # Local methods
    "RPPCEF",
    "PPCEF",
    "DICE",
    "WACH",
    "Artelt",
    "CEM_CF",
    "CCHVAE",
    "TabDCE",
    "CeFlow",
    # Global methods
    "AReS",
    "GLOBE_CE",
    # Group methods
    "GLANCE",
    "TCREx",
]
