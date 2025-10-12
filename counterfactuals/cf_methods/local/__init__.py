"""Local counterfactual methods package."""

from .ares import AReS
from .artelt import Artelt
from .casebased_sace.casebased_sace import CaseBasedSACE
from .cegp.cegp import CEGP
from .cem.cem import CEM_CF
from .cet.cet import CounterfactualExplanationTree as CET
from .dice.dice import DiceExplainerWrapper
from .ppcef.ppcef import PPCEF
from .regression_ppcef.ppcefr import PPCEFR
from .sace.sace import SACE
from .wach.wach import WACH
from .wach.wach_ours import WACH_OURS

# LiCE has optional dependencies (pyomo, onnx, omlt) that may not be installed
try:
    from .lice.lice import LiCE

    _LICE_AVAILABLE = True
except (ImportError, AttributeError) as e:
    _LICE_AVAILABLE = False
    LiCE = None  # type: ignore

__all__ = [
    "PPCEF",
    "WACH",
    "WACH_OURS",
    "Artelt",
    "PPCEFR",
    "DiceExplainerWrapper",
    "CEM_CF",
    "SACE",
    # "LiCE",
    "CET",
    "CEGP",
    "CaseBasedSACE",
]
