"""Local counterfactual methods package."""

from .artelt import Artelt
from .c_chvae.c_chvae import CCHVAE
from .casebased_sace.casebased_sace import CaseBasedSACE
from .cegp.cegp import CEGP
from .cem.cem import CEM_CF
from .ceflow import CeFlow
from .cet.cet import CounterfactualExplanationTree as CET
from .dice.dice import DICE
from .dicoflex import DiCoFlex

# from .lice.lice import LiCE
from .ppcef.ppcef import PPCEF
from .regression_ppcef.ppcefr import PPCEFR
from .sace.sace import SACE
from .tabdce.tabdce import TabDCE
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
    "DiCoFlex",
    "WACH",
    "WACH_OURS",
    "Artelt",
    "PPCEFR",
    "DICE",
    "CEM_CF",
    "CeFlow",
    "SACE",
    # "LiCE",
    "CET",
    "CEGP",
    "CaseBasedSACE",
    "TabDCE",
]
