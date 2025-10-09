"""Local counterfactual methods package.

This module exposes local counterfactual method classes but guards
optional, heavy dependencies so that a simple import doesn't fail
when optional packages are not installed.
"""

import importlib
import warnings

__all__ = []


def _optional_from(module: str, name: str, alias: str | None = None):
    """Try to import ``name`` from ``module`` and expose it in this
    package globals. If import fails, set the name to None and warn.
    """
    as_name = alias or name
    try:
        mod = importlib.import_module(module, package=__package__)
        obj = getattr(mod, name)
        globals()[as_name] = obj
        __all__.append(as_name)
    except Exception as e:
        globals()[as_name] = None
        warnings.warn(
            f"Optional counterfactual method '{as_name}' could not be imported "
            f"from '{module}': {e}. Install optional dependencies to enable it.",
            ImportWarning,
        )


# Core/lightweight methods - prefer these to be available without extras
_optional_from(".ppcef.ppcef", "PPCEF")
_optional_from(".wach.wach", "WACH")
_optional_from(".wach.wach_ours", "WACH_OURS")
_optional_from(".artelt.artelt", "Artelt")
_optional_from(".regression_ppcef.ppcefr", "PPCEFR")

# Other local methods (may have optional deps)
_optional_from(".dice.dice", "DiceExplainerWrapper")
_optional_from(".cem.cem", "CEM_CF")
_optional_from(".ares.ares", "AReS", alias="ARES")
_optional_from(".sace.sace", "SACE")
_optional_from(".cet.cet", "CounterfactualExplanationTree", alias="CET")
_optional_from(".casebased_sace.casebased_sace", "CaseBasedSACE")

# Methods with heavy optional dependencies
_optional_from(".cegp.cegp", "CEGP")  # needs alibi/tensorflow
_optional_from(".lice.lice", "LiCE")  # needs spn package
