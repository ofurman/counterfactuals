"""Pipeline runners for counterfactual generation methods.

Each runner inherits from PipelineRunner and implements the search_counterfactuals method.
Simple methods only need to implement search_counterfactuals(), while group methods
may also override calculate_metrics() for specialized metrics.

Imports are wrapped individually so a missing optional dependency for one runner
(e.g. casebased_sace, cet, ceflow, pumal, tabdce) does not break import of the rest.
"""

import importlib
import logging

_logger = logging.getLogger(__name__)

_RUNNERS = {
    "AReSPipelineRunner": "cel.pipelines.runners.ares_runner",
    "ArteltPipelineRunner": "cel.pipelines.runners.artelt_runner",
    "CADEXPipelineRunner": "cel.pipelines.runners.cadex_runner",
    "CaseBasedSACEPipelineRunner": "cel.pipelines.runners.casebased_sace_runner",
    "CCHVAEPairwisePipelineRunner": "cel.pipelines.runners.cchvae_pairwise_runner",
    "CCHVAEPipelineRunner": "cel.pipelines.runners.cchvae_runner",
    "CeFlowPipelineRunner": "cel.pipelines.runners.ceflow_runner",
    "CEGPPipelineRunner": "cel.pipelines.runners.cegp_runner",
    "CEMPipelineRunner": "cel.pipelines.runners.cem_runner",
    "CETPipelineRunner": "cel.pipelines.runners.cet_runner",
    "DiCEPipelineRunner": "cel.pipelines.runners.dice_runner",
    "GLANCEPipelineRunner": "cel.pipelines.runners.glance_runner",
    "GLOBECEPipelineRunner": "cel.pipelines.runners.globe_ce_runner",
    "GroupGLOBECEPipelineRunner": "cel.pipelines.runners.group_globe_ce_runner",
    "PairwiseMixin": "cel.pipelines.runners.pairwise_mixin",
    "PPCEFPipelineRunner": "cel.pipelines.runners.ppcef_runner",
    "PPCEFRPipelineRunner": "cel.pipelines.runners.ppcefr_runner",
    "PUMALPipelineRunner": "cel.pipelines.runners.pumal_runner",
    "TabDCEPairwisePipelineRunner": "cel.pipelines.runners.tabdce_pairwise_runner",
    "TabDCEPipelineRunner": "cel.pipelines.runners.tabdce_runner",
    "TCRExPipelineRunner": "cel.pipelines.runners.tcrex_runner",
    "WACHPipelineRunner": "cel.pipelines.runners.wach_runner",
}

__all__ = list(_RUNNERS)

for _cls_name, _module_path in _RUNNERS.items():
    try:
        _module = importlib.import_module(_module_path)
        globals()[_cls_name] = getattr(_module, _cls_name)
    except Exception as exc:
        _logger.debug("Skipping %s (%s): %s", _cls_name, _module_path, exc)
        globals()[_cls_name] = None
