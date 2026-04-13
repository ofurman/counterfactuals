"""Pipeline runners for counterfactual generation methods.

Each runner inherits from PipelineRunner and implements the search_counterfactuals method.
Simple methods only need to implement search_counterfactuals(), while group methods
may also override calculate_metrics() for specialized metrics.
"""

from counterfactuals.pipelines.runners.ares_runner import AReSPipelineRunner
from counterfactuals.pipelines.runners.artelt_runner import ArteltPipelineRunner
from counterfactuals.pipelines.runners.cadex_runner import CADEXPipelineRunner
from counterfactuals.pipelines.runners.casebased_sace_runner import CaseBasedSACEPipelineRunner
from counterfactuals.pipelines.runners.cchvae_pairwise_runner import CCHVAEPairwisePipelineRunner
from counterfactuals.pipelines.runners.cchvae_runner import CCHVAEPipelineRunner
from counterfactuals.pipelines.runners.ceflow_runner import CeFlowPipelineRunner
from counterfactuals.pipelines.runners.cegp_runner import CEGPPipelineRunner
from counterfactuals.pipelines.runners.cem_runner import CEMPipelineRunner
from counterfactuals.pipelines.runners.cet_runner import CETPipelineRunner
from counterfactuals.pipelines.runners.dice_pairwise_runner import DiCEPairwisePipelineRunner
from counterfactuals.pipelines.runners.dice_runner import DiCEPipelineRunner
from counterfactuals.pipelines.runners.glance_runner import GLANCEPipelineRunner
from counterfactuals.pipelines.runners.globe_ce_runner import GLOBECEPipelineRunner
from counterfactuals.pipelines.runners.group_globe_ce_runner import GroupGLOBECEPipelineRunner
from counterfactuals.pipelines.runners.pairwise_mixin import PairwiseMixin
from counterfactuals.pipelines.runners.ppcef_runner import PPCEFPipelineRunner
from counterfactuals.pipelines.runners.ppcefr_runner import PPCEFRPipelineRunner
from counterfactuals.pipelines.runners.pumal_runner import PUMALPipelineRunner
from counterfactuals.pipelines.runners.regional_globe_ce_runner import RegionalGLOBECEPipelineRunner
from counterfactuals.pipelines.runners.tabdce_pairwise_runner import TabDCEPairwisePipelineRunner
from counterfactuals.pipelines.runners.tabdce_runner import TabDCEPipelineRunner
from counterfactuals.pipelines.runners.tcrex_runner import TCRExPipelineRunner
from counterfactuals.pipelines.runners.wach_ours_runner import WACHOURSPipelineRunner
from counterfactuals.pipelines.runners.wach_runner import WACHPipelineRunner

__all__ = [
    "PairwiseMixin",
    "PPCEFPipelineRunner",
    "DiCEPipelineRunner",
    "DiCEPairwisePipelineRunner",
    "CCHVAEPipelineRunner",
    "CCHVAEPairwisePipelineRunner",
    "TabDCEPipelineRunner",
    "TabDCEPairwisePipelineRunner",
    "ArteltPipelineRunner",
    "CEGPPipelineRunner",
    "CEMPipelineRunner",
    "CETPipelineRunner",
    "CADEXPipelineRunner",
    "CaseBasedSACEPipelineRunner",
    "WACHOURSPipelineRunner",
    "WACHPipelineRunner",
    "GLANCEPipelineRunner",
    "PUMALPipelineRunner",
    "AReSPipelineRunner",
    "CeFlowPipelineRunner",
    "GLOBECEPipelineRunner",
    "GroupGLOBECEPipelineRunner",
    "RegionalGLOBECEPipelineRunner",
    "TCRExPipelineRunner",
    "PPCEFRPipelineRunner",
]
