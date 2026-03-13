#!/usr/bin/env bash
set -euo pipefail

./scripts/run_calculate_metrics.sh Artelt results/artelt
# ./scripts/run_calculate_metrics.sh CADEX results/cadex
# ./scripts/run_calculate_metrics.sh CaseBasedSACE results/case_based_sace
# ./scripts/run_calculate_metrics.sh CCHVAE results/cchvae
# ./scripts/run_calculate_metrics.sh CEGP results/cegp
# ./scripts/run_calculate_metrics.sh CEM_CF results/cem
# ./scripts/run_calculate_metrics.sh DiceExplainerWrapper results/dice_explainer_wrapper
# ./scripts/run_calculate_metrics.sh GlobalGLANCE results/global_glance
# ./scripts/run_calculate_metrics.sh GLOBE_CE results/globe_ce
# ./scripts/run_calculate_metrics.sh GroupGLANCE results/group_glance
# ./scripts/run_calculate_metrics.sh PPCEF results/ppcef
# ./scripts/run_calculate_metrics.sh TCREx results/tcrex
# ./scripts/run_calculate_metrics.sh WACH_OURS results/wach_ours
# ./scripts/run_calculate_metrics.sh AReS results/ares
