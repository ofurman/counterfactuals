#!/usr/bin/env bash
# Inspect live PLGrid access and write slurm/cluster.env.
#
# Run this ON THE HELIOS LOGIN NODE before anything else. Nothing in this
# directory hardcodes an account or partition: every sbatch file sources
# cluster.env, and this script is what produces it.
#
# Usage:
#   ./slurm/preflight.sh                 # report only
#   ./slurm/preflight.sh --write         # report and write slurm/cluster.env
set -euo pipefail

WRITE=0
[[ "${1:-}" == "--write" ]] && WRITE=1

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

hr() { printf '\n== %s ==\n' "$1"; }

hr "hpc-grants (compute accounts)"
hpc-grants || echo "hpc-grants unavailable"

hr "hpc-fs (storage groups)"
hpc-fs || echo "hpc-fs unavailable"

hr "PLG_GROUPS_STORAGE"
printf '%s\n' "${PLG_GROUPS_STORAGE:-<undefined>}"

hr "partitions"
sinfo -o '%P %a %l %D %c %m %G' || true

hr "home quota"
quota -s 2>/dev/null || true
du -sh "$HOME" 2>/dev/null || true

hr "python / uv availability"
module avail 2>&1 | grep -iE 'ML-bundle|python|uv' | head -20 || true

hr "slurm mail capability"
# Many HPC sites disable outbound mail. If MailProg is unset or points at
# /bin/true, --mail-type will be silently ignored and no alerts will arrive.
scontrol show config 2>/dev/null | grep -iE 'MailProg|MailDomain' || \
  echo "MailProg not reported - Slurm email may be disabled on this cluster"

cat <<'NOTE'

== What to pick ==

These baseline pipelines are CPU-ONLY. run_dice_traintest_pipeline.py:345,
run_cchvae_traintest_pipeline.py:261 and run_dicoflex_traintest_pipeline.py:513
each set CUDA_VISIBLE_DEVICES=-1 at the top of main(), unconditionally. Do NOT
request --gres=gpu on them; pick a CPU account and CPU partition from the
hpc-grants / sinfo output above.

The GPU profile (plgcountercontex-gpu-gh200 / plgrid-gpu-gh200) is only
relevant to the TabDCE runs, which are not part of these scripts.

Check the partition MaxTime column against the walltimes in submit-all.sh
before submitting; CCHVAE on adult took 3.2 h of CF search alone locally.
NOTE

if [[ "$WRITE" -eq 1 ]]; then
  cat > "$HERE/cluster.env" <<'ENV'
# Verified live on Helios 2026-07-26 for grant plgcountercontex.
#
# The grant holds ONLY a gpu-gh200 allocation. sbatch --test-only rejects
# plgrid, plgrid-long and cpu with "Invalid account or account/partition
# combination", so the CPU-only baseline jobs have to run on the GPU
# partition. They are submitted WITHOUT --gres, which allocates cores on a
# GH200 node without reserving a GPU.
PLG_ACCOUNT=plgcountercontex-gpu-gh200
PLG_PARTITION=plgrid-gpu-gh200
PLG_GROUP=plggcfsgenwro
PROJECT_NAME=counterfactuals

# Kept identical across every method so the reported Time column is
# comparable between DiCE, CCHVAE and DiCoFlex. Changing this invalidates
# cross-method timing comparisons.
CPUS_PER_TASK=8

# Job alerts. Leave PLG_MAIL empty to disable email entirely.
# For job arrays Slurm sends ONE mail per array (not per task) unless
# ARRAY_TASKS is added to mail-type, so this stays quiet: roughly two
# messages per array rather than 130.
PLG_MAIL=lukasz.lenkiewicz@pwr.edu.pl
PLG_MAIL_TYPE=BEGIN,END,FAIL
ENV
  printf '\nWrote %s - edit the CHANGE-ME values before submitting.\n' "$HERE/cluster.env"
fi
