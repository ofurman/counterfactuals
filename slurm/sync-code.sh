#!/usr/bin/env bash
# Push code + splits + dataset configs to Helios. Run from the repo root, locally.
#
# Usage:
#   PLG_LOGIN=plgyourlogin ./slurm/sync-code.sh
#   PLG_LOGIN=plgyourlogin PLG_HOST=login01.helios.cyfronet.pl ./slurm/sync-code.sh
set -euo pipefail

: "${PLG_LOGIN:?Set PLG_LOGIN to your PLGrid login}"
PLG_HOST="${PLG_HOST:-login01.helios.cyfronet.pl}"
PROJECT_NAME="${PROJECT_NAME:-counterfactuals}"
# Destination path on the remote, relative to the login home. Helios uses the
# projects/<name> layout; override for any other host (e.g. a workstation that
# already keeps the repo at genwro/counterfactuals).
REMOTE_PATH="${REMOTE_PATH:-projects/$PROJECT_NAME}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# models/ is 1.4 GB of previously trained artefacts and outputs/ is prior
# results; both are excluded so a seed run cannot silently reuse or overwrite
# them. Each seed retrains from scratch into its own results root.
#
# No --delete/--delete-excluded on purpose. macOS ships openrsync (protocol
# 29), which does not implement --delete-excluded and aborts the transfer with
# "unexpected end of file". Omitting deletion also means a re-sync can never
# remove results already written on the cluster.
#
# The remote parent must exist; openrsync has no --mkpath.
ssh "$PLG_LOGIN@$PLG_HOST" "mkdir -p $REMOTE_PATH"

# The heavy directories are anchored with a leading slash so they only match
# at the transfer root. An unanchored 'models' also matches the Python package
# counterfactuals/models/ and cel/models/, which silently ships a broken
# install ("ModuleNotFoundError: No module named 'counterfactuals.models'").
# Patterns without a leading slash below are intended to match at any depth.
rsync -av \
  --exclude '.git' \
  --exclude '.env' \
  --exclude '.venv' \
  --exclude '.cache' \
  --exclude '.local' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '*.out' \
  --exclude '*.err' \
  --exclude '/models' \
  --exclude '/outputs' \
  --exclude '/results' \
  --exclude '/explanations' \
  --exclude '/dice_results' \
  --exclude '/notebooks' \
  --exclude '/methods_notebooks' \
  ./ "$PLG_LOGIN@$PLG_HOST:$REMOTE_PATH/"

printf '\nSynced to %s:projects/%s\n' "$PLG_HOST" "$PROJECT_NAME"
printf 'Next, on the login node:\n'
printf '  cd ~/projects/%s\n' "$PROJECT_NAME"
printf '  ./slurm/preflight.sh --write   # then edit slurm/cluster.env\n'
printf '  ./slurm/bootstrap-storage.sh\n'
printf '  sbatch slurm/setup-env.sbatch\n'
