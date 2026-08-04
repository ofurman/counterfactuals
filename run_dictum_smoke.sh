#!/usr/bin/env bash
# Run the single cheapest cell of the DICTUM-aligned sweep, then score it.
#
# Use this to check the setup end to end — configs, shared classifier, both flip
# directions, CSV layout, and the scorer — before committing to the full sweep.
#
# The cell is DiCE on lending-club at seed 42. lending-club is the cheapest of
# the five datasets (31 one-hot columns against 42-91 for the others), and DiCE
# is the cheapest of the three methods: it generates in one batched call, where
# CCHVAE loops its sampler 100 times per factual and DiCoFlex trains a flow.
# Measured on an M-series laptop: ~2.5 min at full fidelity, ~30 s with --quick.
#
# Usage:
#   ./run_dictum_smoke.sh              # full fidelity, real numbers
#   ./run_dictum_smoke.sh --quick      # shrunken epochs, plumbing check only
#   ./run_dictum_smoke.sh --method cchvae --dataset bank --seed 43
#   ./run_dictum_smoke.sh --keep       # leave results/<tag> in place afterwards
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
UV="$(command -v uv || true)"
if [[ -z "$UV" ]]; then
  echo "uv not found on PATH ($PATH)" >&2
  exit 1
fi

METHOD=dice
DATASET=lending-club
SEED=42
TAG=dictum-smoke
QUICK=0
KEEP=0
N_TEST_SAMPLES=2000

while [[ $# -gt 0 ]]; do
  case "$1" in
    --method) METHOD="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --quick) QUICK=1; shift ;;
    --keep) KEEP=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

OVERRIDES=()
if [[ "$QUICK" -eq 1 ]]; then
  # Enough to exercise every code path, far too little to train anything.
  # Numbers from a --quick run are meaningless; it answers "does it run".
  N_TEST_SAMPLES=50
  OVERRIDES=(--override disc_model.epochs=5 --override gen_model.epochs=3)
  echo "QUICK MODE: shrunken training. Metrics below are NOT meaningful."
  echo
fi

# The scorer labels methods differently from the runner's CLI names.
case "$METHOD" in
  dice) SCORER_METHOD=DiCE ;;
  cchvae) SCORER_METHOD=CCHVAE ;;
  dicoflex) SCORER_METHOD=DiCoFlex ;;
  *) echo "Unknown method '$METHOD'" >&2; exit 1 ;;
esac

RESULTS="$REPO_ROOT/results/$TAG"
rm -rf "$RESULTS"

echo "cell:     $METHOD / $DATASET / seed $SEED"
echo "results:  $RESULTS"
echo

START=$SECONDS

# Delegating to the full runner keeps the hydra overrides in exactly one place,
# so this script cannot drift away from what the real sweep does.
# The array is only expanded when non-empty: bash 3.2 under `set -u` treats
# "${EMPTY[@]}" as an unbound variable.
RUN_CMD=(
  ./run_dictum_experiments.sh
  --methods "$METHOD"
  --datasets "$DATASET"
  --seeds "$SEED"
  --tag "$TAG"
  --n-test-samples "$N_TEST_SAMPLES"
)
if [[ ${#OVERRIDES[@]} -gt 0 ]]; then RUN_CMD+=("${OVERRIDES[@]}"); fi
"${RUN_CMD[@]}"

RUN_SECONDS=$(( SECONDS - START ))
echo
echo "=== scoring with DICTUM metric definitions ==="
"$UV" run python -m scripts.compute_dictum_metrics \
  --results-root "$RESULTS" \
  --datasets "$DATASET" \
  --seeds "$SEED" \
  --methods "$SCORER_METHOD" \
  --output "$RESULTS/dictum_metrics"

echo
echo "=== $METHOD / $DATASET / seed $SEED — ${RUN_SECONDS}s ==="
cat "$RESULTS/dictum_metrics.md"

if [[ "$KEEP" -eq 1 ]]; then
  echo
  echo "Kept: $RESULTS"
else
  rm -rf "$RESULTS"
fi
