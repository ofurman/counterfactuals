#!/bin/bash
# Thin wrapper around scripts/run_constraint_setup_experiments.py.
# Forwards all args, e.g.:
#   ./run_constraint_setup_experiments.sh --datasets adult --setups 1 2
set -e
cd "$(dirname "$0")"
uv run python scripts/run_constraint_setup_experiments.py "$@"
