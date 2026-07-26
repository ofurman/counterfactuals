#!/bin/bash -l
#SBATCH --job-name=cf-env-setup
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/setup-env-%j.out
#SBATCH --error=slurm/logs/setup-env-%j.err

# Builds the Python environment on a COMPUTE node so wheels match the compute
# architecture. Helios login nodes are x86_64 while GH200 compute is aarch64.
#
# Python is pinned to 3.11 deliberately. ML-bundle/25.10 ships Python 3.13,
# and the dependency chain alibi -> spacy -> blis<0.8.0 has no cp313 aarch64
# wheel, so pip falls back to compiling blis 0.7.11 from source and dies in
# Cython ("CompileError: blis/py.pyx"). blis 0.7.11 does publish
# cp311-manylinux_2_17_aarch64, so 3.11 installs from wheels. 3.11 is also
# what the project targets.
#
# uv is not present in the module set, so it is bootstrapped into a throwaway
# venv and then used to resolve uv.lock exactly.

set -euo pipefail

HERE="$SLURM_SUBMIT_DIR/slurm"
# shellcheck disable=SC1091
source "$HERE/cluster.env"

: "${PLG_GROUP:?Set PLG_GROUP in slurm/cluster.env}"
PROJECT="$HOME/projects/$PROJECT_NAME"
HEAVY="$PLG_GROUPS_STORAGE/$PLG_GROUP/$USER/$PROJECT_NAME"
BOOT="$HEAVY/envs/bootstrap"
ENV_DIR="$HEAVY/envs/default"

export XDG_CACHE_HOME="$HEAVY/cache"
export UV_CACHE_DIR="$HEAVY/cache/uv"
export PIP_CACHE_DIR="$HEAVY/cache/pip"
export TORCH_HOME="$HEAVY/cache/torch"
export UV_PYTHON_INSTALL_DIR="$HEAVY/cache/uv-python"

module purge
module load ML-bundle/25.10

cd "$PROJECT"

echo "compute arch: $(uname -m)"
test "$(uname -m)" = aarch64 || echo "WARNING: not aarch64, wheels may not match compute nodes"

# A previous failed attempt can leave a partially populated 3.13 env behind.
if [[ -e "$ENV_DIR" && ! -x "$ENV_DIR/bin/python" ]]; then
  echo "removing incomplete environment at $ENV_DIR"
  rm -rf "$ENV_DIR"
fi
if [[ -x "$ENV_DIR/bin/python" ]]; then
  ver=$("$ENV_DIR/bin/python" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
  if [[ "$ver" != "3.11" ]]; then
    echo "existing env is Python $ver, rebuilding for 3.11"
    rm -rf "$ENV_DIR"
  fi
fi

if [[ ! -x "$BOOT/bin/uv" ]]; then
  echo "bootstrapping uv"
  rm -rf "$BOOT"
  python -m venv "$BOOT"
  "$BOOT/bin/pip" install --quiet --upgrade pip
  "$BOOT/bin/pip" install --quiet uv
fi
UV="$BOOT/bin/uv"
"$UV" --version

echo "installing CPython 3.11 via uv"
"$UV" python install 3.11

echo "syncing locked dependencies into $ENV_DIR"
UV_PROJECT_ENVIRONMENT="$ENV_DIR" "$UV" sync --frozen --python 3.11

test -L "$PROJECT/.venv"
"$PROJECT/.venv/bin/python" - <<'PY'
import platform
import numpy
import torch

print("machine:", platform.machine())
print("python:", platform.python_version())
print("numpy:", numpy.__version__)
print("torch:", torch.__version__)
PY

"$PROJECT/.venv/bin/python" -c \
  "from counterfactuals.pipelines.nodes.seeding import set_global_seed; set_global_seed(0); print('seeding import OK')"

echo "environment ready"
