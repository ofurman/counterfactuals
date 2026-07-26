#!/usr/bin/env bash
# Create the project layout on Helios: code in $HOME, heavy files in group storage.
# Run once on the login node, after preflight.sh --write and editing cluster.env.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/cluster.env"

: "${PLG_GROUP:?Set PLG_GROUP in slurm/cluster.env}"
: "${PROJECT_NAME:?Set PROJECT_NAME in slurm/cluster.env}"
: "${PLG_GROUPS_STORAGE:?PLG_GROUPS_STORAGE is not defined on this host}"

STORE="$PLG_GROUPS_STORAGE/$PLG_GROUP"
# Deliberately NOT $STORE/users/$USER. On plggcfsgenwro the users/ directory
# already exists as drwx--S--- owned by plgofurman, so it is private to them
# and no other group member can create anything beneath it. The group root is
# drwxrws--- (group-writable, setgid), so each member takes a top-level
# directory of their own instead.
HEAVY="$STORE/$USER/$PROJECT_NAME"
PROJECT="$HOME/projects/$PROJECT_NAME"

test -d "$STORE"
test -w "$STORE"

umask 077
mkdir -p "$PROJECT" \
  "$HEAVY/envs" \
  "$HEAVY/local-aarch64" \
  "$HEAVY/cache/uv" \
  "$HEAVY/cache/pip" \
  "$HEAVY/cache/torch" \
  "$HEAVY/datasets" \
  "$HEAVY/models" \
  "$HEAVY/outputs" \
  "$HEAVY/results"
chmod 700 "$HEAVY"

ensure_link() {
  local link_path="$1" target="$2"
  if [[ -L "$link_path" ]]; then
    if [[ "$(readlink "$link_path")" == "$target" ]]; then
      return
    fi
    printf 'Refusing to replace existing symlink: %s -> %s\n' \
      "$link_path" "$(readlink "$link_path")" >&2
    return 1
  fi
  if [[ -e "$link_path" ]]; then
    printf 'Refusing to replace existing path: %s\n' "$link_path" >&2
    return 1
  fi
  ln -s "$target" "$link_path"
}

ensure_link "$PROJECT/.local" "$HEAVY/local-aarch64"
ensure_link "$PROJECT/.cache" "$HEAVY/cache"
ensure_link "$PROJECT/.venv" "$HEAVY/envs/default"
ensure_link "$PROJECT/models" "$HEAVY/models"
ensure_link "$PROJECT/outputs" "$HEAVY/outputs"
ensure_link "$PROJECT/results" "$HEAVY/results"

printf 'PROJECT=%s\nHEAVY=%s\n' "$PROJECT" "$HEAVY"
for item in .local .cache .venv models outputs results; do
  printf '%s -> %s\n' "$PROJECT/$item" "$(readlink "$PROJECT/$item")"
done

cat <<'NOTE'

data_train_test_val/ is 37 MB and is deliberately NOT symlinked: sync-code.sh
copies it into the project directory so the hydra dataset.*_data_path values
resolve without further overrides.
NOTE
