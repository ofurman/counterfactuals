#!/usr/bin/env bash
# Run the DICTUM-aligned sweep with notifications, as one detachable command.
#
# Starts run_dictum_experiments.sh and scripts/notify_experiments.sh together,
# binds the watcher to this sweep's PID so it exits when the sweep does, and
# checks the notification settings up front rather than after hours of running.
#
# Usage:
#   ./run_dictum_with_alerts.sh --detach          # survives the SSH session closing
#   ./run_dictum_with_alerts.sh                   # stay attached, e.g. inside tmux
#   ./run_dictum_with_alerts.sh --detach --methods dice --datasets adult
#
# Notification settings are read from --env-file (default ~/.config/cf-notify.env),
# which should contain at least:
#   NOTIFY_EMAIL_TO=you@example.com
#   SMTP_USER=you@gmail.com
#   SMTP_PASS=your-app-password
# Keep it chmod 600; see docs/dictum_alignment.md for the other backends.
#
# Anything after the recognised flags is passed straight to
# run_dictum_experiments.sh (--methods, --datasets, --seeds, --tag, --jobs, ...).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

ENV_FILE="$HOME/.config/cf-notify.env"
DETACH=0
FINAL_ONLY=0
TAG=dictum
PASSTHROUGH=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --detach) DETACH=1; shift ;;
    --env-file) ENV_FILE="$2"; shift 2 ;;
    --final-only) FINAL_ONLY=1; shift ;;
    --tag) TAG="$2"; PASSTHROUGH+=(--tag "$2"); shift 2 ;;
    *) PASSTHROUGH+=("$1"); shift ;;
  esac
done

SWEEP_LOG="$REPO_ROOT/sweep_${TAG}.log"
NOTIFY_LOG="$REPO_ROOT/notify_${TAG}.log"

# Re-exec under nohup so closing the terminal cannot SIGHUP the sweep. The
# child sees --detach removed, so it takes the branch below instead of looping.
if [[ "$DETACH" -eq 1 ]]; then
  CHILD=("$0" --env-file "$ENV_FILE")
  [[ "$FINAL_ONLY" -eq 1 ]] && CHILD+=(--final-only)
  [[ ${#PASSTHROUGH[@]} -gt 0 ]] && CHILD+=("${PASSTHROUGH[@]}")

  nohup "${CHILD[@]}" > "$SWEEP_LOG" 2>&1 &
  disown
  echo "detached: pid $!"
  echo "sweep log:  $SWEEP_LOG"
  echo "notify log: $NOTIFY_LOG"
  echo
  echo "Verify it survived by closing this session, reconnecting, and running:"
  echo "  pgrep -f run_dictum_experiments.sh"
  exit 0
fi

if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  set -a; . "$ENV_FILE"; set +a
  echo "loaded notification settings from $ENV_FILE"
else
  echo "WARNING: $ENV_FILE not found — the watcher will only log to $NOTIFY_LOG" >&2
fi

# Fail fast on a bad address or password. Hours into a sweep is the wrong time
# to discover the alerts were never going to arrive.
if [[ -n "${NOTIFY_EMAIL_TO:-}" ]]; then
  echo "checking email settings..."
  if uv run python -m scripts.send_email_notification \
      "Sweep '$TAG' started" \
      "Alerts are working. You will get one message per finished cell, plus a summary at the end."
  then
    echo "startup email sent to $NOTIFY_EMAIL_TO"
  else
    echo "ABORTING: could not send the startup email — fix the settings first." >&2
    echo "Test directly with: uv run python -m scripts.send_email_notification test test" >&2
    exit 1
  fi
fi

echo
echo "starting sweep, logging to $SWEEP_LOG"
./run_dictum_experiments.sh ${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"} &
SWEEP_PID=$!
echo "sweep pid: $SWEEP_PID"

WATCH=(./scripts/notify_experiments.sh --tag "$TAG" --pid "$SWEEP_PID")
[[ "$FINAL_ONLY" -eq 1 ]] && WATCH+=(--final-only)
"${WATCH[@]}" > "$NOTIFY_LOG" 2>&1 &
WATCH_PID=$!
echo "watcher pid: $WATCH_PID"
echo

SWEEP_STATUS=0
wait "$SWEEP_PID" || SWEEP_STATUS=$?

# The watcher notices the sweep is gone, drains the last .status files and
# sends the summary, so let it finish before this script exits.
wait "$WATCH_PID" 2> /dev/null || true

echo
echo "sweep exited with status $SWEEP_STATUS"
exit "$SWEEP_STATUS"
