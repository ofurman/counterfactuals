#!/usr/bin/env bash
# Send a notification as sweep cells finish, and once more when the sweep ends.
#
# This is a plain background shell loop watching the .status files that
# run_dictum_experiments.sh writes per cell — no agent, no session to keep open.
# Start it any time, including after the sweep is already running.
#
# Usage:
#   # notify to your phone via ntfy.sh (no account needed, see PRIVACY below)
#   NTFY_TOPIC=my-unguessable-topic-name ./scripts/notify_experiments.sh --tag dictum &
#
#   # also stop when a specific sweep process exits
#   NTFY_TOPIC=... ./scripts/notify_experiments.sh --tag dictum --pid 48469 &
#
#   # email, reading SMTP settings from a file kept out of shell history
#   set -a; . ~/.config/cf-notify.env; set +a
#   ./scripts/notify_experiments.sh --tag dictum --final-only &
#
#   # post to a Slack/Discord-style incoming webhook instead
#   NOTIFY_WEBHOOK=https://hooks.slack.com/services/... ./scripts/notify_experiments.sh &
#
#   # only summarise at the end, no per-cell pings
#   ./scripts/notify_experiments.sh --tag dictum --final-only &
#
# Backends are independent and all optional; every enabled one receives each
# notification. With none configured it still logs to stdout, so redirect it
# somewhere useful:
#   ... ./scripts/notify_experiments.sh --tag dictum > notify.log 2>&1 &
#
#   NTFY_TOPIC       topic on ntfy.sh (or NTFY_SERVER for a self-hosted one)
#   NOTIFY_EMAIL_TO  recipient address; see scripts/send_email_notification.py
#                    for the SMTP_* settings it needs alongside this
#   NOTIFY_WEBHOOK   URL receiving {"text": "..."} as JSON
#   NOTIFY_LOCAL=1   macOS desktop notification on the machine running this
#
# PRIVACY: an ntfy.sh topic is a public channel — anyone who knows or guesses
# the name can read it. Use a long random topic, send nothing sensitive, or
# point NTFY_SERVER at your own instance.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# A non-interactive shell does not source the login profile, so resolve uv
# explicitly; the email backend runs through it.
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
UV="$(command -v uv || echo uv)"

TAG=dictum
WATCH_PID=""
INTERVAL=60
FINAL_ONLY=0
NTFY_SERVER="${NTFY_SERVER:-https://ntfy.sh}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag) TAG="$2"; shift 2 ;;
    --pid) WATCH_PID="$2"; shift 2 ;;
    --interval) INTERVAL="$2"; shift 2 ;;
    --final-only) FINAL_ONLY=1; shift ;;
    --ntfy-topic) NTFY_TOPIC="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

LOGS="$REPO_ROOT/results/$TAG/logs"

notify() {
  local title="$1" body="$2"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $title — $body"

  if [[ -n "${NTFY_TOPIC:-}" ]]; then
    curl -fsS -H "Title: $title" -d "$body" "$NTFY_SERVER/$NTFY_TOPIC" > /dev/null \
      || echo "  (ntfy delivery failed)" >&2
  fi

  if [[ -n "${NOTIFY_WEBHOOK:-}" ]]; then
    # Escape the payload with a JSON encoder rather than by hand: cell labels
    # contain slashes and could otherwise produce invalid JSON.
    local payload
    payload=$(TITLE="$title" BODY="$body" python3 -c \
      'import json, os; print(json.dumps({"text": os.environ["TITLE"] + ": " + os.environ["BODY"]}))')
    curl -fsS -H 'Content-Type: application/json' -d "$payload" "$NOTIFY_WEBHOOK" > /dev/null \
      || echo "  (webhook delivery failed)" >&2
  fi

  if [[ -n "${NOTIFY_EMAIL_TO:-}" ]]; then
    "$UV" run python -m scripts.send_email_notification "$title" "$body" > /dev/null \
      || echo "  (email delivery failed)" >&2
  fi

  if [[ "${NOTIFY_LOCAL:-0}" == "1" ]] && command -v osascript > /dev/null; then
    osascript -e "display notification \"$body\" with title \"$title\"" 2> /dev/null || true
  fi
}

sweep_alive() {
  # With no --pid, treat any running sweep process as the sweep.
  if [[ -n "$WATCH_PID" ]]; then
    kill -0 "$WATCH_PID" 2> /dev/null
  else
    pgrep -f "run_dictum_experiments.sh" > /dev/null 2>&1
  fi
}

# Cells already finished before this watcher started are recorded but not
# announced, so attaching mid-sweep does not replay the whole backlog.
seen=""
if [[ -d "$LOGS" ]]; then
  for st in "$LOGS"/*.status; do
    [[ -f "$st" ]] || continue
    seen="$seen $(basename "$st")"
  done
fi
already=$(echo "$seen" | wc -w | tr -d ' ')

echo "watching:  $LOGS"
echo "interval:  ${INTERVAL}s"
echo "pid:       ${WATCH_PID:-any run_dictum_experiments.sh}"
echo "backends:  ntfy=${NTFY_TOPIC:+yes} email=${NOTIFY_EMAIL_TO:+yes}" \
     "webhook=${NOTIFY_WEBHOOK:+yes} local=${NOTIFY_LOCAL:-0}"
echo "already finished at start: $already cell(s), not re-announced"
echo

ok_count=0
fail_count=0

while true; do
  if [[ -d "$LOGS" ]]; then
    for st in "$LOGS"/*.status; do
      [[ -f "$st" ]] || continue
      name="$(basename "$st")"
      case " $seen " in *" $name "*) continue ;; esac
      seen="$seen $name"

      read -r state rest < "$st" || true
      if [[ "$state" == "FAILED" ]]; then
        fail_count=$(( fail_count + 1 ))
        notify "Experiment FAILED" "$rest"
      else
        ok_count=$(( ok_count + 1 ))
        [[ "$FINAL_ONLY" -eq 1 ]] || notify "Experiment finished" "$rest"
      fi
    done
  fi

  if ! sweep_alive; then
    # One grace pass so the last cell's .status file is not missed if the
    # sweep exits between the scan above and this check.
    sleep 5
    for st in "$LOGS"/*.status; do
      [[ -f "$st" ]] || continue
      name="$(basename "$st")"
      case " $seen " in *" $name "*) continue ;; esac
      seen="$seen $name"
      read -r state rest < "$st" || true
      if [[ "$state" == "FAILED" ]]; then
        fail_count=$(( fail_count + 1 ))
        notify "Experiment FAILED" "$rest"
      else
        ok_count=$(( ok_count + 1 ))
        [[ "$FINAL_ONLY" -eq 1 ]] || notify "Experiment finished" "$rest"
      fi
    done

    total=$(echo "$seen" | wc -w | tr -d ' ')
    notify "Sweep '$TAG' done" \
      "$total cell(s) total, $ok_count ok and $fail_count failed since watching started"
    exit 0
  fi

  sleep "$INTERVAL"
done
