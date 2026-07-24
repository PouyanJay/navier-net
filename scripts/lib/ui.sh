# shellcheck shell=bash
# Shared terminal UI for the scripts/ suite. Source, don't execute:
#   source "$(dirname "$0")/lib/ui.sh"
#
# Targets bash 3.2 (macOS system bash): no associative arrays, no ${var^^},
# no ${var,,}. Colours and spinners auto-disable when stdout is not a TTY
# (CI logs stay clean); NO_COLOR/FORCE_COLOR/TERM=dumb are respected.

# ── Capability detection ─────────────────────────────────────────────────────

UI_COLOR=0
if [ -n "${FORCE_COLOR:-}" ]; then
  UI_COLOR=1
elif [ -z "${NO_COLOR:-}" ] && [ -t 1 ] && [ "${TERM:-dumb}" != "dumb" ]; then
  UI_COLOR=1
fi

UI_UNICODE=0
case "${LC_ALL:-}${LC_CTYPE:-}${LANG:-}" in
  *UTF-8* | *utf8*) UI_UNICODE=1 ;;
esac

if [ "$UI_COLOR" = 1 ]; then
  UI_PRIMARY=$'\033[36m'
  UI_SUCCESS=$'\033[32m'
  UI_WARN=$'\033[33m'
  UI_ERROR=$'\033[31m'
  UI_DIM=$'\033[2m'
  UI_BOLD=$'\033[1m'
  UI_RESET=$'\033[0m'
else
  UI_PRIMARY='' UI_SUCCESS='' UI_WARN='' UI_ERROR='' UI_DIM='' UI_BOLD='' UI_RESET=''
fi

if [ "$UI_UNICODE" = 1 ]; then
  UI_ICON_OK='✔' UI_ICON_FAIL='✖' UI_ICON_WARN='⚠' UI_ICON_INFO='ℹ' UI_ICON_SKIP='⊘' UI_ICON_STEP='▸'
else
  UI_ICON_OK='[ok]' UI_ICON_FAIL='[FAIL]' UI_ICON_WARN='[!]' UI_ICON_INFO='[i]' UI_ICON_SKIP='[-]' UI_ICON_STEP='>'
fi

# ── Banner and steps ─────────────────────────────────────────────────────────

ui::banner() { # <title> [context]
  printf '\n%s%s%s%s\n' "$UI_BOLD" "$UI_PRIMARY" "$1" "$UI_RESET"
  printf '%s%s%s\n\n' "$UI_DIM" "${2:+$2 — }$(date '+%Y-%m-%d %H:%M:%S')" "$UI_RESET"
}

ui::step() { # <n> <total> <description>
  printf '%s%s %s/%s%s %s%s%s\n' \
    "$UI_PRIMARY" "$UI_ICON_STEP" "$1" "$2" "$UI_RESET" "$UI_BOLD" "$3" "$UI_RESET"
}

ui::ok() { printf '  %s%s%s %s\n' "$UI_SUCCESS" "$UI_ICON_OK" "$UI_RESET" "$1"; }
ui::fail() { printf '  %s%s %s%s\n' "$UI_ERROR" "$UI_ICON_FAIL" "$1" "$UI_RESET" >&2; }
ui::warn() { printf '  %s%s %s%s\n' "$UI_WARN" "$UI_ICON_WARN" "$1" "$UI_RESET"; }
ui::skip() { printf '  %s%s %s%s\n' "$UI_DIM" "$UI_ICON_SKIP" "$1" "$UI_RESET"; }
ui::info() { printf '  %s%s%s %s\n' "$UI_PRIMARY" "$UI_ICON_INFO" "$UI_RESET" "$1"; }

# Fatal error with the exact command that fixes it, then exit.
ui::die() { # <message> [remediation]
  ui::fail "$1"
  [ -n "${2:-}" ] && printf '    %sFix:%s %s\n' "$UI_BOLD" "$UI_RESET" "$2" >&2
  exit 1
}

# ── Spinner ──────────────────────────────────────────────────────────────────

UI_SPINNER_PID=''
UI_SPINNER_LABEL=''

ui::spinner_start() { # <label>
  UI_SPINNER_LABEL="$1"
  if [ ! -t 1 ]; then # no spinner off-TTY: print the label once and return
    printf '  %s...\n' "$1"
    return 0
  fi
  (
    frames="|/-\\"
    [ "$UI_UNICODE" = 1 ] && frames='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
    i=0
    while :; do
      f=${frames:$((i % ${#frames})):1}
      printf '\r  %s%s%s %s' "$UI_PRIMARY" "$f" "$UI_RESET" "$UI_SPINNER_LABEL"
      i=$((i + 1))
      sleep 0.1
    done
  ) &
  UI_SPINNER_PID=$!
}

ui::spinner_stop() {
  if [ -n "$UI_SPINNER_PID" ]; then
    kill "$UI_SPINNER_PID" 2>/dev/null
    wait "$UI_SPINNER_PID" 2>/dev/null
    UI_SPINNER_PID=''
    printf '\r\033[K' # clear the spinner line before the status line replaces it
  fi
  UI_SPINNER_LABEL=''
}

# Run a command behind a spinner, capturing all output. Success prints one ✔
# line; failure dumps the captured output with the exit code. Returns the
# command's exit code — callers under `set -e` should `|| handle` it.
ui::run() { # <label> <command> [args...]
  local label="$1" out rc=0
  shift
  out="$(mktemp "${TMPDIR:-/tmp}/naviernet-ui.XXXXXX")"
  ui::spinner_start "$label"
  "$@" >"$out" 2>&1 || rc=$?
  ui::spinner_stop
  if [ "$rc" -eq 0 ]; then
    ui::ok "$label"
  else
    ui::fail "$label (exit $rc)"
    sed 's/^/      /' "$out" >&2
  fi
  rm -f "$out"
  return "$rc"
}

# ── Summary dashboard ────────────────────────────────────────────────────────
# Parallel indexed arrays instead of an associative array (bash 3.2).

UI_SUMMARY_LABELS=()
UI_SUMMARY_VALUES=()
UI_SUMMARY_STATUSES=()

ui::summary_row() { # <label> <value> <ok|warn|fail|skip>
  UI_SUMMARY_LABELS[${#UI_SUMMARY_LABELS[@]}]="$1"
  UI_SUMMARY_VALUES[${#UI_SUMMARY_VALUES[@]}]="$2"
  UI_SUMMARY_STATUSES[${#UI_SUMMARY_STATUSES[@]}]="$3"
}

ui::summary() { # [title]
  local i icon colour
  printf '\n%s%s%s\n' "$UI_BOLD" "${1:-Summary}" "$UI_RESET"
  for i in $(seq 0 $((${#UI_SUMMARY_LABELS[@]} - 1))); do
    case "${UI_SUMMARY_STATUSES[$i]}" in
      ok) icon="$UI_ICON_OK" colour="$UI_SUCCESS" ;;
      warn) icon="$UI_ICON_WARN" colour="$UI_WARN" ;;
      fail) icon="$UI_ICON_FAIL" colour="$UI_ERROR" ;;
      *) icon="$UI_ICON_SKIP" colour="$UI_DIM" ;;
    esac
    printf '  %s%s%s %-22s %s%s%s\n' \
      "$colour" "$icon" "$UI_RESET" "${UI_SUMMARY_LABELS[$i]}" \
      "$UI_DIM" "${UI_SUMMARY_VALUES[$i]}" "$UI_RESET"
  done
  printf '%s%ss elapsed%s\n' "$UI_DIM" "$SECONDS" "$UI_RESET"
}
