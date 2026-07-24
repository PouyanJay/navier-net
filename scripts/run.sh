#!/usr/bin/env bash
# Boot the platform stack: the FastAPI server and the Vite dev server.
#
# Ports are allocated automatically: each service prefers its default (API
# 8000, web 5173; overridable via .env), and when a port is held by a foreign
# process the next free one is used. Orphaned servers from previous runs are
# recognised by their command line (they run from this repo) and stopped
# before starting, so reruns always restart cleanly.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=scripts/lib/ui.sh
source "$ROOT/scripts/lib/ui.sh"
# shellcheck source=scripts/lib/checks.sh
source "$ROOT/scripts/lib/checks.sh"
cd "$ROOT"

STATE_DIR="$ROOT/.run-state"
DEFAULT_API_PORT=8000
DEFAULT_WEB_PORT=5173
API_READY_TIMEOUT=60 # first boot imports torch; generous on purpose
WEB_READY_TIMEOUT=30
PORT_SCAN_SPAN=20 # how far above the preferred port to look for a free one

# Our services are identified by command line, pinned to this repo so no
# other project's server can ever match. The API needs a cwd check on top:
# macOS re-execs venv python as the framework binary, so the venv path never
# appears in its argv.
api_pids() {
  local pid
  for pid in $(pgrep -f -- '-m naviernet_api' 2>/dev/null || true); do
    if lsof -a -p "$pid" -d cwd -Fn 2>/dev/null | grep -qx "n$ROOT"; then
      echo "$pid"
    fi
  done
  return 0
}

web_pids() {
  pgrep -f "$ROOT/apps/web/node_modules/.bin/vite" 2>/dev/null || true
}

usage() {
  cat <<'EOF'
Usage: scripts/run.sh [mode] [-h|--help]

Modes (default: start API + web dev server):
  --api-only   Start only the API server
  --web-only   Start only the Vite dev server (expects the API to be running)
  --prod       Build the web app, then serve API + UI from one process
  --stop       Stop everything this script started (including orphans)
  --status     Show what is running, on which ports

Preferred ports come from .env (NAVIERNET_API_PORT, NAVIERNET_WEB_PORT) or
default to 8000/5173; busy ports fall through to the next free one.
State lives in .run-state/ (PIDs, ports, logs).
EOF
}

MODE=all
case "${1:-}" in
  '') ;;
  --api-only) MODE=api ;;
  --web-only) MODE=web ;;
  --prod) MODE=prod ;;
  --stop) MODE=stop ;;
  --status) MODE=status ;;
  -h | --help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    ui::die "Unknown flag: $1"
    ;;
esac
if [ "$#" -gt 1 ]; then
  usage >&2
  ui::die "Unexpected extra arguments: ${*:2} (modes are mutually exclusive)"
fi

# ── Port helpers ─────────────────────────────────────────────────────────────

port_listener_pid() { # <port> — pid listening on the port, empty if free
  lsof -nP -ti "tcp:$1" -sTCP:LISTEN 2>/dev/null | head -n1
}

# Prefer the requested port; if a foreign process holds it, scan upward for
# the next free one (our own processes were already stopped by this point).
allocate_port() { # <preferred> <service> — echoes the chosen port
  local preferred="$1" service="$2" port="$1" holder
  local limit=$((preferred + PORT_SCAN_SPAN))
  while [ "$port" -le "$limit" ]; do
    holder="$(port_listener_pid "$port")"
    if [ -z "$holder" ]; then
      # stdout is the return channel (command substitution) — warn on stderr.
      if [ "$port" -ne "$preferred" ]; then
        ui::warn "Port $preferred is busy ($(ps -o comm= -p "$(port_listener_pid "$preferred")" 2>/dev/null || echo unknown)) — $service moves to $port" >&2
      fi
      echo "$port"
      return 0
    fi
    port=$((port + 1))
  done
  ui::die "No free port in $preferred-$limit for $service" "scripts/run.sh --stop, or free port $preferred"
}

# ── Process helpers ──────────────────────────────────────────────────────────

wait_gone() { # <pid> — wait up to 5s for exit, then SIGKILL
  local i=0
  while kill -0 "$1" 2>/dev/null; do
    i=$((i + 1))
    if [ "$i" -ge 50 ]; then
      kill -9 "$1" 2>/dev/null || true
      return 0
    fi
    sleep 0.1
  done
}

# Stop every process of ours: the pidfile-tracked one and any orphan from a
# run whose state was lost.
stop_service() { # <name> <pid-lister-fn> — returns 0 if anything was stopped
  local name="$1" list_pids="$2" stopped=1 pid
  if [ -f "$STATE_DIR/$name.pid" ]; then
    pid="$(cat "$STATE_DIR/$name.pid")"
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      wait_gone "$pid"
      ui::ok "Stopped $name (pid $pid)"
      stopped=0
    fi
    rm -f "$STATE_DIR/$name.pid" "$STATE_DIR/$name.port"
  fi
  for pid in $("$list_pids"); do
    kill "$pid" 2>/dev/null || true
    wait_gone "$pid"
    ui::warn "Stopped orphan $name (pid $pid)"
    stopped=0
  done
  return "$stopped"
}

wait_ready() { # <url> <timeout-seconds>
  local i=0
  until curl -sf -o /dev/null "$1"; do
    i=$((i + 1))
    [ "$i" -ge "$2" ] && return 1
    sleep 1
  done
}

die_with_log() { # <name> — readiness failed; show the evidence
  ui::fail "$1 did not become ready (last log lines below)"
  tail -n 20 "$STATE_DIR/$1.log" | sed 's/^/      /' >&2
  ui::die "$1 failed to start" "tail -f .run-state/$1.log"
}

# Mode-aware: an --api-only/--web-only session must never tear down the
# other service — it may belong to a second terminal.
on_interrupt() {
  ui::spinner_stop 2>/dev/null || true
  printf '\n'
  ui::warn "Interrupted — stopping services"
  case "$MODE" in
    api | prod) stop_service api api_pids || true ;;
    web) stop_service web web_pids || true ;;
    *)
      stop_service api api_pids || true
      stop_service web web_pids || true
      ;;
  esac
  exit 130
}

# ── Service starters ─────────────────────────────────────────────────────────

API_PORT=''
WEB_PORT=''

start_api() {
  API_PORT="$(allocate_port "${NAVIERNET_API_PORT:-$DEFAULT_API_PORT}" API)"
  # CORS list covers the web port even though the dev proxy makes it same-origin.
  NAVIERNET_API_PORT="$API_PORT" \
    NAVIERNET_CORS_ORIGINS="http://localhost:${WEB_PORT:-$DEFAULT_WEB_PORT},http://127.0.0.1:${WEB_PORT:-$DEFAULT_WEB_PORT}" \
    nohup .venv/bin/python -m naviernet_api >"$STATE_DIR/api.log" 2>&1 &
  echo $! >"$STATE_DIR/api.pid"
  echo "$API_PORT" >"$STATE_DIR/api.port"
  ui::spinner_start "API starting on :$API_PORT"
  wait_ready "http://127.0.0.1:$API_PORT/healthz" "$API_READY_TIMEOUT" || {
    ui::spinner_stop
    die_with_log api
  }
  ui::spinner_stop
  ui::ok "API ready on http://127.0.0.1:$API_PORT"
}

start_web() {
  # In `all` mode the port was already reserved (the API's CORS list names it).
  [ -n "$WEB_PORT" ] || WEB_PORT="$(allocate_port "${NAVIERNET_WEB_PORT:-$DEFAULT_WEB_PORT}" web)"
  # NAVIERNET_API_PORT tells vite.config.ts where to proxy /api; --strictPort
  # because the port was just verified free — silent drift would desync state.
  # Absolute vite path: the orphan sweep recognises our processes by the
  # $ROOT-anchored command line, so a relative argv would make them invisible.
  (
    cd apps/web \
      && NAVIERNET_API_PORT="${API_PORT:-$(cat "$STATE_DIR/api.port" 2>/dev/null || echo "$DEFAULT_API_PORT")}" \
        exec nohup "$ROOT/apps/web/node_modules/.bin/vite" --port "$WEB_PORT" --strictPort
  ) >"$STATE_DIR/web.log" 2>&1 &
  echo $! >"$STATE_DIR/web.pid"
  echo "$WEB_PORT" >"$STATE_DIR/web.port"
  ui::spinner_start "Web dev server starting on :$WEB_PORT"
  wait_ready "http://127.0.0.1:$WEB_PORT/" "$WEB_READY_TIMEOUT" || {
    ui::spinner_stop
    die_with_log web
  }
  ui::spinner_stop
  ui::ok "Web ready on http://127.0.0.1:$WEB_PORT"
}

build_web() {
  ui::run "Building web app (tsc + vite build)" bash -c 'cd apps/web && npm run build' \
    || ui::die "Web build failed" "cd apps/web && npm run build"
}

# ── Modes ────────────────────────────────────────────────────────────────────

service_status_row() { # <name> <pid-lister-fn> <ready-path>
  local name="$1" list_pids="$2" path="$3" pid='' port=''
  [ -f "$STATE_DIR/$name.pid" ] && pid="$(cat "$STATE_DIR/$name.pid")"
  [ -f "$STATE_DIR/$name.port" ] && port="$(cat "$STATE_DIR/$name.port")"
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null \
    && curl -sf -o /dev/null "http://127.0.0.1:$port$path"; then
    ui::summary_row "$name" "running — http://127.0.0.1:$port (pid $pid)" ok
  elif [ -n "$("$list_pids")" ]; then
    ui::summary_row "$name" "running but untracked — make stop to clean up" warn
  else
    ui::summary_row "$name" "stopped" skip
  fi
}

mkdir -p "$STATE_DIR"
# .env is optional local config (preferred ports); exported so starters see it.
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

case "$MODE" in
  stop)
    ui::banner "naviernet" "stopping services"
    stopped=1
    if stop_service api api_pids; then stopped=0; fi
    if stop_service web web_pids; then stopped=0; fi
    [ "$stopped" -eq 0 ] || ui::info "Nothing was running"
    exit 0
    ;;
  status)
    ui::banner "naviernet" "status"
    service_status_row api api_pids /healthz
    service_status_row web web_pids /
    ui::summary "Services"
    exit 0
    ;;
esac

trap on_interrupt INT TERM
ui::banner "naviernet" "starting: $MODE"

case "$MODE" in
  all)
    require_api_env
    require_web_env
    ui::step 1 3 "Cleaning up previous instances"
    stop_service api api_pids || ui::skip "No stale API instance"
    stop_service web web_pids || ui::skip "No stale web instance"
    ui::step 2 3 "Starting API"
    # Reserve the web port first so the API's CORS list can name it.
    WEB_PORT="$(allocate_port "${NAVIERNET_WEB_PORT:-$DEFAULT_WEB_PORT}" web)"
    start_api
    ui::step 3 3 "Starting web dev server"
    start_web
    ui::summary_row "Web UI" "http://127.0.0.1:$WEB_PORT" ok
    ui::summary_row "API" "http://127.0.0.1:$API_PORT (docs: /docs)" ok
    ui::summary_row "Logs" ".run-state/api.log · .run-state/web.log" ok
    ui::summary "Platform is up"
    ;;
  api)
    require_api_env
    stop_service api api_pids || true
    start_api
    ui::summary_row "API" "http://127.0.0.1:$API_PORT (docs: /docs)" ok
    ui::summary_row "Log" ".run-state/api.log" ok
    ui::summary "API is up"
    ;;
  web)
    require_web_env
    stop_service web web_pids || true
    curl -sf -o /dev/null "http://127.0.0.1:$(cat "$STATE_DIR/api.port" 2>/dev/null || echo "$DEFAULT_API_PORT")/healthz" \
      || ui::warn "API is not responding — /api requests will fail (make start)"
    start_web
    ui::summary_row "Web UI" "http://127.0.0.1:$WEB_PORT" ok
    ui::summary_row "Log" ".run-state/web.log" ok
    ui::summary "Web is up"
    ;;
  prod)
    require_api_env
    require_web_env
    ui::step 1 3 "Building web app"
    build_web
    ui::step 2 3 "Cleaning up previous instances"
    stop_service api api_pids || ui::skip "No stale API instance"
    stop_service web web_pids || ui::skip "No stale web instance"
    ui::step 3 3 "Starting the platform (one process)"
    start_api
    ui::summary_row "Platform" "http://127.0.0.1:$API_PORT (UI + API)" ok
    ui::summary_row "Log" ".run-state/api.log" ok
    ui::summary "Platform is up"
    ;;
esac
