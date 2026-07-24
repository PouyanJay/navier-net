#!/usr/bin/env bash
# One-command dev environment setup, uv-based and idempotent: rerunning after
# a successful install is a no-op (hash stamps under .run-state/ decide when
# dependencies actually need reinstalling).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=scripts/lib/ui.sh
source "$ROOT/scripts/lib/ui.sh"
cd "$ROOT"

STATE_DIR="$ROOT/.run-state"
PY_STAMP="$STATE_DIR/install-python.stamp"
WEB_STAMP="$STATE_DIR/install-web.stamp"
TOTAL=4

usage() {
  cat <<'EOF'
Usage: scripts/install.sh [--force] [-h|--help]

Sets up the full dev environment: uv + node prerequisites, the .venv
(Python pinned by .python-version), both editable Python packages with dev
extras, and the web app's npm dependencies.

  --force     Reinstall dependencies even if nothing changed
  -h, --help  Show this help
EOF
}

FORCE=0
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      ui::die "Unknown flag: $arg"
      ;;
  esac
done

# Hash of the files that define a dependency set; when it matches the stored
# stamp, the corresponding install step is skipped.
fingerprint() { # <file...>
  cat "$@" 2>/dev/null | shasum | cut -d' ' -f1
}

stamp_fresh() { # <stamp-file> <fingerprint>
  [ "$FORCE" -eq 0 ] && [ -f "$1" ] && [ "$(cat "$1")" = "$2" ]
}

ui::banner "naviernet setup" "install"
mkdir -p "$STATE_DIR"

# ── 1/4 Prerequisites ────────────────────────────────────────────────────────
ui::step 1 "$TOTAL" "Prerequisites"

if command -v uv >/dev/null 2>&1; then
  ui::skip "uv $(uv --version | cut -d' ' -f2) already installed"
elif command -v brew >/dev/null 2>&1; then
  ui::run "Installing uv (brew)" brew install uv \
    || ui::die "uv installation failed" "brew install uv"
else
  ui::die "uv is not installed" "curl -LsSf https://astral.sh/uv/install.sh | sh"
fi

if command -v npm >/dev/null 2>&1; then
  ui::skip "node $(node --version) / npm $(npm --version) already installed"
elif command -v brew >/dev/null 2>&1; then
  ui::run "Installing node (brew)" brew install node \
    || ui::die "node installation failed" "brew install node"
else
  ui::die "node/npm is not installed" "brew install node"
fi

# ── 2/4 Python virtual environment ──────────────────────────────────────────
ui::step 2 "$TOTAL" "Python environment"

PY_WANT="$(cat .python-version)"
if [ -x .venv/bin/python ] \
  && .venv/bin/python -c "import sys; sys.exit(0 if '${PY_WANT}' == '.'.join(map(str, sys.version_info[:2])) else 1)" 2>/dev/null; then
  ui::skip ".venv already on Python $PY_WANT"
else
  # uv resolves the interpreter from .python-version, downloading a managed
  # build if no matching local Python exists. --clear replaces a stale venv.
  ui::run "Creating .venv (Python $PY_WANT via uv)" uv venv --clear .venv \
    || ui::die "Could not create the virtual environment" "uv venv --clear .venv"
  rm -f "$PY_STAMP" # new interpreter: force package reinstall below
fi

# ── 3/4 Python packages ─────────────────────────────────────────────────────
ui::step 3 "$TOTAL" "Python packages"

PY_FP="$(fingerprint pyproject.toml apps/api/pyproject.toml .python-version)"
if stamp_fresh "$PY_STAMP" "$PY_FP"; then
  ui::skip "naviernet + naviernet-api unchanged since last install"
else
  ui::run "Installing naviernet + naviernet-api (editable, dev extras)" \
    uv pip install --python .venv/bin/python -e ".[dev]" -e "./apps/api[dev]" \
    || ui::die "Python package installation failed"
  printf '%s\n' "$PY_FP" >"$PY_STAMP"
fi

if .venv/bin/python -c "import naviernet, naviernet_api" 2>/dev/null; then
  ui::ok "Import check: naviernet, naviernet_api"
else
  ui::die "Installed packages fail to import" "scripts/install.sh --force"
fi

# ── 4/4 Web packages ────────────────────────────────────────────────────────
ui::step 4 "$TOTAL" "Web packages"

WEB_FP="$(fingerprint apps/web/package-lock.json)"
if [ -d apps/web/node_modules ] && stamp_fresh "$WEB_STAMP" "$WEB_FP"; then
  ui::skip "apps/web dependencies unchanged since last install"
else
  ui::run "npm ci (apps/web)" bash -c 'cd apps/web && npm ci' \
    || ui::die "npm ci failed"
  printf '%s\n' "$WEB_FP" >"$WEB_STAMP"
fi

ui::summary_row "uv" "$(uv --version | cut -d' ' -f2)" ok
ui::summary_row "Python (.venv)" "$(.venv/bin/python --version | cut -d' ' -f2)" ok
ui::summary_row "Python packages" "naviernet + naviernet-api (editable)" ok
ui::summary_row "node / npm" "$(node --version) / $(npm --version)" ok
ui::summary_row "Web packages" "apps/web/node_modules" ok
ui::summary "Setup complete"
