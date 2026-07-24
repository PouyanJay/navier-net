#!/usr/bin/env bash
# Run all code-quality checks (or a scoped subset), aggregating failures the
# same way run-tests.sh does. --fix applies autofixes first, then re-checks,
# so a clean exit always means the tree actually passes.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=scripts/lib/ui.sh
source "$ROOT/scripts/lib/ui.sh"
# shellcheck source=scripts/lib/checks.sh
source "$ROOT/scripts/lib/checks.sh"
cd "$ROOT"

usage() {
  cat <<'EOF'
Usage: scripts/run-linters.sh [scopes] [--fix] [-h|--help]

Scopes (combinable; default --all):
  --all      Python (ruff) + web (eslint, tsc)
  --python   ruff check + ruff format
  --web      eslint + typescript typecheck

Modifiers:
  --fix      Apply autofixes first (ruff/eslint), then re-check

The linter set deliberately mirrors CI (.github/workflows/ci.yml), so a
clean local run means a clean CI lint job.
EOF
}

RUN_PYTHON=0
RUN_WEB=0
FIX=0
for arg in "$@"; do
  case "$arg" in
    --all)
      RUN_PYTHON=1
      RUN_WEB=1
      ;;
    --python) RUN_PYTHON=1 ;;
    --web) RUN_WEB=1 ;;
    --fix) FIX=1 ;;
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
if [ "$((RUN_PYTHON + RUN_WEB))" -eq 0 ]; then
  RUN_PYTHON=1
  RUN_WEB=1
fi

AGGREGATE=0

check() { # <label> <command> [args...] — capture output, show it on failure
  local label="$1" rc=0
  shift
  ui::run "$label" "$@" || rc=$?
  if [ "$rc" -eq 0 ]; then
    ui::summary_row "$label" "clean" ok
  else
    ui::summary_row "$label" "failed" fail
    AGGREGATE=1
  fi
}

fix() { # <label> <command> [args...] — stream output so applied fixes are visible
  local label="$1" rc=0
  shift
  "$@" || rc=$?
  if [ "$rc" -eq 0 ]; then
    ui::ok "$label"
  else
    ui::warn "$label (exit $rc)"
  fi
}

# Paths mirror CI (.github/workflows/ci.yml) so local lint == CI lint.
PY_PATHS=(src tests apps/api)

TOTAL=$((RUN_PYTHON + RUN_WEB))
STEP=0
ui::banner "naviernet lint" "$([ "$FIX" -eq 1 ] && echo 'fix + check' || echo check)"

if [ "$RUN_PYTHON" -eq 1 ]; then
  STEP=$((STEP + 1))
  ui::step "$STEP" "$TOTAL" "Python"
  require_venv_tool ruff
  if [ "$FIX" -eq 1 ]; then
    fix "ruff check --fix" .venv/bin/ruff check --fix "${PY_PATHS[@]}"
    fix "ruff format" .venv/bin/ruff format "${PY_PATHS[@]}"
  fi
  check "ruff check" .venv/bin/ruff check "${PY_PATHS[@]}"
  check "ruff format --check" .venv/bin/ruff format --check "${PY_PATHS[@]}"
fi

if [ "$RUN_WEB" -eq 1 ]; then
  STEP=$((STEP + 1))
  ui::step "$STEP" "$TOTAL" "Web"
  require_web_env
  if [ "$FIX" -eq 1 ]; then
    fix "eslint --fix" bash -c 'cd apps/web && npx eslint . --fix'
  fi
  check "eslint" bash -c 'cd apps/web && npm run lint'
  check "typecheck (tsc)" bash -c 'cd apps/web && npm run typecheck'
fi

if [ "$AGGREGATE" -ne 0 ] && [ "$FIX" -eq 0 ]; then
  ui::info "Autofixable? Try: make lint-fix"
fi
ui::summary "Lint results"
exit "$AGGREGATE"
