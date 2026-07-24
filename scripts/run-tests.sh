#!/usr/bin/env bash
# Run any or all test suites. Every selected suite runs even when an earlier
# one fails — the point is to see ALL failures, not the first — and the exit
# code is the aggregate, so CI still fails correctly.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=scripts/lib/ui.sh
source "$ROOT/scripts/lib/ui.sh"
cd "$ROOT"

usage() {
  cat <<'EOF'
Usage: scripts/run-tests.sh [suites] [modifiers] [-h|--help]

Suites (combinable; default --all):
  --all      Python fast suite + web unit tests
  --python   Python tests (pytest)
  --web      Web unit tests (vitest)
  --e2e      Browser end-to-end tests (playwright; boots its own servers)

Modifiers:
  --full     Python suite runs everything, including slow and needs_data tests
EOF
}

RUN_PYTHON=0
RUN_WEB=0
RUN_E2E=0
FULL=0
for arg in "$@"; do
  case "$arg" in
    --all)
      RUN_PYTHON=1
      RUN_WEB=1
      ;;
    --python) RUN_PYTHON=1 ;;
    --web) RUN_WEB=1 ;;
    --e2e) RUN_E2E=1 ;;
    --full) FULL=1 ;;
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
if [ "$((RUN_PYTHON + RUN_WEB + RUN_E2E))" -eq 0 ]; then
  RUN_PYTHON=1
  RUN_WEB=1
fi

TOTAL=$((RUN_PYTHON + RUN_WEB + RUN_E2E))
STEP=0
AGGREGATE=0

# Streams the suite's own output (a live test run beats a spinner), records
# the verdict, and never aborts the remaining suites.
run_suite() { # <label> <command> [args...]
  local label="$1" rc=0
  shift
  STEP=$((STEP + 1))
  ui::step "$STEP" "$TOTAL" "$label"
  "$@" || rc=$?
  if [ "$rc" -eq 0 ]; then
    ui::ok "$label passed"
    ui::summary_row "$label" "passed" ok
  else
    ui::fail "$label failed (exit $rc)"
    ui::summary_row "$label" "failed (exit $rc)" fail
    AGGREGATE=1
  fi
}

ui::banner "naviernet tests" "$TOTAL suite(s)"

if [ "$RUN_PYTHON" -eq 1 ]; then
  [ -x .venv/bin/pytest ] || ui::die "pytest is not installed" "make setup"
  if [ "$FULL" -eq 1 ]; then
    run_suite "Python (full, incl. slow + needs_data)" .venv/bin/pytest
  else
    run_suite "Python (fast)" .venv/bin/pytest -m "not slow and not needs_data"
  fi
fi

if [ "$RUN_WEB" -eq 1 ]; then
  [ -d apps/web/node_modules ] || ui::die "Web dependencies are not installed" "make setup"
  run_suite "Web unit (vitest)" bash -c 'cd apps/web && npm test'
fi

if [ "$RUN_E2E" -eq 1 ]; then
  [ -d apps/web/node_modules ] || ui::die "Web dependencies are not installed" "make setup"
  # Idempotent: a cached chromium makes this a fast no-op.
  ui::run "Ensuring playwright chromium" bash -c 'cd apps/web && npx playwright install chromium' \
    || ui::warn "Chromium install failed — e2e will likely fail"
  run_suite "End-to-end (playwright)" bash -c 'cd apps/web && npm run test:e2e'
fi

ui::summary "Test results"
exit "$AGGREGATE"
