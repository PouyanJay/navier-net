#!/usr/bin/env bash
# Command reference for `make help` (and bare `make`).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=scripts/lib/ui.sh
source "$ROOT/scripts/lib/ui.sh"

group() { printf '\n  %s%s%s\n' "$UI_DIM" "$1" "$UI_RESET"; }
cmd() { printf '    %s%-18s%s %s\n' "$UI_PRIMARY" "$1" "$UI_RESET" "$2"; }

printf '\n  %s%snaviernet%s %s— PINN solver pipeline + web platform%s\n' \
  "$UI_BOLD" "$UI_PRIMARY" "$UI_RESET" "$UI_DIM" "$UI_RESET"

group "Setup & run"
cmd "make run" "Everything end to end: install deps, then start the stack"
cmd "make setup" "Install all dev dependencies (uv + npm; idempotent)"
cmd "make start" "Start API + web dev server (auto-allocates ports)"
cmd "make start-api" "Start only the API"
cmd "make start-web" "Start only the web dev server"
cmd "make serve" "Production mode: build the UI, serve everything on one port"
cmd "make stop" "Stop all services (including orphans from lost sessions)"
cmd "make status" "Show what is running and where"

group "Solver pipeline"
cmd "make preprocess" "Raw TIFFs -> tensors + QC figure"
cmd "make train" "Train (resumable); STEPS=N to override"
cmd "make evaluate" "IoU report and kinematic checks"
cmd "make figures" "All result figures"
cmd "make video" "Slow-motion MP4"
cmd "make pipeline" "Every stage in order"

group "Testing"
cmd "make test" "Fast Python suite + web unit tests"
cmd "make test-python" "Python tests only"
cmd "make test-web" "Web unit tests only"
cmd "make test-e2e" "Browser end-to-end tests (playwright)"
cmd "make test-all" "Everything, incl. slow, data-dependent, and e2e"

group "Linting"
cmd "make lint" "All CI linters: ruff, eslint, tsc"
cmd "make lint-fix" "Apply autofixes, then re-check"

group "Housekeeping"
cmd "make clean" "Remove caches and build artefacts"
cmd "make clean-runs" "Delete generated run outputs"

printf '\n  %sScripts take flags too: scripts/run.sh --help, scripts/run-tests.sh --help%s\n\n' \
  "$UI_DIM" "$UI_RESET"
