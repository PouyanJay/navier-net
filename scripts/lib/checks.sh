# shellcheck shell=bash
# Precondition checks shared by the scripts/ suite. Requires lib/ui.sh to be
# sourced first, and the caller's working directory to be the repo root.

require_venv_tool() { # <binary under .venv/bin>
  [ -x ".venv/bin/$1" ] || ui::die "$1 is not installed" "make setup"
}

require_api_env() {
  if ! { [ -x .venv/bin/python ] && .venv/bin/python -c "import naviernet_api" 2>/dev/null; }; then
    ui::die "The API is not installed" "make setup"
  fi
}

# The vite binary doubles as the "npm ci has run" marker: it only exists once
# apps/web/node_modules is fully populated.
require_web_env() {
  [ -x apps/web/node_modules/.bin/vite ] || ui::die "Web dependencies are not installed" "make setup"
}
