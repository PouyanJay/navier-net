# Thin dispatch layer only — all logic lives in scripts/ (see scripts/*.sh
# --help for flags). Solver pipeline targets call the venv CLI directly.
SHELL := /bin/bash
.DEFAULT_GOAL := help
BIN := .venv/bin

# ── Setup & orchestration ────────────────────────────────────────────
.PHONY: help setup install run start start-api start-web serve stop status

help:  ## Show the command reference
	@./scripts/help.sh

setup:  ## Install all dev dependencies (uv + npm; idempotent)
	@./scripts/install.sh

install: setup  ## Alias for setup (back-compat)

run: setup  ## Everything end to end: install, then start the full stack
	@./scripts/run.sh

start:  ## Start API + web dev server (auto-allocates ports)
	@./scripts/run.sh

start-api:  ## Start only the API
	@./scripts/run.sh --api-only

start-web:  ## Start only the web dev server
	@./scripts/run.sh --web-only

serve:  ## Production mode: build the UI, serve everything on one port
	@./scripts/run.sh --prod

stop:  ## Stop all services, including orphans
	@./scripts/run.sh --stop

status:  ## Show what is running and where
	@./scripts/run.sh --status

# ── Testing ──────────────────────────────────────────────────────────
.PHONY: test test-python test-web test-e2e test-all

test:  ## Fast Python suite + web unit tests
	@./scripts/run-tests.sh --all

test-python:  ## Python tests only
	@./scripts/run-tests.sh --python

test-web:  ## Web unit tests only
	@./scripts/run-tests.sh --web

test-e2e:  ## Browser end-to-end tests
	@./scripts/run-tests.sh --e2e

test-all:  ## Everything: slow + data-dependent Python tests, web, e2e
	@./scripts/run-tests.sh --all --full --e2e

# ── Linting ──────────────────────────────────────────────────────────
.PHONY: lint lint-fix format

lint:  ## All CI linters: ruff, eslint, tsc
	@./scripts/run-linters.sh --all

lint-fix:  ## Apply autofixes, then re-check
	@./scripts/run-linters.sh --all --fix

format: lint-fix  ## Alias for lint-fix (back-compat)

# ── Solver pipeline ──────────────────────────────────────────────────
.PHONY: preprocess train evaluate figures video pipeline all

preprocess:  ## Raw TIFFs -> tensors + QC figure
	$(BIN)/naviernet stage=preprocess

train:  ## Train (resumes from the run's checkpoint); STEPS=N to override
	$(BIN)/naviernet stage=train $(if $(STEPS),training.steps=$(STEPS),)

evaluate:  ## IoU report and kinematic checks
	$(BIN)/naviernet stage=evaluate

figures:  ## All result figures
	$(BIN)/naviernet stage=figures

video:  ## Slow-motion MP4
	$(BIN)/naviernet stage=video

pipeline:  ## Every stage in order
	$(BIN)/naviernet stage=all

all: pipeline  ## Alias for pipeline (back-compat)

# ── Housekeeping ─────────────────────────────────────────────────────
.PHONY: clean clean-runs

clean:  ## Remove caches and build artefacts
	rm -rf build dist .pytest_cache .ruff_cache src/*.egg-info
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +

clean-runs:  ## Delete every generated run output (keeps raw and processed data)
	rm -rf outputs/*
