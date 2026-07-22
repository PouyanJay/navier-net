.DEFAULT_GOAL := help
PY ?= python
VENV := .venv
BIN := $(VENV)/bin

.PHONY: help venv install lint format test test-all preprocess train evaluate figures video all clean clean-runs

help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

venv:  ## Create the virtual environment
	$(PY) -m venv $(VENV)

install: venv  ## Install the package and dev tooling (editable)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e ".[dev]"

lint:  ## Check style and import order
	$(BIN)/ruff check src tests
	$(BIN)/ruff format --check src tests

format:  ## Apply formatting and autofixes
	$(BIN)/ruff check --fix src tests
	$(BIN)/ruff format src tests

test:  ## Run the fast test suite
	$(BIN)/pytest -m "not slow and not needs_data"

test-all:  ## Run every test, including slow and data-dependent ones
	$(BIN)/pytest

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

all:  ## Every stage in order
	$(BIN)/naviernet stage=all

clean:  ## Remove caches and build artefacts
	rm -rf build dist .pytest_cache .ruff_cache **/__pycache__ src/*.egg-info
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +

clean-runs:  ## Delete every generated run output (keeps raw and processed data)
	rm -rf outputs/*
