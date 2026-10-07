# Cross-platform: run inside an activated virtual environment, or override PYTHON,
# e.g. `make test PYTHON=.venv/Scripts/python` (Windows) or `PYTHON=.venv/bin/python` (Linux/macOS).
PYTHON ?= python

.PHONY: help install lint format type-check test test-real test-all ci clean funsd-download pilot docker

help:
	@echo "Targets: install lint format type-check test test-real test-all ci clean funsd-download pilot docker"

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	$(PYTHON) -m ruff check src tests

format:
	$(PYTHON) -m black src tests

type-check:
	$(PYTHON) -m mypy src

# Offline suite: no API keys, no network, no cost.
test:
	$(PYTHON) -m pytest tests/ -v

# Tests marked `real` need live API keys / network.
test-real:
	$(PYTHON) -m pytest tests/ -v -m real

test-all:
	$(PYTHON) -m pytest tests/ -v -m ""

ci: lint type-check test

clean:
	$(PYTHON) -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for pat in ('__pycache__', '.pytest_cache', '.ruff_cache', '.mypy_cache') for p in pathlib.Path('.').rglob(pat) if '.venv' not in p.parts]"

# Downloads FUNSD from its official source and builds data/documents_all.json.
funsd-download:
	$(PYTHON) -m agentic_docs.funsd.download

# Small live pilot (20 documents, costs API credits).
pilot:
	$(PYTHON) -m agentic_docs.cli run --limit 20
	$(PYTHON) -m agentic_docs.cli analyze
	$(PYTHON) -m agentic_docs.cli visualize

# Container image running the offline test suite.
docker:
	docker build -t relational-context-failure-modes .
