# YDT Makefile

.PHONY: help install install-dev test test-fast test-integration lint format build clean

# Default target
help:
	@echo "Available commands:"
	@echo "  install      Install package in development mode"
	@echo "  install-dev  Install package with development dependencies"
	@echo "  test         Run all tests"
	@echo "  test-fast    Run only fast tests"
	@echo "  test-integration Run integration tests"
	@echo "  lint         Run linting checks"
	@echo "  format       Format code with black"
	@echo "  build        Build package for distribution"
	@echo "  clean        Clean build artifacts"
	@echo "  ci           Run full CI pipeline"

# Installation
install:
	uv venv
	uv pip install -e .

install-dev:
	uv venv
	uv pip install -e ".[dev]"

# Testing
test:
	uv run pytest tests/ -v

test-fast:
	uv run pytest tests/ -m "not slow" -v

test-integration:
	uv run pytest tests/ -m "integration" -v

test-coverage:
	uv run pytest tests/ --cov=. --cov-report=html --cov-report=xml

# Code quality
lint:
	uv run black --check .
	uv run ruff check .
	uv run mypy .

format:
	uv run black .
	uv run ruff check --fix .

# Building and distribution
build:
	uv run python -m build

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

# CI pipeline
ci: clean install-dev test lint build

# Development helpers
dev-setup:
	uv venv
	uv pip install -e ".[dev]"
	pre-commit install

watch-test:
	uv run pytest-watch tests/ -- -v

docs-build:
	cd docs && uv run mkdocs build

docs-serve:
	cd docs && uv run mkdocs serve