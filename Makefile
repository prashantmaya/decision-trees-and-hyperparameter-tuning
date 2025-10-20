.PHONY: help install install-dev train run test lint format docker-build docker-run clean

# Python interpreter - use parent .venv if it exists, otherwise local venv, otherwise python3
PYTHON := $(shell if [ -d "../.venv" ]; then echo "../.venv/bin/python"; elif [ -d "venv" ]; then echo "venv/bin/python"; elif [ -d ".venv" ]; then echo ".venv/bin/python"; else echo "python3"; fi)

help:
	@echo "Available commands:"
	@echo "  install      - Install production dependencies"
	@echo "  install-dev  - Install development dependencies"
	@echo "  train        - Train the Decision Tree model"
	@echo "  run          - Run the API server"
	@echo "  test         - Run tests"
	@echo "  lint         - Run code linters"
	@echo "  format       - Format code with black and isort"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run   - Run Docker container"
	@echo "  clean        - Remove generated files"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

train:
	PYTHONPATH=. $(PYTHON) scripts/train.py

run:
	PYTHONPATH=. $(PYTHON) -m uvicorn src.decision_tree_predictor.api:app --reload --host 0.0.0.0 --port 8000

test:
	PYTHONPATH=. $(PYTHON) -m pytest

test-verbose:
	PYTHONPATH=. $(PYTHON) -m pytest -v

test-coverage:
	PYTHONPATH=. $(PYTHON) -m pytest --cov=src --cov-report=html --cov-report=term

lint:
	flake8 src scripts --exclude=venv,env,.venv,__pycache__
	mypy src scripts --ignore-missing-imports
	pylint src scripts --disable=C0111,R0903 || true

format:
	black src scripts tests
	isort src scripts tests

docker-build:
	docker-compose build

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down

docker-logs:
	docker-compose logs -f

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/
	rm -f .coverage

