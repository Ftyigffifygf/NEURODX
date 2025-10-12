# Makefile for NeuroDx-MultiModal development

.PHONY: help install setup test lint format clean docker-build docker-run

# Default target
help:
	@echo "Available targets:"
	@echo "  install          Install Python dependencies"
	@echo "  setup            Complete project setup including genomics workflow"
	@echo "  test             Run tests"
	@echo "  lint             Run code linting"
	@echo "  format           Format code with black"
	@echo "  clean            Clean up temporary files"
	@echo "  docker-build     Build Docker image"
	@echo "  docker-run       Run Docker container"
	@echo "  validate-setup   Validate system configuration"

# Install dependencies
install:
	pip install -r requirements.txt
	pip install -e ".[dev]"

# Complete project setup
setup: install
	@echo "Setting up project directories..."
	mkdir -p data/images data/wearable data/genomics
	mkdir -p models/checkpoints models/cache
	mkdir -p logs results
	@echo "Setting up genomics workflow..."
	python scripts/setup_genomics_workflow.py
	@echo "Copying environment configuration..."
	@if [ ! -f .env ]; then cp .env.example .env; echo "Created .env file - please configure your API keys"; fi
	@echo "Setup completed! Please configure your .env file with API keys."

# Run tests
test:
	pytest tests/ -v --cov=src --cov-report=term-missing

# Run linting
lint:
	flake8 src/ tests/ scripts/
	mypy src/

# Format code
format:
	black src/ tests/ scripts/
	isort src/ tests/ scripts/

# Clean up
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/

# Docker targets
docker-build:
	docker build -t neurodx-multimodal:latest .

docker-run:
	docker run -p 5000:5000 --env-file .env neurodx-multimodal:latest

# Validate system setup
validate-setup:
	python -c "from src.config.settings import validate_nvidia_setup, validate_monai_setup; print('NVIDIA setup:', validate_nvidia_setup()); print('MONAI setup:', validate_monai_setup())"

# Development server
dev:
	python main.py

# Install pre-commit hooks
pre-commit:
	pre-commit install
	pre-commit run --all-files