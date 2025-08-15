.PHONY: install install-dev install-test test test-unit test-integration coverage coverage-html clean lint format type-check build publish help

# Default target
.DEFAULT_GOAL := help

# Help command
help:
	@echo "Available commands:"
	@echo "  make install        Install package in production mode"
	@echo "  make install-dev    Install package with development dependencies"
	@echo "  make install-test   Install package with test dependencies only"
	@echo "  make test           Run all tests"
	@echo "  make test-unit      Run unit tests only"
	@echo "  make test-integration Run integration tests only"
	@echo "  make coverage       Run tests with coverage report"
	@echo "  make coverage-html  Generate and open HTML coverage report"
	@echo "  make lint           Run all linters (flake8, black check, isort check)"
	@echo "  make format         Format code with black and isort"
	@echo "  make type-check     Run mypy type checking"
	@echo "  make clean          Clean up generated files"
	@echo "  make build          Build distribution packages"
	@echo "  make publish        Publish to PyPI (requires credentials)"
	@echo "  make pre-commit     Run pre-commit hooks on all files"
	@echo "  make pre-commit-staged Run pre-commit hooks on staged files"
	@echo "  make pre-commit-update Update pre-commit hooks to latest versions"
	@echo "  make dev-setup      Set up development environment with pre-commit"

# Install package in production mode
install:
	pip install -e .

# Install package with development dependencies
install-dev:
	pip install -e ".[dev]"

# Install package with test dependencies only
install-test:
	pip install -e ".[test]"

# Run all tests
test:
	pytest

# Run only unit tests
test-unit:
	pytest -m unit

# Run only integration tests
test-integration:
	pytest -m integration

# Run tests with coverage
coverage:
	pytest --cov=hl_scai --cov-report=term-missing --cov-report=html --cov-report=xml

# Open coverage report in browser
coverage-html: coverage
	@echo "Opening coverage report..."
	@python -m webbrowser htmlcov/index.html || open htmlcov/index.html || xdg-open htmlcov/index.html || echo "Please open htmlcov/index.html in your browser"

# Run all linters
lint: lint-flake8 lint-black lint-isort lint-mypy

# Run flake8
lint-flake8:
	flake8 hl_scai tests

# Run black in check mode
lint-black:
	black --check hl_scai tests

# Run isort in check mode
lint-isort:
	isort --check-only hl_scai tests

# Run mypy type checking
lint-mypy:
	mypy hl_scai

# Format code
format:
	black hl_scai tests
	isort hl_scai tests

# Type checking
type-check:
	mypy hl_scai

# Clean up generated files
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf htmlcov/
	rm -f coverage.xml
	rm -f .coverage
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*~" -delete
	find . -type f -name ".DS_Store" -delete

# Build distribution packages
build: clean
	python -m build

# Publish to PyPI
publish: build
	python -m twine upload dist/*

# Run tests in verbose mode
test-verbose:
	pytest -vvs

# Run tests and stop on first failure
test-fast:
	pytest -x

# Run tests with specific pattern
test-pattern:
	@read -p "Enter test pattern: " pattern; \
	pytest -k "$$pattern"

# Watch tests (requires pytest-watch)
watch:
	ptw -- -x

# Run security checks
security:
	pip-audit
	bandit -r hl_scai

# Create development environment
dev-setup: install-dev
	pre-commit install
	@echo "Development environment setup complete!"

# Run pre-commit on all files
pre-commit:
	pre-commit run --all-files

# Run pre-commit on staged files only
pre-commit-staged:
	pre-commit run

# Update pre-commit hooks
pre-commit-update:
	pre-commit autoupdate

# Clean pre-commit cache
pre-commit-clean:
	pre-commit clean
	pre-commit gc
