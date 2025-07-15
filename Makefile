.PHONY: install test lint format type-check security-scan build clean coverage help

# Default target
help:
	@echo "Available targets:"
	@echo "  install       - Install development dependencies"
	@echo "  test          - Run tests"
	@echo "  coverage      - Run tests with coverage report"
	@echo "  lint          - Run linting checks"
	@echo "  format        - Format code with black"
	@echo "  format-check  - Check code formatting"
	@echo "  type-check    - Run type checking"
	@echo "  security-scan - Run security scan"
	@echo "  build         - Build package"
	@echo "  clean         - Clean build artifacts"
	@echo "  ci            - Run all CI checks"

# Install development dependencies
install:
	pip install -r requirements.txt
	pip install pytest pytest-cov black flake8 mypy bandit build
	pip install -e .

# Run tests
test:
	pytest tests/ -v

# Run tests with coverage
coverage:
	pytest tests/ --cov=llm_evaluator --cov-report=term-missing --cov-report=html

# Run linting
lint:
	flake8 llm_evaluator/ --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 llm_evaluator/ --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

# Format code
format:
	black llm_evaluator/

# Check code formatting
format-check:
	black --check --diff llm_evaluator/

# Run type checking
type-check:
	mypy llm_evaluator/ --ignore-missing-imports

# Run security scan
security-scan:
	bandit -r llm_evaluator/ -f json -o bandit-report.json

# Build package
    - name: Build package
      run: |
        # Try using the modern build tool first, fallback to setuptools
        python -m build --sdist --wheel --outdir dist/ || python setup.py sdist bdist_wheel

# Clean build artifacts
clean:
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -f bandit-report.json

# Run all CI checks
ci: lint format-check type-check test security-scan
	@echo "All CI checks completed"