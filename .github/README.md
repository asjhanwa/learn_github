# CI/CD Pipeline Configuration

This repository includes a comprehensive CI/CD pipeline that ensures code quality, testing, and automated publishing.

## Workflow Overview

The CI pipeline includes the following components:

### 1. **Test and Lint** (runs on Python 3.8, 3.9, 3.10, 3.11)
- **Linting**: Code quality checks using flake8
- **Formatting**: Code formatting checks using black
- **Type Checking**: Static type analysis using mypy
- **Testing**: Unit tests with pytest
- **Coverage**: Code coverage reporting (target: 95%)
- **Security**: Security scan using bandit

### 2. **Build and Publish** (runs on main branch push)
- **Package Build**: Creates source and wheel distributions
- **Artifact Upload**: Stores build artifacts
- **Publishing**: Publishes to Test PyPI (when configured)

### 3. **Coverage Reporting** (runs on pull requests)
- **HTML Coverage Report**: Detailed coverage analysis
- **Coverage Artifacts**: Downloadable coverage reports

## Configuration Files

- **`.github/workflows/ci.yml`**: Main CI/CD workflow
- **`.flake8`**: Linting configuration
- **`pyproject.toml`**: Tool configuration (black, mypy, pytest, coverage)

## Running Locally

To run the same checks locally:

```bash
# Install development dependencies
pip install pytest pytest-cov black flake8 mypy bandit build

# Run linting
flake8 llm_evaluator/

# Run formatting check
black --check llm_evaluator/

# Run type checking
mypy llm_evaluator/ --ignore-missing-imports

# Run tests with coverage
pytest tests/ --cov=llm_evaluator --cov-report=term-missing

# Run security scan
bandit -r llm_evaluator/

# Build package
python -m build
```

## Coverage Requirements

The pipeline enforces a minimum code coverage of 95%. Currently, the codebase has ~46% coverage, so this will need improvement to meet the target.

## Security Scanning

The pipeline includes security scanning with bandit to identify potential security issues in the codebase.

## Publishing

The workflow is configured to publish to Test PyPI on main branch pushes. To enable this:

1. Create a Test PyPI account
2. Generate an API token
3. Add the token as `TEST_PYPI_API_TOKEN` secret in GitHub repository settings

## Artifacts

The pipeline generates the following artifacts:
- **Python package distributions** (source and wheel)
- **Security scan results** (bandit report)
- **Coverage HTML report** (for pull requests)

## Status Badges

Add these badges to your README to show build status:

```markdown
![CI Pipeline](https://github.com/asjhanwa/learn_github/workflows/CI%20Pipeline/badge.svg)
![Coverage](https://img.shields.io/codecov/c/github/asjhanwa/learn_github)
```