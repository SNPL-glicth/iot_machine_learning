# Contributing to ZENIN ML Engine

Thank you for your interest in contributing to the ZENIN ML Engine! This document provides guidelines and instructions for contributors.

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## How to Contribute

### Reporting Bugs

Before creating bug reports, please check the existing issues to avoid duplicates.

When creating a bug report, include:
- A clear and descriptive title
- Steps to reproduce the bug
- Expected behavior vs. actual behavior
- Environment information (Python version, OS, etc.)
- Relevant logs or error messages
- Screenshots if applicable

### Suggesting Enhancements

Enhancement suggestions are welcome! Please:
- Use a clear and descriptive title
- Provide a detailed description of the enhancement
- Explain why this enhancement would be useful
- List any alternative solutions or features you've considered

### Pull Requests

1. **Fork the repository** and create your branch from `master`
2. **Make your changes** following our coding standards
3. **Write tests** for your changes
4. **Ensure all tests pass**
5. **Update documentation** if needed
6. **Submit a pull request** with a clear description

#### Pull Request Guidelines

- Keep PRs focused and atomic (one feature or fix per PR)
- Write descriptive commit messages
- Include tests for new functionality
- Update relevant documentation
- Ensure CI checks pass
- Request review from maintainers

## Development Setup

### Prerequisites

- Python 3.10 or higher
- Poetry (for dependency management)
- Docker (for containerized development)

### Setup

```bash
# Clone the repository
git clone https://github.com/SNPL-glicth/iot_machine_learning.git
cd iot_machine_learning

# Install dependencies with Poetry
poetry install

# Or with pip
pip install -r requirements.txt
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/infrastructure/test_cognitive_orchestrator.py

# Run with coverage
pytest --cov=domain --cov=application --cov=infrastructure
```

### Code Style

We use:
- **Ruff** for linting and formatting
- **MyPy** for type checking

```bash
# Run linter
ruff check .

# Format code
ruff format .

# Run type checker
mypy infrastructure/ domain/ application/
```

## Coding Standards

### Python

- Follow PEP 8 style guide
- Use type hints for function signatures
- Write docstrings for all public functions and classes
- Keep functions focused and small
- Use descriptive variable names

### Architecture

- Follow Clean Architecture principles
- Respect domain boundaries
- Use dependency injection
- Keep infrastructure separate from domain logic
- Write unit tests for business logic

### Documentation

- Update README.md for user-facing changes
- Add inline comments for complex logic
- Update docstrings for API changes
- Maintain architecture documentation

## Project Structure

```
iot_machine_learning/
├── domain/           # Business logic and entities
├── application/      # Use cases and application services
├── infrastructure/   # External concerns (DB, API, etc.)
├── ml_service/       # ML service layer
├── core/             # Shared utilities and constants
├── tests/            # Test suite
└── docs/             # Documentation
```

## Testing

### Test Coverage

We aim for high test coverage. New features should include:
- Unit tests for business logic
- Integration tests for external dependencies
- E2E tests for critical paths

### Test Naming

- Use descriptive test names: `test_calculate_weight_with_zero_mae_returns_uniform`
- Group related tests in test classes
- Use fixtures for common test setup

## Release Process

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Create a git tag
4. Push to GitHub
5. Create a GitHub release

## Getting Help

- Open an issue for bugs or questions
- Check existing documentation in `docs/`
- Review existing issues for similar problems

## License

This project is proprietary software with All Rights Reserved. The Zenin Team remains the sole owner of this codebase.

**Important:**
- Contributions are accepted only by explicit permission from the Zenin Team
- By contributing, you agree that your contributions become exclusive property of Zenin Team
- Contributors will be recognized for their contributions but will NOT gain ownership or rights to the code
- Contributors will NOT become co-owners or partners in the project
- Contributors have NO claim to the intellectual property
- Contributors have NO rights to use their contributed code outside this project

**Cloning this repository:**
Cloning does NOT grant you any rights to use, modify, or distribute the code. The code remains proprietary property of Zenin Team regardless of cloning.

For permission to use this software, contact: zeninenterprise8@gmail.com

## Recognition

Contributors will be recognized in the project's CONTRIBUTORS.md file.

Thank you for contributing to ZENIN ML Engine!
