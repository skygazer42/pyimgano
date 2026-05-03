# Contributing to PyImgAno

First off, thank you for considering contributing to PyImgAno! It's people like you that make PyImgAno such a great tool for the computer vision community.

## Code of Conduct

This project and everyone participating in it is governed by our commitment to
fostering an open and welcoming environment.

Please read and follow `CODE_OF_CONDUCT.md`.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the existing issues to avoid duplicates. When you create a bug report, include as many details as possible:

- **Use a clear and descriptive title**
- **Describe the exact steps to reproduce the problem**
- **Provide specific examples** (code snippets, sample data, etc.)
- **Describe the behavior you observed** and what you expected
- **Include Python version, OS, and PyImgAno version**

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion:

- **Use a clear and descriptive title**
- **Provide a detailed description** of the suggested enhancement
- **Explain why this enhancement would be useful** to most users
- **List any similar features** in other tools if applicable

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Set up your development environment**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/pyimgano.git
   cd pyimgano
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e .[dev]
   ```

3. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

4. **Make your changes**:
   - Write clear, documented code
   - Follow the existing code style
   - Add tests for new functionality
   - Update documentation as needed

5. **Run tests and checks**:
   ```bash
   # Run the focused tests for your change first
   pytest tests/test_<your_area>.py -v

   # Then run the default suite (no coverage by default)
   pytest

   # Run coverage explicitly when you need the report / gate
   tox -e coverage

   # Check code formatting
   black --check pyimgano tests
   isort --check-only pyimgano tests

   # Run linters
   flake8 --config .flake8 pyimgano tests
   ruff check pyimgano tests tools

   # Type checking and repository audits
   mypy pyimgano
   python tools/audit_public_api.py
   python tools/audit_registry.py
   python tools/audit_repo_links.py
   ```

6. **Commit your changes**:
   - Use clear and meaningful commit messages
   - Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:
     - `feat:` for new features
     - `fix:` for bug fixes
     - `docs:` for documentation changes
     - `test:` for test additions/changes
     - `refactor:` for code refactoring
     - `style:` for formatting changes
     - `chore:` for maintenance tasks

7. **Push to your fork** and submit a pull request

## Development Setup

### Prerequisites

- Python >= 3.9
- pip or conda
- Git

### Installation

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/pyimgano.git
cd pyimgano

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e .[dev,diffusion,docs]

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests (default path; no coverage by default)
pytest

# Run with coverage explicitly
tox -e coverage

# Run specific test file
pytest tests/test_augmentation_registry.py

# Run tests matching a pattern
pytest -k "test_augmentation"

# Run repository audit tooling
python tools/audit_public_api.py
python tools/audit_registry.py
python tools/audit_repo_links.py
```

### Code Style

We use the following tools to maintain code quality:

- **Black**: Code formatting (line length: 100)
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

Configuration is in `pyproject.toml` and `.flake8`.

### Adding New Models

To add a new anomaly detection model:

1. Create a new file in `pyimgano/models/`
2. Inherit from `BaseVisionDetector` or `BaseVisionDeepDetector`
3. Implement required methods: `fit()`, `predict()`, `decision_function()`
4. Register your model:
   ```python
   from pyimgano.models.registry import MODEL_REGISTRY

   @MODEL_REGISTRY.register("your_model_name", tags=["ml", "supervised"])
   class YourModel(BaseVisionDetector):
       pass
   ```
5. Add tests in `tests/`
6. Update documentation
7. If benchmark behavior changes, update the relevant reproducibility preset or benchmark docs

### Documentation

Documentation is written in Markdown and reStructuredText (for Sphinx). When adding new features:

1. Add docstrings to all public functions and classes
2. Follow the [NumPy docstring style](https://numpydoc.readthedocs.io/en/latest/format.html)
3. Update relevant documentation files
4. Add usage examples where appropriate

Example docstring:
```python
def your_function(param1: int, param2: str) -> bool:
    """
    Short description of the function.

    Longer description if needed, explaining behavior,
    algorithms, or important details.

    Parameters
    ----------
    param1 : int
        Description of param1
    param2 : str
        Description of param2

    Returns
    -------
    bool
        Description of return value

    Examples
    --------
    >>> your_function(42, "test")
    True
    """
    pass
```

## Project Structure

```
pyimgano/
├── pyimgano/           # Main package
│   ├── models/         # Anomaly detection models
│   ├── datasets/       # Data loading utilities
│   ├── utils/          # Helper functions
│   └── visualization/  # Visualization tools
├── tests/              # Test suite
├── docs/               # Documentation
├── examples/           # Usage examples
└── .github/            # GitHub workflows
```

## Release Process

Releases are handled by maintainers:

1. Update version in `pyproject.toml`
2. Update `pyimgano/__init__.py`
3. Update `CHANGELOG.md` or release notes
4. Run `python -m build`, `twine check dist/*`, and `python tools/audit_repo_links.py`
5. Create and push the release tag
6. Create a new GitHub release
7. CI/CD automatically publishes to PyPI

## Questions?

Feel free to:
- Open an issue for discussion
- Reach out to maintainers
- Check existing documentation and examples

## Recognition

Contributors will be recognized in:
- GitHub contributors page
- Release notes
- Project documentation

Thank you for contributing to PyImgAno! 🎉
