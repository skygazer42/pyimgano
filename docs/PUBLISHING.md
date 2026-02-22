# Publishing to PyPI

This project is set up as a standard Python package (PEP 621 via `pyproject.toml`).
To make it installable via:

```bash
pip install pyimgano
```

you need to publish a release to **PyPI**.

## Prerequisites

- A PyPI account (https://pypi.org/)
- A PyPI API token (recommended) or a username/password (not recommended)
- Python 3.9+

## 1) Pick a version

PyPI does **not** allow re-uploading the same version.

Update both:

- `pyproject.toml` → `[project].version`
- `pyimgano/__init__.py` → `__version__`

## 2) Install build tools

If you're working from a clean environment:

```bash
pip install -U build twine
```

Or, from this repo:

```bash
pip install -e ".[dev]"
```

## 3) Build artifacts (sdist + wheel)

From the repo root:

```bash
python -m build
```

This writes artifacts to `dist/`:

- `dist/*.tar.gz` (sdist)
- `dist/*.whl` (wheel)

## 4) Validate metadata

```bash
twine check dist/*
```

## 5) Upload to TestPyPI (recommended)

```bash
twine upload --repository testpypi dist/*
```

Then install from TestPyPI to validate the install experience:

```bash
pip install -i https://test.pypi.org/simple/ pyimgano
```

## 6) Upload to PyPI (official)

```bash
twine upload dist/*
```

After that, users can install from the official index:

```bash
pip install pyimgano
```

## Optional: GitHub Actions "Trusted Publishing"

PyPI supports publishing without long-lived secrets via GitHub Actions (OIDC).
If you want this, set up a "Trusted Publisher" on PyPI that points at this repo,
and add a release workflow that builds and publishes on tags.

