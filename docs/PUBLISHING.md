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

## Recommended: GitHub Actions release workflow (this repo)

This repository includes a publish workflow:

- `.github/workflows/publish.yml`

It publishes to **PyPI** when a **GitHub Release** is published, and can publish
to **TestPyPI** via manual dispatch.

### One-time setup (GitHub Secrets)

Add repository secrets (GitHub → Settings → Secrets and variables → Actions):

- `PYPI_API_TOKEN`
- (optional) `TEST_PYPI_API_TOKEN`

**First-time publish note:** If `pyimgano` has never been uploaded to PyPI
before, the token must be **account-scoped** (not project-scoped), because
PyPI does not allow a project-scoped token to create a brand-new project.

**After the first publish:** Once the `pyimgano` project exists on PyPI, prefer
rotating to a **project-scoped** token (scope: project `pyimgano`) and store it
in the same GitHub secret (`PYPI_API_TOKEN`). Project-scoped tokens reduce blast
radius if a token is ever leaked.

### Publish to PyPI (official)

1) Tag and push a release (for example `v0.6.8`).
2) Create a GitHub Release for that tag and click **Publish release**.

That triggers the workflow and uploads to PyPI automatically.

### Publish to TestPyPI (optional)

GitHub → Actions → "Publish to PyPI" → Run workflow → `test_pypi=true`.

## Optional: GitHub Actions "Trusted Publishing"

PyPI supports publishing without long-lived secrets via GitHub Actions (OIDC).
If you want this, set up a "Trusted Publisher" on PyPI that points at this repo,
and add a release workflow that builds and publishes on tags.
