"""Legacy setuptools entrypoint.

The authoritative package metadata lives in `pyproject.toml` (PEP 621). This
file is kept intentionally minimal so that:

- `pip install .` uses the same metadata as `pip build`.
- We avoid version / dependency drift between `setup.py` and `pyproject.toml`.
"""

from setuptools import setup


if __name__ == "__main__":
    setup()
