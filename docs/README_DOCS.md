# PyImgAno Documentation

This directory contains the documentation for PyImgAno, built using [Sphinx](https://www.sphinx-doc.org/).

## Building the Documentation

### Prerequisites

Install documentation dependencies:

```bash
pip install pyimgano[docs]
# Or manually:
pip install sphinx sphinx-rtd-theme sphinxcontrib-napoleon
```

### Build HTML Documentation

```bash
cd docs
make html
```

The generated documentation will be in `build/html/`. Open `build/html/index.html` in your browser.

### Other Formats

```bash
# PDF (requires LaTeX)
make latexpdf

# ePub
make epub

# Plain text
make text

# Check for broken links
make linkcheck
```

### Clean Build

```bash
make clean
make html
```

## Documentation Structure

```
docs/
├── source/
│   ├── conf.py              # Sphinx configuration
│   ├── index.rst            # Main documentation page
│   ├── api/                 # API reference
│   │   ├── detectors.rst    # Detector algorithms
│   │   ├── preprocessing.rst # Preprocessing operations
│   │   └── augmentation.rst # Augmentation techniques
│   ├── _static/             # Static files (CSS, images)
│   └── _templates/          # Custom templates
├── build/                   # Generated documentation (not in git)
├── Makefile                 # Build script (Unix/Mac)
├── make.bat                 # Build script (Windows)
├── QUICKSTART.md            # Quick start guide
├── COMPARISON.md            # PyImgAno vs PyOD comparison
└── CAPABILITY_ASSESSMENT.md # Capability assessment
```

## Recommended reading order (for users)

If you’re updating docs, these pages form the "happy path" for most industrial users:

- `docs/QUICKSTART.md` (install + basic usage)
- `docs/WORKBENCH.md` (train/eval/export loop; artifacts)
- `docs/MANIFEST_DATASET.md` (recommended custom dataset format: JSONL manifest, paths-first)
- `docs/INDUSTRIAL_INFERENCE.md` (numpy-first + tiling + defects output)
- `docs/FALSE_POSITIVE_DEBUGGING.md` (practical FP tuning with overlays/ROI/filters)
- `docs/CLI_REFERENCE.md` (CLI flags + JSONL schemas)

## Writing Documentation

### Docstring Format

PyImgAno uses Google-style docstrings:

```python
def my_function(param1: int, param2: str) -> bool:
    """Brief description of the function.

    More detailed description if needed. Can be multiple paragraphs.

    Args:
        param1: Description of param1.
        param2: Description of param2.

    Returns:
        Description of return value.

    Raises:
        ValueError: When param1 is negative.

    Examples:
        >>> my_function(42, "hello")
        True
    """
    return True
```

### Adding New API Documentation

1. Create or update RST file in `source/api/`
2. Add autodoc directives:

```rst
My New Module
=============

.. automodule:: pyimgano.my_module
   :members:
   :undoc-members:
   :show-inheritance:
```

3. Add to table of contents in `source/index.rst`

### Adding New Pages

1. Create RST file in `source/`
2. Add to `toctree` in `source/index.rst`:

```rst
.. toctree::
   :maxdepth: 2

   my_new_page
```

## Live Rebuild

For development, use live rebuild:

```bash
pip install sphinx-autobuild
make livehtml
```

Open http://127.0.0.1:8000 in your browser. Documentation will rebuild automatically when you save files.

## Documentation Guidelines

### Style Guide

- Use clear, concise language
- Include code examples
- Add cross-references using `:doc:`, `:class:`, `:func:`
- Use admonitions for notes, warnings, tips:

```rst
.. note::
   This is a note.

.. warning::
   This is a warning.

.. tip::
   This is a tip.
```

### Code Examples

Always include working code examples:

```rst
.. code-block:: python

   import numpy as np
   from pyimgano.models import create_model

   class IdentityExtractor:
       def extract(self, X):
           return np.asarray(X)

   detector = create_model(
       "vision_iforest",
       feature_extractor=IdentityExtractor(),
       contamination=0.1,
   )
   detector.fit(X_train)
```

### Cross-References

Link to other parts of documentation:

```rst
See :doc:`quickstart` for getting started.
See :func:`pyimgano.models.create_model` for model creation.
See :class:`pyimgano.models.iforest.VisionIForest` for the Isolation Forest wrapper.
See :func:`pyimgano.preprocessing.edge_detection` for usage.
```

## Publishing Documentation

### GitHub Pages

To publish on GitHub Pages:

```bash
# Build documentation
make html

# Copy to docs root (if using GitHub Pages from docs/)
# Or push build/html to gh-pages branch
```

### Read the Docs

1. Sign up at https://readthedocs.org/
2. Import your GitHub repository
3. RTD will automatically build from `docs/source/conf.py`

Configuration is in `docs/source/conf.py`:

```python
html_theme = 'sphinx_rtd_theme'
```

## Troubleshooting

### Import Errors

If Sphinx can't import pyimgano:

```python
# In conf.py, ensure this is present:
import sys
import os
sys.path.insert(0, os.path.abspath('../..'))
```

### Missing Dependencies

```bash
pip install sphinx sphinx-rtd-theme sphinxcontrib-napoleon
```

### Build Warnings

Fix warnings to ensure documentation quality:

```bash
make html SPHINXOPTS="-W"  # Treat warnings as errors
```

### Autodoc Not Finding Modules

Make sure pyimgano is installed:

```bash
pip install -e .  # Install in development mode
```

## Checking Documentation

### Coverage

Check documentation coverage:

```bash
make coverage
```

### Linkcheck

Check for broken links:

```bash
make linkcheck
```

### Spell Check

```bash
pip install sphinxcontrib-spelling
# Add to conf.py extensions: 'sphinxcontrib.spelling'
make spelling
```

## Documentation Workflow

1. **Write code with docstrings**
   ```python
   def my_function(param):
       """Brief description.

       Args:
           param: Description.

       Returns:
           Description.
       """
       pass
   ```

2. **Add to API documentation**
   ```rst
   .. autofunction:: pyimgano.module.my_function
   ```

3. **Build and review**
   ```bash
   make html
   open build/html/index.html
   ```

4. **Commit and push**
   ```bash
   git add docs/
   git commit -m "docs: Add documentation for my_function"
   git push
   ```

## Resources

- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [reStructuredText Primer](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)
- [Google Style Docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- [Read the Docs](https://docs.readthedocs.io/)

## Contributing

Contributions to documentation are welcome! See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

## License

Documentation is licensed under MIT License, same as PyImgAno.
