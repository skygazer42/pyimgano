Stability & Compatibility
=========================

This page documents PyImgAno's stability and compatibility expectations for
users who want to depend on the project long-term.

Versioning
----------

PyImgAno aims to follow Semantic Versioning (SemVer):

* Patch releases: bugfixes and internal changes, no intentional breaking API changes.
* Minor releases: new features and improvements. Breaking changes are avoided, but may happen when still below 1.0.0.
* Major releases: may include breaking changes.

Public API
----------

The following are considered public API and should remain stable:

* The top-level ``pyimgano`` exports listed in ``pyimgano.__all__``.
* The registry entrypoints used by the CLI tools.
* Documented JSON schemas and run artifacts used by production workflows.

The following are not guaranteed stable:

* Private modules, names prefixed with ``_``, and undocumented internal helpers.
* Experimental models/aliases that are not referenced in docs or baseline suites.
* Implementation details of optional backends and third-party wrappers.

Deprecation (Best Effort)
-------------------------

When a public API is changed, PyImgAno aims to emit a deprecation warning first
and keep deprecated functionality for at least one minor release before removal.

Supported Python Versions
-------------------------

PyImgAno supports Python versions declared in ``pyproject.toml``.
