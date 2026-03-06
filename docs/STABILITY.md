# Stability & Compatibility

This document describes PyImgAno's compatibility expectations for users who want
to depend on the project long-term.

## Versioning

PyImgAno aims to follow Semantic Versioning (SemVer):

- **Patch** releases: bugfixes and internal changes, no intentional breaking API
  changes.
- **Minor** releases: new features and improvements. Breaking changes are
  avoided, but may happen when still below `1.0.0`.
- **Major** releases: may include breaking changes.

## What Is Considered Public API

The following are considered part of the public API and should remain stable:

- The top-level module `pyimgano` exports listed in `pyimgano.__all__`.
- The registry entrypoints used by the CLI tools (`pyimgano-benchmark`,
  `pyimgano-infer`, etc.).
- Documented JSON schemas and run artifacts used by production workflows (for
  example, inference JSONL output and exported `infer_config.json`).

The following are **not** guaranteed stable:

- Private modules, names prefixed with `_`, and undocumented internal helpers.
- Experimental models/aliases that are not referenced in docs or baseline suites.
- Implementation details of optional backends and third-party wrappers.

## Deprecation Policy (Best Effort)

When a public API is changed, PyImgAno aims to:

- Emit a deprecation warning first (when practical).
- Keep deprecated functionality for at least one minor release before removal.

## Supported Python Versions

PyImgAno supports Python versions declared in `pyproject.toml`.

## Optional Dependencies

Optional extras (torch/onnx/openvino/etc.) are supported on a best-effort basis.
When reporting bugs involving extras, include:

- OS, Python version, and PyImgAno version
- The extra(s) you installed (for example `pyimgano[onnx]`)
- The backend runtime versions (for example `onnxruntime` version)
