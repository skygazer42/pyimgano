# Stability & Compatibility

This document describes PyImgAno's compatibility expectations for users who want
to depend on the project long-term. The current package and runtime version is
`0.10.0`.

## Versioning

PyImgAno aims to follow Semantic Versioning (SemVer):

- **Patch** releases: bug fixes and internal changes, with no intentional breaking
  public API or schema changes.
- **Minor** releases: new features and improvements. Breaking changes are avoided,
  but can still occur while the project is below `1.0.0` after the deprecation
  notice described below.
- **Major** releases: may include breaking changes.

## What Is Considered Public API

The following are public contracts:

- The top-level exports listed in `pyimgano.__all__`.
- Documented exports from `pyimgano.artifacts`, `pyimgano.exporting`, and
  `pyimgano.inference`, including `load_artifact`, `export_run`, `import_onnx`,
  and `bind_policy`.
- Console scripts declared in `[project.scripts]`, including `pyimgano-export`,
  `pyimgano-artifact`, and `pyimgano-infer`.
- Documented JSON schemas and production artifacts, including inference JSONL,
  `infer_config.json`, `export_index.json`, and `artifact_manifest.json`.
- Published schema/version constants such as `ARTIFACT_SCHEMA_VERSION` and
  `EXPORT_INDEX_SCHEMA_VERSION`.

The following are **not** guaranteed stable:

- Private modules, names prefixed with `_`, and undocumented internal helpers.
- Experimental models/aliases that are not referenced in documentation or
  baseline suites.
- Undocumented implementation details of optional backends and third-party
  wrappers.

## Trained artifact schema v1

PyImgAno 0.10.x writes the `pyimgano-artifact` schema with
`schema_version: 1`. Schema v1 includes the complete runtime component closure,
an artifact-local inference policy, adapter/codec identities, compatibility
metadata, and verification evidence. Patch releases in the 0.10 line will not
silently reinterpret those fields. A future incompatible representation must use
a new schema version and publish a migration path; current loaders reject unknown
future versions.

Raw `.onnx`, `.pt`, or OpenVINO files are not schema-v1 artifacts. They do not, by
themselves, define preprocessing, anomaly-score direction, thresholds, or map
semantics. See [Trained Artifacts](TRAINED_ARTIFACTS.md) for the explicit ONNX
import and fitted-run export paths.

## Trust boundary

Artifact manifests are untrusted input until validation finishes. Loading checks
relative contained paths, symlink/path traversal, declared sizes and SHA-256
digests, schema compatibility, and the complete referenced file closure before a
backend session is created. Verified bytes are staged privately for runtime use.

Executable checkpoint deserialization is disabled by default. The
`trust_checkpoint=True` API option and `--trust-checkpoint` CLI flag are an
explicit trust decision, not an integrity bypass: the artifact still must pass
manifest and digest verification, and callers remain responsible for provenance.

## Release-certified artifact matrix

The independent 0.10.0 artifact release gate covers the following cells. `CPU`
means a GitHub-hosted x86_64 CPU runner; no GPU provider is implied.

| Format | OS | Python | Backend/provider | Runtime profile |
|---|---|---:|---|---|
| Native | Ubuntu (`ubuntu-latest`) | 3.10 | `pyimgano` / CPU | Base plus model-specific dependencies |
| ONNX | Ubuntu (`ubuntu-latest`) | 3.10 | `onnxruntime` / `CPUExecutionProvider` | `onnx-runtime` |
| TorchScript | Ubuntu (`ubuntu-latest`) | 3.10 | `torchscript` / CPU | `torch` |
| OpenVINO | Ubuntu (`ubuntu-latest`) | 3.10 | `openvino` / CPU | `openvino-runtime` |

Other OS, Python, accelerator, and provider cells may work and can be covered by
the project's general compatibility jobs, but they are **not release-certified
artifact cells** and must not be presented as supported artifact combinations.
ONNX and OpenVINO runtime-only jobs explicitly prove that `torch` is absent.

## Deprecation and migration policy

When a public contract changes, PyImgAno aims to publish a notice, provide a
migration path, and retain the old path throughout at least the minor release in
which it is deprecated. A Python warning is emitted when practical. Package
installer aliases cannot emit Python warnings during dependency resolution, so
this stability document and the changelog are the authoritative notice for extra
names.

The 0.10 release has two time-bounded compatibility surfaces:

- `pyimgano-export-onnx` and `pyimgano-export-torchscript` are legacy
  embedding/backbone exporters. They do not export a fitted detector and do not
  create `artifact_manifest.json`. Use `pyimgano-export`. They remain available
  throughout 0.10.x and are eligible for removal no earlier than `0.11.0`.
- `onnx` and `openvino` are compatibility extras. New installations should use
  `onnx-runtime`, `onnx-export`, `openvino-runtime`, or `openvino-export`. The
  aliases remain available throughout 0.10.x and are eligible for removal no
  earlier than `0.11.0`.

For pre-schema-v1 runs and inference configs,
`load_legacy_artifact(..., allow_legacy=True)` is an explicit migration bridge
that emits `LegacyArtifactWarning`. Prefer exporting a schema-v1 artifact from
the persisted run whenever its fitted model has a certified adapter.

## Supported Python versions and optional dependencies

The package metadata declares `requires-python = ">=3.9"`; the release
compatibility jobs currently cover Python 3.9 through 3.12. That general tested
package range is broader than the release-certified artifact matrix above, while
newer Python versions allowed by the metadata are not certified until added to
the compatibility jobs.

When reporting an optional-backend issue, include:

- OS, architecture, Python version, and PyImgAno version.
- The exact extra installed (for example `pyimgano[onnx-runtime]`).
- Backend/runtime versions and selected provider/device.
- The artifact manifest's format, schema version, and verification level without
  disclosing sensitive model data.
