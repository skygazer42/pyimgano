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

## Authentication (Twine)

For token-based auth (recommended), set:

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD="pypi-<your-api-token>"
```

Then:

```bash
python -m twine upload dist/*
```

If you see:

- `HTTPError: 403 Forbidden ... Invalid or non-existent authentication information`

it almost always means the token is missing/incorrect, or `TWINE_USERNAME` was not set to `__token__`.

## 1) Pick a version

PyPI does **not** allow re-uploading the same version.

Update both:

- `pyproject.toml` → `[project].version`
- `pyimgano/__init__.py` → `__version__`

## Release checklist

Before tagging a release, verify:

```bash
python -m build
python -m twine check dist/*
python3 tools/audit_release_version.py --tag vX.Y.Z
python3 tools/audit_repo_links.py
python3 tools/audit_public_api.py
python3 tools/audit_registry.py
python3 tools/audit_release_surface.py
python3 tools/audit_adoption_docs.py
python3 tools/audit_audited_fastpath_docs.py
python3 tools/audit_deploy_smoke_docs.py
python3 tools/audit_release_checklist.py
pytest --no-cov tests/test_package_version_sync.py \
  tests/test_publishing_docs_contract.py \
  tests/test_publish_workflow_contract.py \
  tests/test_artifact_workflow_contract.py -q
pytest --no-cov tests/test_artifact_portability_e2e.py \
  tests/test_artifact_security_e2e.py -q
```

For 0.10.x, also confirm `ARTIFACT_SCHEMA_VERSION == 1`, the release notes
describe the trust boundary and migration path, and the legacy compatibility
surfaces retain their documented earliest-removal version (`0.11.0`).

If the release includes benchmark-facing changes, also keep the benchmark docs
and official preset references aligned.

For deploy-facing releases, also run the shortest audited handoff gate:

```bash
pyimgano-doctor --profile deploy-smoke --json
pyimgano bundle validate runs/<run_dir>/deploy_bundle --json
pyimgano runs acceptance runs/<run_dir> --require-status audited --check-bundle-hashes --json
```

That checklist should leave the deploy bundle carrying:

- `deploy_bundle/bundle_manifest.json`
- `deploy_bundle/handoff_report.json`

For benchmark-facing releases, also verify the trust contract surfaces are present:

- `leaderboard_metadata.json` carries `evaluation_contract`
- `leaderboard_metadata.json` carries benchmark `citation` when an official config is used
- `leaderboard_metadata.json` carries `audit_refs.report_json|config_json|environment_json`
- `leaderboard_metadata.json` carries matching `audit_digests.report_json|config_json|environment_json`
- `leaderboard_metadata.json` carries matching `exported_file_digests.*` for exported leaderboard tables
- `pyimgano-runs quality` exposes `trust_summary`
- `pyimgano-runs publication` exposes `trust_signals`

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
python -m twine check dist/*
```

## 5) Upload to TestPyPI (recommended)

```bash
python -m twine upload --repository testpypi dist/*
```

Then install from TestPyPI to validate the install experience:

```bash
pip install -i https://test.pypi.org/simple/ pyimgano
```

## 6) Upload to PyPI (official)

```bash
python -m twine upload dist/*
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

Before upload, the workflow calls the reusable
`.github/workflows/artifact-e2e.yml` gate and then runs a **Release Readiness**
job. The artifact gate:

- builds exactly one wheel and installs that wheel, never an editable checkout,
  into fresh environments
- independently exports, relocates, loads, and infers native, ONNX,
  TorchScript, and OpenVINO artifacts
- proves the ONNX and OpenVINO runtime-only environments have no importable
  `torch` module
- runs `pip check`, every published console script, documentation command
  contracts, workflow/version contracts, and artifact security negative tests
- treats a skipped declared runtime, conditional job, or `continue-on-error` as
  a gate failure

The **Release Readiness** job then:

- audits release/docs surfaces (`audit_release_surface`, `audit_adoption_docs`, `audit_audited_fastpath_docs`, `audit_deploy_smoke_docs`, `audit_release_checklist`)
- runs the deploy-smoke chain end-to-end
- requires `pyimgano bundle validate ./_release_deploy_smoke_run/deploy_bundle --json`
- requires `pyimgano runs acceptance ./_release_deploy_smoke_run --require-status audited --check-bundle-hashes --json`

The publish job depends on both gates. It downloads the exact wheel that passed
artifact E2E, builds only the source distribution, runs
`python -m twine check dist/*`, and uploads that tested wheel plus the sdist.
It does not rebuild an untested wheel.

### Release-certified artifact matrix

The 0.10.0 release gate certifies only these artifact cells:

| Format | OS | Python | Backend/provider | Runtime extra |
|---|---|---:|---|---|
| Native | Ubuntu x86_64 (`ubuntu-latest`) | 3.10 | `pyimgano` / CPU | Base plus model-specific dependencies |
| ONNX | Ubuntu x86_64 (`ubuntu-latest`) | 3.10 | `onnxruntime` / `CPUExecutionProvider` | `onnx-runtime` |
| TorchScript | Ubuntu x86_64 (`ubuntu-latest`) | 3.10 | `torchscript` / CPU | `torch` |
| OpenVINO | Ubuntu x86_64 (`ubuntu-latest`) | 3.10 | `openvino` / CPU | `openvino-runtime` |

Other cells are not release-certified artifact combinations, even when the base
package's broader compatibility suite passes on that OS or Python version.

### Artifact trust and migration checks

Schema-v1 artifacts must pass contained-path, symlink, size, digest, schema, and
component-closure validation before backend construction. Executable checkpoint
deserialization remains disabled unless the caller explicitly chooses
`trust_checkpoint`; integrity verification still applies after that opt-in.

Migrate a persisted fitted run with:

```bash
pyimgano-export --from-run runs/<run_dir> --format native --out exports
pyimgano-infer --artifact exports --input test_images --save-jsonl results.jsonl
```

`pyimgano-export-onnx` / `pyimgano-export-torchscript` are deprecated
embedding-only exporters, not fitted-detector deployment commands. Those scripts
and the `onnx` / `openvino` compatibility extras remain through 0.10.x and are
eligible for removal no earlier than `0.11.0`; new automation should use
`pyimgano-export` and the explicit runtime/export extras.

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

1) Pick a package version that has never been uploaded to PyPI.
2) Update `pyproject.toml` and `pyimgano/__init__.py`, then verify:

```bash
python3 tools/audit_release_version.py --tag vX.Y.Z
```

For prereleases, keep the package version PEP 440-compatible. For example,
`version = "0.10.1rc1"` may be released with tag `v0.10.1-rc1`.

3) Tag and push a release (for example `v0.10.0`).
4) Create a GitHub Release for that tag and click **Publish release**.

That triggers the workflow and uploads to PyPI automatically.

Pushing a git tag by itself is not enough for this workflow; it listens for the
GitHub Release **published** event. PyPI also does not allow overwriting files
for an already-uploaded version, so use a new patch/minor/prerelease version for
any package content that differs from the current PyPI release.

### Publish to TestPyPI (optional)

GitHub → Actions → "Publish to PyPI" → Run workflow → `test_pypi=true`.

## Optional: GitHub Actions "Trusted Publishing"

PyPI supports publishing without long-lived secrets via GitHub Actions (OIDC).
If you want this, set up a "Trusted Publisher" on PyPI that points at this repo,
and add a release workflow that builds and publishes on tags.
