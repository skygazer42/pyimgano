# PyImgAno Next 100 Tasks (v12) Implementation Plan

> **For Codex/Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make all primary CLIs **offline-safe by default** by disabling implicit pretrained-weight behavior unless explicitly requested.

**Why (industrial):**
- Factory deployments often run in restricted networks; “silent downloads” are an outage risk.
- Explicit `--pretrained` makes weight usage intentional and audit-friendly.
- Keeps unit tests and CI deterministic across environments.

**Scope (v12):**
- `pyimgano-benchmark`: default `--no-pretrained`
- `pyimgano-robust-benchmark`: default `--no-pretrained`
- `pyimgano-infer --model ...` (direct mode): default `--no-pretrained`
- Update docs examples that rely on deep backbones to include `--pretrained` explicitly.

**Constraints:** no new deps, no implicit downloads, one final commit.

---

## Phase 0 — Tests (Guardrails)

### Task 801: Add regression tests for CLI pretrained defaults
**Files:**
- Create: `tests/test_cli_pretrained_defaults_offline_safe_v12.py`

**Verify:**
- `pytest -q -o addopts='' tests/test_cli_pretrained_defaults_offline_safe_v12.py -v`

---

## Phase 1 — Implementation (Defaults)

### Task 811: Make `pyimgano-benchmark` default to `--no-pretrained`
**Files:**
- Modify: `pyimgano/cli.py`

### Task 812: Make `pyimgano-robust-benchmark` default to `--no-pretrained`
**Files:**
- Modify: `pyimgano/robust_cli.py`

### Task 813: Make `pyimgano-infer --model ...` default to `--no-pretrained`
**Files:**
- Modify: `pyimgano/infer_cli.py`

---

## Phase 2 — Docs + Changelog

### Task 821: Update CLI docs to show explicit `--pretrained` where appropriate
**Files:**
- Modify: `docs/CLI_REFERENCE.md`
- Modify: `docs/EVALUATION_AND_BENCHMARK.md`
- Modify: `docs/INDUSTRIAL_INFERENCE.md`
- Modify: `docs/ROBUSTNESS_BENCHMARK.md`
- Modify: `docs/source/quickstart.rst`

### Task 822: Update changelog
**Files:**
- Modify: `CHANGELOG.md`

---

## Phase 3 — Verification + One Final Commit

### Task 891: Run full unit test suite
- `pytest -q -o addopts=''`

### Task 892: Run audits
- `python tools/audit_public_api.py`
- `python tools/audit_registry.py`
- `python tools/audit_score_direction.py`
- `python tools/audit_third_party_notices.py`
- `python tools/audit_import_costs.py`
- `python tools/audit_no_reference_clones_tracked.py`
- `python tools/audit_pixel_map_models.py --strict`

### Task 893: One final commit
- `git status --porcelain`
- `git add -A`
- `git commit -m "feat: industrial v12 (offline-safe pretrained defaults for CLIs)"`

