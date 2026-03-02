# PyImgAno Next 100 Tasks (v8) Implementation Plan

> **For Codex/Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve industrial **discovery UX** under the v7 lazy registry by making `pyimgano --model-info <name>` show the **real constructor signature + accepted kwargs** (materializing a single model module on demand) while keeping default imports lightweight and offline-safe.

**Architecture (v8 focus):**
- Keep v7 invariant: `import pyimgano.models` stays lightweight (no implicit `torch`/`cv2` imports).
- Add a small “materialize model constructor” helper (registry-level) to safely import the owning module for **one** model name.
- Use materialization only where it’s appropriate:
  - CLI `--model-info` (user explicitly asked for details)
  - CLI kwargs validation/build paths (already materialize, but centralize the logic)
  - Audits that need runtime methods (pixel-map audit already materializes pixel models)

**Tech Stack:** Python stdlib (`importlib`, `inspect`), existing model registry + metadata.

---

## Constraints / Non‑Negotiables (Industrial + Repo Policy)

- **No new required dependencies.**
- **No implicit weight downloads** by default.
- **Single final commit policy for this batch:** commit once after full verification.

---

## Phase 0 — Guardrails

### Task 401: Add a regression test: `--model-info` must show real kwargs for a known model
**Design:** call `pyimgano.cli.main(["--model-info", "...", "--json"])` and assert the payload includes an expected kwarg (e.g. `"device"` for `vision_patchcore`).

**Files:**
- Create: `tests/test_cli_model_info_materializes_signature_v8.py`

**Verify:**
- `pytest -q -o addopts='' tests/test_cli_model_info_materializes_signature_v8.py -v`

---

## Phase 1 — Registry Helper (Single-Model Materialization)

### Task 411: Add `materialize_model_constructor(name)` helper in registry
**Design:**
- If the registry entry is a v7 lazy placeholder (`metadata._lazy_placeholder=True`), import `pyimgano.models.<module>` from `metadata._lazy_module`.
- Return the updated constructor after import.
- No instantiation; import only.

**Files:**
- Modify: `pyimgano/models/registry.py`

**Verify:**
- `python -c \"import pyimgano.models as m; from pyimgano.models.registry import materialize_model_constructor; materialize_model_constructor('vision_patchcore'); print(m.MODEL_REGISTRY.info('vision_patchcore').constructor)\"`

---

## Phase 2 — CLI `--model-info` Materialization

### Task 421: Materialize the target model for `--model-info`
**Design:**
- In `pyimgano.cli.main`, when `--model-info NAME` is used, call `materialize_model_constructor(NAME)` before computing `model_info(NAME)`.

**Files:**
- Modify: `pyimgano/cli.py`

**Verify:**
- `pytest -q -o addopts='' tests/test_cli_model_info_materializes_signature_v8.py -v`
- `pytest -q -o addopts='' tests/test_cli_discovery.py -q`

---

## Phase 3 — Final Verification + One Final Commit

### Task 491: Run full unit test suite
Run:
- `pytest -q -o addopts=''`

### Task 492: Run audits
Run:
- `python tools/audit_public_api.py`
- `python tools/audit_registry.py`
- `python tools/audit_score_direction.py`
- `python tools/audit_third_party_notices.py`
- `python tools/audit_import_costs.py`
- `python tools/audit_no_reference_clones_tracked.py`
- `python tools/audit_pixel_map_models.py --strict`

### Task 493: Update changelog
Files:
- Modify: `CHANGELOG.md`

### Task 494: One final commit (single commit policy)
Run:
- `git status --porcelain`
- `git add -A`
- `git commit -m \"feat: industrial v8 (materialized model-info under lazy registry)\"`

