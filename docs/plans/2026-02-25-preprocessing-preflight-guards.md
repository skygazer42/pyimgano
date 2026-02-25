# Preprocessing Preflight Guards Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add preflight/runtime guards so `preprocessing.illumination_contrast` fails early and clearly when used with non-numpy-capable models, and provide an example config.

**Architecture:** Add a lightweight capability check in workbench preflight (model registry + computed capabilities). Add an example config under `examples/configs/`. Keep behavior additive and deterministic (no silent fallback).

**Tech Stack:** Python, model registry (`pyimgano.models.registry`), capabilities (`pyimgano.models.capabilities`), workbench preflight (`pyimgano.workbench.preflight`), pytest.

---

### Task 1: Add a failing preflight test

**Files:**
- Create: `tests/test_workbench_preflight_preprocessing.py`

**Step 1: Write the failing test**

```python
def test_preflight_errors_when_preprocessing_enabled_on_non_numpy_model(tmp_path):
    # register dummy non-numpy model
    # run preflight with preprocessing enabled
    # assert PREPROCESSING_REQUIRES_NUMPY_MODEL in issues
    ...
```

**Step 2: Run test to verify it fails**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_workbench_preflight_preprocessing.py
```

Expected: FAIL (missing preflight guard).

---

### Task 2: Implement preflight capability guard

**Files:**
- Modify: `pyimgano/workbench/preflight.py`

**Step 1: Add best-effort model capability check**

- If `config.preprocessing.illumination_contrast` is set:
  - try `import pyimgano.models` (registry population)
  - try `MODEL_REGISTRY.info(config.model.name)`
  - compute `input_modes` via `compute_model_capabilities(entry)`
  - if `"numpy"` not in input modes → add `error` issue:
    - code: `PREPROCESSING_REQUIRES_NUMPY_MODEL`

**Step 2: Run the test**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_workbench_preflight_preprocessing.py
```

Expected: PASS.

---

### Task 3: Add a workbench example config

**Files:**
- Create: `examples/configs/industrial_adapt_preprocessing_illumination.json`
- Modify (optional): `docs/WORKBENCH.md`

**Step 1: Add config**

- Base it on `industrial_adapt_fast.json`
- Add `preprocessing.illumination_contrast` with a conservative default:
  - `white_balance="gray_world"`
  - `clahe=true`, `gamma=0.9`

**Step 2: Mention in docs**

- Add it to the quickstart “Start from an example config” list in `docs/WORKBENCH.md`.

---

### Task 4: Verify and release

**Files:**
- Modify: `CHANGELOG.md`
- Modify: `pyproject.toml`
- Modify: `pyimgano/__init__.py`

**Step 1: Run focused test set**

```bash
.venv/bin/python -m pytest -q \
  tests/test_workbench_preflight_preprocessing.py \
  tests/test_workbench_preprocessing_config.py \
  tests/test_infer_cli_smoke.py
```

**Step 2: Bump version + changelog**

- `0.6.23`
- Add bullet(s) for preflight guard + example config.

**Step 3: Commit + tag + push**

```bash
git add -A
git commit -m "Release 0.6.23 (preflight guard for preprocessing)"
git tag v0.6.23
git push origin main
git push origin v0.6.23
```

