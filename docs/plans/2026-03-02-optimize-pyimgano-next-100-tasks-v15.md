# PyImgAno Next 100 Tasks (v15) Implementation Plan

> **For Codex/Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Strengthen the industrial “deep embedding + classical core” route by adding a **TorchScript embedding extractor** that:
- requires an explicit local checkpoint path (no downloads),
- stays import-light (`import pyimgano.features` must not pull heavy roots),
- supports optional disk caching for path inputs.

**Architecture:**
- New feature extractor `torchscript_embed` lives under `pyimgano.features.*` and follows the existing registry pattern.
- The extractor performs preprocessing via PIL + NumPy (no torchvision required), then runs inference via `torch.jit.load`.
- It integrates naturally with `vision_embedding_core` and other feature-pipeline models.

**Constraints:**
- No new required dependencies.
- No implicit downloads.
- Keep imports lightweight.
- Avoid adding broad `try/except` in model/extractor code where deterministic checks are possible.
- No TensorRT (`tensorrt` / `trt`) imports anywhere under `pyimgano/`.

---

## Phase 0 — Feature Extractor + Registry Wiring

### Task 1101: Add `torchscript_embed` feature extractor
**Files:**
- Create: `pyimgano/features/torchscript_embed.py`
- Modify: `pyimgano/features/__init__.py`

**Verify:**
- `python -c "from pyimgano.features import list_feature_extractors; assert 'torchscript_embed' in list_feature_extractors()"`

### Task 1102: Add smoke test + import-light guardrail test
**Files:**
- Create: `tests/test_feature_torchscript_embed_extractor.py`
- Create: `tests/test_import_pyimgano_features_is_lightweight_v15.py`

**Verify:**
- `pytest -q -o addopts='' tests/test_feature_torchscript_embed_extractor.py -v`
- `pytest -q -o addopts='' tests/test_import_pyimgano_features_is_lightweight_v15.py -v`

---

## Phase 1 — Docs

### Task 1111: Document the extractor as an industrial deployment option
**Files:**
- Modify: `docs/FEATURE_EXTRACTORS.md`
- Modify: `docs/INDUSTRIAL_EMBEDDING_PLUS_CORE.md`
- Modify: `docs/RECIPES_EMBEDDINGS_PLUS_CORE.md`

**Verify:**
- `python -c "import pathlib; assert 'torchscript_embed' in pathlib.Path('docs/FEATURE_EXTRACTORS.md').read_text(encoding='utf-8')"`

---

## Phase 2 — Verification + One Final Commit

### Task 1191: Run full unit test suite (no addopts)
- `pytest -q -o addopts=''`

### Task 1192: Run audits
- `python tools/audit_public_api.py`
- `python tools/audit_registry.py`
- `python tools/audit_score_direction.py`
- `python tools/audit_third_party_notices.py`
- `python tools/audit_import_costs.py`
- `python tools/audit_no_reference_clones_tracked.py`
- `python tools/audit_no_tensorrt_imports.py`
- `python tools/audit_pixel_map_models.py --strict`

### Task 1193: Update changelog and commit once
**Files:**
- Modify: `CHANGELOG.md`

**Run:**
- `git status --porcelain`
- `git add -A`
- `git commit -m "feat: industrial v15 (torchscript embedding extractor)"`

