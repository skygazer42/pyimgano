# PyImgAno Next 100 Tasks (v7) Implementation Plan

> **For Codex/Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce **import-time cost** and **heavy implicit imports** by making `import pyimgano.models` **lazy + auditable**, while preserving model discovery (`--list-models`, tag filters, metadata) and keeping all deep models **offline-safe by default**.

**Architecture (v7 focus):**
- Keep `BaseDetector` semantics as the “physics law”: **higher score ⇒ more anomalous**.
- Keep discovery fast: `pyimgano.models` should populate the registry without importing heavy roots (`torch`, `cv2`, `open_clip`, `diffusers`, ...).
- Keep runtime behavior unchanged: constructing a model (`create_model(...)`) is allowed to import its dependencies and fail with actionable errors.
- Keep registry metadata stable: tags + metadata should still work for CLI filters without importing the underlying model modules.

**Tech Stack:** Python stdlib (`ast`, `inspect`, `importlib`), NumPy/sklearn/OpenCV/Torch as optional runtime deps (must not be imported implicitly by discovery).

---

## Constraints / Non‑Negotiables (Industrial + Repo Policy)

- **No new required dependencies.**
- **No implicit weight downloads** (torchvision/openclip/diffusers/torch.hub). Pretrained weights must be explicit opt-in.
- **No bundled weights / large assets / datasets** in git.
- **Single final commit policy for this batch:** no incremental commits; commit once after full verification.

---

## Phase 0 — Guardrails (Import-Safety Regression Tests)

### Task 301: Add a regression test: importing `pyimgano.models` must not import heavy roots
**Why:** Discovery (`--list-models`) should be cheap and offline/auditable; heavy deps should load only when constructing models.

**Files:**
- Create: `tests/test_import_pyimgano_models_is_lightweight_v7.py`

**Verify:**
- `pytest -q -o addopts='' tests/test_import_pyimgano_models_is_lightweight_v7.py -v`

---

## Phase 1 — Lazy Model Registry Population (Source Scan + Lazy Constructors)

### Task 311: Populate model registry by scanning source (no module imports)
**Design:**
- Keep the current “module allowlist” (the list previously passed to `_auto_import([...])`) as the authoritative set.
- For each module in the allowlist:
  - Parse its AST and find `@register_model("name", tags=..., metadata=...)`.
  - Register a **lazy constructor** for each discovered model that imports the owning module on first call.
  - Preserve tags + metadata (best-effort `ast.literal_eval` for constants).
  - Apply the existing pixel-map auto-tagging rule at scan time: if the decorated class defines `predict_anomaly_map`/`get_anomaly_map`, ensure the `pixel_map` tag exists.

**Files:**
- Modify: `pyimgano/models/__init__.py`
- Modify: `pyimgano/models/registry.py` (allow real registrations to replace lazy placeholders)

**Verify:**
- `python -c "import pyimgano.models as m; print(len(m.list_models()))"`
- `python -c \"import pyimgano.models as m; print('vision_patchcore' in m.list_models(tags=['vision','deep']))\"`

### Task 312: Keep optional re-exports lazy (avoid importing `cv2` / `torch` for re-exports)
**Files:**
- Modify: `pyimgano/models/__init__.py` (PEP 562 `__getattr__` for `VisionLODA`/`VAEAnomalyDetector`/`OptimizedAEDetector`)

---

## Phase 2 — Base Class Import Hygiene

### Task 321: Remove eager `torch/torchvision` imports from `BaseVisionDeepDetector`
**Why:** `pyimgano.models` re-exports `BaseVisionDeepDetector`; importing it must not implicitly import torch.

**Files:**
- Modify: `pyimgano/models/baseCv.py`

**Verify:**
- Covered by Task 301 test.

---

## Phase 3 — Tooling Updates

### Task 331: Make pixel-map audit robust to lazy constructors
**Design:**
- Ensure `tools/audit_pixel_map_models.py` validates pixel-map contract against the **real constructors**, not lazy placeholders.

**Files:**
- Modify: `tools/audit_pixel_map_models.py`
- Verify via existing test: `pytest -q -o addopts='' tests/test_pixel_map_audit_strict_v6.py -v`

---

## Phase 4 — Final Verification + One Final Commit

### Task 391: Run full unit test suite
Run:
- `pytest -q -o addopts=''`

### Task 392: Run audits
Run:
- `python tools/audit_public_api.py`
- `python tools/audit_registry.py`
- `python tools/audit_score_direction.py`
- `python tools/audit_third_party_notices.py`
- `python tools/audit_import_costs.py`
- `python tools/audit_no_reference_clones_tracked.py`
- `python tools/audit_pixel_map_models.py --strict`

### Task 393: Update changelog
Files:
- Modify: `CHANGELOG.md`

### Task 394: One final commit (single commit policy)
Run:
- `git status --porcelain`
- `git add -A`
- `git commit -m \"feat: industrial v7 (lazy model registry + import-cost guardrails)\"`

