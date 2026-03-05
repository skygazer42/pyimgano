# Industrial baseline suite v4 expansion — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add more **industrial anomaly detection baseline presets** (CPU-friendly feature pipelines + optional texture baselines), publish them as a new suite `industrial-v4`, and provide a small sweep profile for quick parameter scans — while keeping core dependencies lightweight and respecting optional extras boundaries.

**Architecture:** Build on existing primitives:
1) Keep baselines as **JSON-ready presets** in `pyimgano.presets.industrial_classical`.
2) Curate an expanded suite via `pyimgano.baselines.suites` (import-light).
3) Provide a bounded sweep plan via `pyimgano.baselines.sweeps` (import-light).
4) Validate via CLI discovery tests (`pyimgano-benchmark --list-suites/--suite-info/--list-sweeps`).

**Tech Stack:** Python, NumPy, scikit-learn, OpenCV (core) + scikit-image (optional `pyimgano[skimage]`).

---

## Phase A — Presets (Tasks 1–12)

### Task 1: Add JSON-friendly helper for `edge_stats` feature extractor
**Files:** Modify `pyimgano/presets/industrial_classical.py`  
**Notes:** keep kwargs scalar-only; no runtime imports.

### Task 2: Add JSON-friendly helper for `patch_stats` feature extractor
**Files:** Modify `pyimgano/presets/industrial_classical.py`  
**Notes:** represent tuples as lists for JSON friendliness.

### Task 3: Add JSON-friendly helper for `color_hist` feature extractor
**Files:** Modify `pyimgano/presets/industrial_classical.py`

### Task 4: Add JSON-friendly helper for `fft_lowfreq` feature extractor
**Files:** Modify `pyimgano/presets/industrial_classical.py`

### Task 5: Add optional texture feature helpers (`lbp`, `hog`, `gabor_bank`)
**Files:** Modify `pyimgano/presets/industrial_classical.py`  
**Notes:** mark these presets optional + `requires_extras=("skimage",)`.

### Task 6: Add preset `industrial-edge-ecod`
**Files:** Modify `pyimgano/presets/industrial_classical.py`  
**Model:** `vision_feature_pipeline` (edge_stats → core_ecod)

### Task 7: Add preset `industrial-patch-stats-ecod`
**Files:** Modify `pyimgano/presets/industrial_classical.py`  
**Model:** `vision_feature_pipeline` (patch_stats → core_ecod)

### Task 8: Add preset `industrial-color-hist-ecod`
**Files:** Modify `pyimgano/presets/industrial_classical.py`  
**Model:** `vision_feature_pipeline` (color_hist → core_ecod)

### Task 9: Add preset `industrial-fft-lowfreq-ecod`
**Files:** Modify `pyimgano/presets/industrial_classical.py`  
**Model:** `vision_feature_pipeline` (fft_lowfreq → core_ecod)

### Task 10: Add optional preset `industrial-lbp-ecod`
**Files:** Modify `pyimgano/presets/industrial_classical.py`  
**Extra:** `skimage`

### Task 11: Add optional preset `industrial-hog-ecod`
**Files:** Modify `pyimgano/presets/industrial_classical.py`  
**Extra:** `skimage`

### Task 12: Add optional preset `industrial-gabor-ecod`
**Files:** Modify `pyimgano/presets/industrial_classical.py`  
**Extra:** `skimage`

---

## Phase B — Suite + Sweeps (Tasks 13–20)

### Task 13: Add suite `industrial-v4`
**Files:** Modify `pyimgano/baselines/suites.py`  
**Notes:** keep older suites stable; `v4` should add the new feature baselines.

### Task 14: Add sweep plan `industrial-feature-small`
**Files:** Modify `pyimgano/baselines/sweeps.py`  
**Notes:** keep bounded variants; focus on stable extractor knobs.

### Task 15: Ensure sweep variants deep-merge cleanly into preset kwargs
**Files:** Modify `pyimgano/baselines/sweeps.py` (if needed)  
**Test:** Add/extend CLI sweep-info test.

### Task 16: Document `industrial-v4` in README algorithm selection examples
**Files:** Modify `README.md`

### Task 17: Document `industrial-v4` in CLI reference suite list
**Files:** Modify `docs/CLI_REFERENCE.md`

### Task 18: Update optional deps docs to use `industrial-v4` as the canonical suite-info example
**Files:** Modify `docs/OPTIONAL_DEPENDENCIES.md`

### Task 19: (Optional) Add short note about `industrial-feature-small` sweep in docs
**Files:** Modify `docs/CLI_REFERENCE.md` (tiny mention)

### Task 20: (Optional) Add suite/sweep summary to `docs/ALGORITHM_SELECTION_GUIDE.md`
**Files:** Modify `docs/ALGORITHM_SELECTION_GUIDE.md`

---

## Phase C — Tests (Tasks 21–27)

### Task 21: Update benchmark CLI suite listing tests to include `industrial-v4`
**Files:** Modify `tests/test_cli_baseline_suites_v16.py`

### Task 22: Update benchmark CLI sweep listing tests to include `industrial-feature-small`
**Files:** Modify `tests/test_cli_baseline_suites_v16.py`

### Task 23: Add suite-info test that verifies `industrial-hog-ecod` is optional and requires `skimage`
**Files:** Modify `tests/test_cli_baseline_suites_v16.py`

### Task 24: Run targeted tests for suite/sweep discovery
Run: `pytest tests/test_cli_baseline_suites_v16.py -q`

### Task 25: Run a small CLI suite run smoke test (custom dataset)
Run: `pytest tests/test_cli_baseline_suites_v16.py::test_benchmark_cli_can_run_suite_smoke -q`

### Task 26: Run import-light sanity (subprocess smoke)
Run: `pytest tests/test_optional_extras_torch_onnx_openvino_v16.py -q`

### Task 27: Run formatting (black/isort) on modified files
Run: `python -m black pyimgano/presets/industrial_classical.py pyimgano/baselines/suites.py pyimgano/baselines/sweeps.py tests/test_cli_baseline_suites_v16.py`  
Run: `python -m isort pyimgano/presets/industrial_classical.py pyimgano/baselines/suites.py pyimgano/baselines/sweeps.py tests/test_cli_baseline_suites_v16.py`

---

## Phase D — Release + Merge (Tasks 28–30)

### Task 28: Bump patch version + changelog
**Files:** Modify `pyproject.toml`, `pyimgano/__init__.py`, `CHANGELOG.md`

### Task 29: Tag and push release
Run: `git tag -a vX.Y.Z -m "release: vX.Y.Z"`  
Run: `git push origin feat/industrial-baselines-v4 --tags`

### Task 30: Merge into `main` and push
Run: `git checkout main && git merge --ff-only feat/industrial-baselines-v4 && git push origin main`

