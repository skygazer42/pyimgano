# Industrial Robustness + VAND-Style SegF1 Benchmark Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a deploy-style pixel SegF1 benchmark with a single global threshold, plus a deterministic industrial drift robustness suite (clean + corruptions).

**Architecture:** Extend existing pixel evaluation to compute SegF1 + background FPR under a single calibrated threshold, add a `pyimgano.robustness` module with deterministic corruptions, and expose everything via CLI + JSON reports.

**Tech Stack:** Python 3.9+, NumPy, OpenCV, scikit-learn (already), existing `pyimgano` pipelines/CLI.

---

### Task 1: Add pixel SegF1 + bg-FPR helpers

**Files:**
- Modify: `pyimgano/evaluation.py`
- Test: `tests/test_evaluation_pixel.py`

**Step 1: Write the failing test**

Add a test that checks SegF1 is computed correctly for a tiny mask/score tensor using one threshold.

**Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest tests/test_evaluation_pixel.py -q`  
Expected: FAIL with `AttributeError` / missing function.

**Step 3: Write minimal implementation**

Implement:
- `compute_pixel_segf1(pixel_labels, pixel_scores, threshold)`
- `compute_bg_fpr(pixel_labels, pixel_scores, threshold)`

**Step 4: Run test to verify it passes**

Run: `./.venv/bin/python -m pytest tests/test_evaluation_pixel.py -q`  
Expected: PASS.

**Step 5: Commit**

Run:
```bash
git add pyimgano/evaluation.py tests/test_evaluation_pixel.py
git commit -m "feat: add pixel SegF1 and bg FPR metrics"
```

---

### Task 2: Add pixel threshold calibration (normal-pixel quantile)

**Files:**
- Create: `pyimgano/calibration/pixel_threshold.py`
- Modify: `pyimgano/calibration/__init__.py`
- Test: `tests/test_evaluation_pixel.py`

**Step 1: Write the failing test**

Test that a set of “normal pixels” calibrates to `np.quantile(scores, q)`.

**Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest tests/test_evaluation_pixel.py -q`  
Expected: FAIL missing function/module.

**Step 3: Write minimal implementation**

Add:
- `calibrate_normal_pixel_quantile_threshold(pixel_scores, pixel_labels=None, q=0.999)`
  - if `pixel_labels` provided: use `pixel_labels==0` pixels only
  - if not provided: use all pixels (assumes all-normal calibration set)

**Step 4: Run test to verify it passes**

Run: `./.venv/bin/python -m pytest tests/test_evaluation_pixel.py -q`

**Step 5: Commit**

```bash
git add pyimgano/calibration/pixel_threshold.py pyimgano/calibration/__init__.py tests/test_evaluation_pixel.py
git commit -m "feat: add normal-pixel quantile threshold calibration"
```

---

### Task 3: Extend `evaluate_detector()` to optionally report SegF1 under a fixed threshold

**Files:**
- Modify: `pyimgano/evaluation.py`
- Test: `tests/test_evaluation_pixel.py`

**Step 1: Write failing test**

Call `evaluate_detector(..., pixel_labels=..., pixel_scores=..., pixel_threshold=...)` and assert it returns:
- `pixel_metrics.pixel_segf1`
- `pixel_metrics.bg_fpr`

**Step 2: Run test (should fail)**

Run: `./.venv/bin/python -m pytest tests/test_evaluation_pixel.py -q`

**Step 3: Implement minimal code**

Add optional params:
- `pixel_threshold: Optional[float]`
- (and/or) `pixel_threshold_strategy` handled at pipeline level (preferred)

Keep defaults backward-compatible: only compute SegF1/bg FPR when threshold is provided.

**Step 4: Re-run tests**

Run: `./.venv/bin/python -m pytest tests/test_evaluation_pixel.py -q`

**Step 5: Commit**

```bash
git add pyimgano/evaluation.py tests/test_evaluation_pixel.py
git commit -m "feat: report pixel SegF1/bg FPR when threshold provided"
```

---

### Task 4: Add calibration split selection helper (validation-good or train hold-out)

**Files:**
- Create: `pyimgano/utils/splits.py`
- Test: `tests/test_pipeline_smoke.py` (or a new unit test)

**Step 1: Write failing test**

Test that:
- with N inputs and seed fixed, the hold-out split is deterministic
- splits are disjoint and cover the original list

**Step 2: Run test (should fail)**

Run: `./.venv/bin/python -m pytest tests/test_pipeline_smoke.py -q`

**Step 3: Implement**

Add:
- `split_train_calibration(paths, calibration_fraction=0.2, seed=0) -> (train_paths, calib_paths)`

**Step 4: Re-run**

Run: `./.venv/bin/python -m pytest tests/test_pipeline_smoke.py -q`

**Step 5: Commit**

```bash
git add pyimgano/utils/splits.py tests/test_pipeline_smoke.py
git commit -m "feat: deterministic train/calibration split helper"
```

---

### Task 5: Extend pipeline evaluation to compute SegF1 with `normal_pixel_quantile`

**Files:**
- Modify: `pyimgano/pipelines/mvtec_visa.py`
- Test: `tests/test_pipeline_pixel_scores.py`

**Step 1: Write failing test**

Use a dummy detector that returns deterministic anomaly maps. Assert:
- calibration threshold computed from calibration normal maps
- SegF1 computed on test maps with the calibrated threshold

**Step 2: Run test (should fail)**

Run: `./.venv/bin/python -m pytest tests/test_pipeline_pixel_scores.py -q`

**Step 3: Implement**

Add optional args to `evaluate_split()`:
- `pixel_segf1: bool`
- `pixel_threshold_strategy: Literal["normal_pixel_quantile","none"]`
- `pixel_normal_quantile: float`
- `calibration_fraction: float`
- `calibration_seed: int`

Return the calibrated threshold in the results JSON.

**Step 4: Re-run tests**

Run: `./.venv/bin/python -m pytest tests/test_pipeline_pixel_scores.py -q`

**Step 5: Commit**

```bash
git add pyimgano/pipelines/mvtec_visa.py tests/test_pipeline_pixel_scores.py
git commit -m "feat: pipeline SegF1 + normal-quantile pixel threshold"
```

---

### Task 6: Add CLI flags for SegF1 + pixel threshold strategy

**Files:**
- Modify: `pyimgano/cli.py`
- Test: `tests/test_cli_smoke.py`

**Step 1: Write failing CLI test**

Add a smoke test that parses:
- `--pixel-segf1`
- `--pixel-threshold-strategy normal_pixel_quantile`
- `--pixel-normal-quantile 0.999`

**Step 2: Run test (should fail)**

Run: `./.venv/bin/python -m pytest tests/test_cli_smoke.py -q`

**Step 3: Implement flags + wiring**

CLI should forward args into `evaluate_split(...)`.

**Step 4: Re-run**

Run: `./.venv/bin/python -m pytest tests/test_cli_smoke.py -q`

**Step 5: Commit**

```bash
git add pyimgano/cli.py tests/test_cli_smoke.py
git commit -m "feat(cli): expose pixel SegF1 + threshold strategy"
```

---

### Task 7: Create robustness package skeleton

**Files:**
- Create: `pyimgano/robustness/__init__.py`
- Create: `pyimgano/robustness/types.py`
- Test: `tests/test_optional_deps.py` (or a new unit test)

**Step 1: Write failing test**

Import `pyimgano.robustness` and ensure it’s lightweight (no heavy deps imported).

**Step 2: Run**

Run: `./.venv/bin/python -m pytest tests/test_optional_deps.py -q`

**Step 3: Implement minimal package**

Define a small `Corruption` protocol/type + severity mapping constants.

**Step 4: Re-run**

Run: `./.venv/bin/python -m pytest tests/test_optional_deps.py -q`

**Step 5: Commit**

```bash
git add pyimgano/robustness/__init__.py pyimgano/robustness/types.py tests/test_optional_deps.py
git commit -m "feat: add robustness module skeleton"
```

---

### Task 8: Implement deterministic lighting corruption

**Files:**
- Create: `pyimgano/robustness/corruptions.py`
- Test: `tests/test_robustness_corruptions.py`

**Step 1: Write failing test**

Given a fixed seed and severity, output must be bitwise identical across runs.

**Step 2: Run**

Run: `./.venv/bin/python -m pytest tests/test_robustness_corruptions.py -q`

**Step 3: Implement**

Add:
- `apply_lighting(image, *, severity, rng)` (brightness/contrast/gamma + WB gain)

**Step 4: Re-run**

Run: `./.venv/bin/python -m pytest tests/test_robustness_corruptions.py -q`

**Step 5: Commit**

```bash
git add pyimgano/robustness/corruptions.py tests/test_robustness_corruptions.py
git commit -m "feat: add deterministic lighting corruption"
```

---

### Task 9: Implement deterministic JPEG corruption

**Files:**
- Modify: `pyimgano/robustness/corruptions.py`
- Test: `tests/test_robustness_corruptions.py`

**Steps:** add `apply_jpeg(image, severity, rng)` calling existing `jpeg_compress()` with a severity→quality mapping.

Commit: `git commit -m "feat: add deterministic jpeg corruption"`

---

### Task 10: Implement deterministic blur corruption

**Files:**
- Modify: `pyimgano/robustness/corruptions.py`
- Test: `tests/test_robustness_corruptions.py`

**Steps:** add `apply_blur(...)` using severity→(defocus radius / gaussian sigma) mapping.

Commit: `git commit -m "feat: add deterministic blur corruption"`

---

### Task 11: Implement deterministic glare/specular corruption

**Files:**
- Modify: `pyimgano/robustness/corruptions.py`
- Test: `tests/test_robustness_corruptions.py`

**Steps:** add `apply_glare(...)` with one or more synthetic blobs/lines, deterministic via RNG.

Commit: `git commit -m "feat: add deterministic glare corruption"`

---

### Task 12: Implement deterministic geo-jitter corruption (image + mask warp)

**Files:**
- Modify: `pyimgano/robustness/corruptions.py`
- Test: `tests/test_robustness_corruptions.py`

**Steps:** add `apply_geo_jitter(image, mask, ...)` using `cv2.warpAffine`. Mask uses nearest-neighbor.

Commit: `git commit -m "feat: add deterministic geo-jitter corruption"`

---

### Task 13: Add a robustness benchmark runner (clean + corrupted)

**Files:**
- Create: `pyimgano/robustness/benchmark.py`
- Test: `tests/test_robustness_benchmark.py`

**Step 1: Write failing test**

Use a dummy detector that returns stable maps and check the benchmark JSON schema includes:
- per-corruption results
- aggregated means
- latency (best-effort)

**Step 2: Run**

Run: `./.venv/bin/python -m pytest tests/test_robustness_benchmark.py -q`

**Step 3: Implement**

Implement:
- `run_robustness_benchmark(detector, split, corruptions, severities, ...)`
- Evaluate clean once, then each corruption.

**Step 4: Re-run**

Run: `./.venv/bin/python -m pytest tests/test_robustness_benchmark.py -q`

**Step 5: Commit**

```bash
git add pyimgano/robustness/benchmark.py tests/test_robustness_benchmark.py
git commit -m "feat: add robustness benchmark runner"
```

---

### Task 14: Add `pyimgano-robust-benchmark` CLI entry

**Files:**
- Create: `pyimgano/robust_cli.py`
- Modify: `pyproject.toml`
- Test: `tests/test_cli_smoke.py`

**Steps:** implement CLI parsing, call robustness runner, write JSON.

Commit: `git commit -m "feat(cli): add pyimgano-robust-benchmark command"`

---

### Task 15: Add an “industrial drift” augmentation preset for training

**Files:**
- Modify: `pyimgano/preprocessing/augmentation_pipeline.py`
- Test: `tests/test_industrial_augmentation.py`

**Steps:** add `get_industrial_drift_augmentation()` that composes blur/jpeg/gamma/channel gain/glare.

Commit: `git commit -m "feat: add industrial drift augmentation preset"`

---

### Task 16: Add `vision_superad` model entry (DINO patch-kNN, k-th NN distance)

**Files:**
- Create: `pyimgano/models/superad.py`
- Modify: `pyimgano/models/__init__.py`
- Test: `tests/test_superad_core.py`

**Steps:** implement injectable embedder for tests, kNN memory bank, pixel map output.

Commit: `git commit -m "feat: add vision_superad (DINO patch-kNN baseline)"`

---

### Task 17: (Optional) Add `vision_snarm` experimental entry (behind `pyimgano[mamba]`)

**Files:**
- Create: `pyimgano/models/snarm.py`
- Modify: `pyimgano/models/__init__.py`
- Test: `tests/test_snarm_optional.py`

**Steps:** keep it optional + lazy import; skip if scope is tight.

Commit: `git commit -m "feat: add optional vision_snarm entry point"`

---

### Task 18: Documentation for robustness benchmark

**Files:**
- Create: `docs/ROBUSTNESS_BENCHMARK.md`
- Modify: `README.md`

Commit: `git commit -m "docs: add robustness benchmark guide"`

---

### Task 19: Release prep (version + changelog)

**Files:**
- Modify: `pyproject.toml`
- Modify: `CHANGELOG.md`

Commit: `git commit -m "chore: release 0.5.0"`

---

### Task 20: Full verification + merge to main + tag

**Step 1: Run full test suite**

Run: `./.venv/bin/python -m pytest -q`  
Expected: `0 failed`.

**Step 2: Merge/squash to main**

From the main worktree:
```bash
git checkout main
git merge --squash feat-industrial-robust-v0.5.0
git commit -m "v0.5.0: robustness benchmark + SegF1"
```

**Step 3: Tag + push**

```bash
git tag v0.5.0
git push origin main --tags
```

