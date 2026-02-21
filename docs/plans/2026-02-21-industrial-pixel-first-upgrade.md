# Industrial Pixel-First Upgrade (MVTec/VisA) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `pyimgano` “pixel-first” strong for industrial visual anomaly detection (MVTec/VisA), improving localization robustness, API consistency, and CLI/docs ergonomics while keeping the core install lightweight.

**Architecture:** Keep `pyimgano` as the stable API + datasets + pixel-eval + postprocess + reporting layer. Strengthen patch-token pipelines with shared helpers, add a SoftPatch-inspired robust patch-memory detector, and expand inference-first `anomalib` backend aliases.

**Tech Stack:** Python, NumPy, OpenCV, scikit-learn, PyTorch/TorchVision; optional `faiss-cpu`; optional `anomalib`.

---

### Task 1: Add patch-kNN helper module (numpy-only)

**Files:**
- Create: `pyimgano/models/patchknn_core.py`
- Test: `tests/test_patchknn_core.py`

**Step 1: Write the failing test**

Create `tests/test_patchknn_core.py` with:

```python
import numpy as np
import pytest

from pyimgano.models.patchknn_core import (
    aggregate_patch_scores,
    reshape_patch_scores,
)


def test_aggregate_patch_scores_topk_mean():
    scores = np.arange(100, dtype=np.float32)
    out = aggregate_patch_scores(scores, method="topk_mean", topk=0.1)
    assert 90.0 <= out <= 99.0


def test_reshape_patch_scores_requires_exact_count():
    with pytest.raises(ValueError, match="Expected"):
        reshape_patch_scores(np.ones((3,), dtype=np.float32), grid_h=2, grid_w=2)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_patchknn_core.py -v`  
Expected: FAIL (module not found)

**Step 3: Write minimal implementation**

Implement:
- `aggregate_patch_scores(scores, method, topk)`
- `reshape_patch_scores(scores, grid_h, grid_w)`

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_patchknn_core.py -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add pyimgano/models/patchknn_core.py tests/test_patchknn_core.py
git commit -m "feat: add patch-kNN core helpers"
```

---

### Task 2: Refactor `vision_anomalydino` to use patch-kNN helpers

**Files:**
- Modify: `pyimgano/models/anomalydino.py`
- Modify: `pyimgano/models/openclip_backend.py`
- Test: `tests/test_anomalydino_core.py`
- Test: `tests/test_openclip_promptscore_core.py`

**Step 1: Write the failing test**

Update an existing test to import helpers from the new module (or add a new test):

```python
from pyimgano.models.patchknn_core import aggregate_patch_scores
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_anomalydino_core.py -v`  
Expected: FAIL until refactor lands.

**Step 3: Implement minimal refactor**

- Move `_aggregate_patch_scores` → `aggregate_patch_scores`
- Move `_reshape_patch_scores` → `reshape_patch_scores`
- Update import sites (AnomalyDINO, OpenCLIP backend) to use new names.

**Step 4: Run tests**

Run: `pytest tests/test_anomalydino_core.py tests/test_openclip_promptscore_core.py -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add pyimgano/models/anomalydino.py pyimgano/models/openclip_backend.py tests/test_anomalydino_core.py
git commit -m "refactor: share patch-kNN helpers across detectors"
```

---

### Task 3: Add optional coreset sampling to `vision_anomalydino`

**Files:**
- Modify: `pyimgano/models/anomalydino.py`
- Test: `tests/test_anomalydino_core.py`

**Step 1: Write the failing test**

```python
import numpy as np

from pyimgano.models.anomalydino import VisionAnomalyDINO


def test_anomalydino_coreset_sampling_reduces_bank(fake_embedder):
    det = VisionAnomalyDINO(embedder=fake_embedder, coreset_sampling_ratio=0.5)
    det.fit(["a.png", "b.png", "c.png"])
    assert det.memory_bank_size_ > 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_anomalydino_core.py -v`  
Expected: FAIL (`coreset_sampling_ratio` / `memory_bank_size_` not implemented)

**Step 3: Implement**

- Add `coreset_sampling_ratio` parameter (default `1.0`).
- Add `memory_bank_size_` property for introspection.
- Sampling strategy: deterministic RNG seed via `np.random.default_rng(0)` for unit tests.

**Step 4: Re-run tests**

Run: `pytest tests/test_anomalydino_core.py -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add pyimgano/models/anomalydino.py tests/test_anomalydino_core.py
git commit -m "feat: add coreset sampling to vision_anomalydino"
```

---

### Task 4: Add a SoftPatch-inspired robust patch-memory detector (core skeleton)

**Files:**
- Create: `pyimgano/models/softpatch.py`
- Modify: `pyimgano/models/__init__.py`
- Test: `tests/test_softpatch_core.py`

**Step 1: Write the failing test**

```python
import numpy as np
from pyimgano.models import create_model


def test_softpatch_registry_and_basic_api(tmp_path, make_synthetic_images):
    paths = make_synthetic_images(tmp_path)
    det = create_model("vision_softpatch", pretrained=False, device="cpu", coreset_sampling_ratio=1.0)
    det.fit(paths["normal"])
    scores = det.decision_function(paths["all"])
    assert scores.shape == (len(paths["all"]),)
    m = det.get_anomaly_map(paths["all"][0])
    assert m.ndim == 2
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_softpatch_core.py -v`  
Expected: FAIL (model not found)

**Step 3: Implement minimal detector**

- Implement a PatchCore-like baseline with:
  - wide_resnet50 feature extraction
  - coreset memory bank
  - kNN distances → patch scores → anomaly map
- Register as `vision_softpatch` with tags: `vision, deep, softpatch, patchknn, robust`.

**Step 4: Run tests**

Run: `pytest tests/test_softpatch_core.py -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add pyimgano/models/softpatch.py pyimgano/models/__init__.py tests/test_softpatch_core.py
git commit -m "feat: add vision_softpatch robust patch-memory detector"
```

---

### Task 5: Implement robust memory filtering for SoftPatch

**Files:**
- Modify: `pyimgano/models/softpatch.py`
- Test: `tests/test_softpatch_core.py`

**Step 1: Write the failing test**

Add a test that includes “contaminated normal” patches and asserts that robust filtering
removes some patches.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_softpatch_core.py -v`  
Expected: FAIL

**Step 3: Implement**

- During `fit()`, compute a simple patch outlier score on the training patch set:
  - `score = ||x - mean||` (fast baseline) or kNN distance within train patches (slower).
- Remove the top `train_patch_outlier_quantile` fraction from the memory bank.

**Step 4: Re-run tests**

Run: `pytest tests/test_softpatch_core.py -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add pyimgano/models/softpatch.py tests/test_softpatch_core.py
git commit -m "feat: robust memory filtering for vision_softpatch"
```

---

### Task 6: Add `vision_softpatch` docs entry

**Files:**
- Modify: `docs/DEEP_LEARNING_MODELS.md`

**Step 1: Write minimal doc section**

Add a section describing:
- when to use SoftPatch (noisy normal training)
- required inputs/outputs
- recommended parameters (coreset ratio, outlier quantile)

**Step 2: Run docs lint (optional)**

No strict docs CI required.

**Step 3: Commit**

```bash
git add docs/DEEP_LEARNING_MODELS.md
git commit -m "docs: add SoftPatch detector guide"
```

---

### Task 7: Expand anomalib backend aliases (Dinomaly / CFA)

**Files:**
- Modify: `pyimgano/models/anomalib_backend.py`
- Test: `tests/test_anomalib_backend_optional.py`

**Step 1: Write failing test**

Add a registry test that `list_models(tags=[\"anomalib\"])` includes new names.

**Step 2: Run test**

Run: `pytest tests/test_anomalib_backend_optional.py -v`  
Expected: FAIL (missing names)

**Step 3: Implement**

Add alias classes:
- `vision_dinomaly_anomalib`
- `vision_cfa_anomalib`

**Step 4: Re-run tests**

Run: `pytest tests/test_anomalib_backend_optional.py -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add pyimgano/models/anomalib_backend.py tests/test_anomalib_backend_optional.py
git commit -m "feat: add anomalib aliases for dinomaly and cfa"
```

---

### Task 8: Add robust normalization option to anomaly-map postprocess

**Files:**
- Modify: `pyimgano/postprocess/anomaly_map.py`
- Test: `tests/test_postprocess_anomaly_map.py`

**Step 1: Write failing test**

Add a case where min/max are extreme and percentile normalization is stable.

**Step 2: Run test**

Run: `pytest tests/test_postprocess_anomaly_map.py -v`  
Expected: FAIL

**Step 3: Implement**

Add `normalize_method: Literal["minmax","percentile","none"]` and
`percentile_range=(1,99)` support.

**Step 4: Re-run**

Run: `pytest tests/test_postprocess_anomaly_map.py -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add pyimgano/postprocess/anomaly_map.py tests/test_postprocess_anomaly_map.py
git commit -m "feat: percentile normalization for anomaly-map postprocess"
```

---

### Task 9: Make CLI use pipeline pixel-map alignment

**Files:**
- Modify: `pyimgano/cli.py`
- Test: `tests/test_cli_smoke.py`

**Step 1: Write failing test**

Update CLI smoke test to request `--pixel` and ensure it succeeds without shape errors.

**Step 2: Run test**

Run: `pytest tests/test_cli_smoke.py -v`  
Expected: FAIL until CLI delegates to pipeline helpers.

**Step 3: Implement**

- Replace ad-hoc map stacking with a call into `pyimgano.pipelines.mvtec_visa.evaluate_split()`.

**Step 4: Re-run**

Run: `pytest tests/test_cli_smoke.py -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add pyimgano/cli.py tests/test_cli_smoke.py
git commit -m "fix: CLI pixel metrics use pipeline map alignment"
```

---

### Task 10: Add CLI options for anomaly-map postprocess

**Files:**
- Modify: `pyimgano/cli.py`
- Test: `tests/test_cli_smoke.py`

**Step 1: Write failing test**

Add `--postprocess` flags (sigma/open/close/threshold/min-area) and ensure CLI still runs.

**Step 2: Implement**

Parse new args and build `AnomalyMapPostprocess(...)` when `--pixel` is enabled.

**Step 3: Commit**

`git commit -m "feat: CLI anomaly-map postprocess flags"`

---

### Task 11: Add an example script for SoftPatch on MVTec/VisA

**Files:**
- Create: `examples/softpatch_mvtec_visa.py`

**Steps:**
- Use `load_benchmark_split()` + `evaluate_split()` to print image + pixel metrics.
- Save JSON report via `pyimgano.reporting.report.save_run_report`.
- Commit: `git commit -m "examples: add softpatch mvtec/visa benchmark script"`

---

### Task 12: Add an example script comparing PatchCore vs AnomalyDINO vs OpenCLIP PatchKNN vs SoftPatch

**Files:**
- Create: `examples/pixel_first_compare.py`

**Steps:**
- Evaluate 3–4 models on one category, output a small summary table.
- Commit: `git commit -m "examples: add pixel-first model comparison script"`

---

### Task 13: Document pixel-first workflow in QUICKSTART

**Files:**
- Modify: `docs/QUICKSTART.md`

**Steps:**
- Add a “Pixel-first industrial workflow” section:
  - dataset paths
  - pick a detector
  - run pipeline
  - interpret pixel metrics
- Commit: `git commit -m "docs: add pixel-first industrial quickstart section"`

---

### Task 14: Update algorithm selection guide with pixel-first recommendations

**Files:**
- Modify: `docs/ALGORITHM_SELECTION_GUIDE.md`

**Steps:**
- Add a “Pixel localization (recommended)” table including:
  - PatchCore
  - AnomalyDINO
  - OpenCLIP PatchKNN
  - SoftPatch
  - anomalib wrappers (Dinomaly/CFA) as optional
- Commit: `git commit -m "docs: pixel-first algorithm selection recommendations"`

---

### Task 15: Add registry alias for WinCLIP naming consistency

**Files:**
- Modify: `pyimgano/models/winclip.py`
- Test: `tests/test_models_import_optional.py`

**Steps:**
- Add a `vision_winclip` registry alias class that reuses `WinCLIPDetector`.
- Ensure existing `winclip` name still works.
- Commit: `git commit -m "fix: add vision_winclip registry alias"`

---

### Task 16: Add a “pixel-map contract” test for key detectors

**Files:**
- Create: `tests/test_pixel_map_contract.py`

**Steps:**
- For a small synthetic image set, verify detectors that claim pixel maps return:
  - `float32`
  - 2D maps
  - finite values
- Keep test fast; use `pretrained=False` on TorchVision backbones.
- Commit: `git commit -m "test: add pixel anomaly-map contract coverage"`

---

### Task 17: Add optional FAISS test coverage for patch-kNN detectors

**Files:**
- Modify: `tests/test_knn_index.py`

**Steps:**
- If `faiss` import available, add a tiny test that `FaissKNNIndex` produces correct shapes.
- Commit: `git commit -m \"test: extend faiss knn index coverage\"`

---

### Task 18: Add report schema stabilization (include pixel metrics when present)

**Files:**
- Modify: `pyimgano/reporting/report.py`
- Test: `tests/test_reporting_json.py`

**Steps:**
- Ensure report writer always JSON-serializes nested dicts and numpy types consistently.
- Commit: `git commit -m "fix: stabilize JSON report serialization"`

---

### Task 19: README updates for pixel-first + SoftPatch + anomalib aliases

**Files:**
- Modify: `README.md`

**Steps:**
- Add a short “Pixel-first industrial” example snippet using `pyimgano-benchmark --pixel`.
- Mention new `vision_softpatch` and anomalib alias expansion.
- Commit: `git commit -m "docs: update README for pixel-first industrial workflow"`

---

### Task 20: Final verification + merge to main

**Files:**
- N/A

**Steps:**
- Run: `pytest -q`
- Run: `python -m pyimgano.cli --help` (or `pyimgano-benchmark --help`)
- Merge branch to `main` and push.

