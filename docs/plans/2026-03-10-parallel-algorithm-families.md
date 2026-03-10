# Parallel Algorithm Families Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add eight new anomaly-detection algorithm families to `pyimgano` in parallel using isolated worktrees, production-safe adapters, deterministic tests, and a single integration branch.

**Architecture:** Build the shared registry/discovery foundation once on the integration branch, then implement each family in its own feature branch/worktree with mostly disjoint file ownership. Merge all tracks into `feat/algo-families-2026q1`, resolve shared integration points there, and run focused verification before merging back to `main`.

**Tech Stack:** Python, NumPy, PyTorch, optional CLIP/OpenCLIP backends, optional diffusion/VLM dependencies, pytest, setuptools extras, git worktrees.

---

### Task 1: Reserve Shared Module Slots Before Parallel Branching

**Files:**
- Modify: `pyimgano/models/__init__.py`
- Create: `pyimgano/models/visionad.py`
- Create: `pyimgano/models/univad.py`
- Create: `pyimgano/models/filopp.py`
- Create: `pyimgano/models/adaclip.py`
- Create: `pyimgano/models/aaclip.py`
- Create: `pyimgano/models/one_to_normal.py`
- Create: `pyimgano/models/logsad.py`
- Create: `pyimgano/models/anogen.py`
- Test: `tests/test_parallel_algorithm_registry_smoke.py`

**Step 1: Write the failing test**

Create `tests/test_parallel_algorithm_registry_smoke.py`:

```python
import pyimgano.models as models


def test_parallel_algorithm_family_placeholders_register():
    available = set(models.list_models())
    expected = {
        "vision_visionad",
        "vision_univad",
        "vision_filopp",
        "vision_adaclip",
        "vision_aaclip",
        "vision_one_to_normal",
        "vision_logsad",
        "vision_anogen_adapter",
    }
    assert expected.issubset(available)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_parallel_algorithm_registry_smoke.py -v`

Expected: FAIL because the modules and registry entries do not exist yet.

**Step 3: Write minimal implementation**

- Add the eight module names to `_MODEL_MODULE_ALLOWLIST` in `pyimgano/models/__init__.py`
- Create each module with:
  - one lightweight registered class
  - minimal metadata with `description`, `paper`, and `year`
  - a constructor that either accepts injected components or raises a clean `NotImplementedError` placeholder

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_parallel_algorithm_registry_smoke.py -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add pyimgano/models/__init__.py pyimgano/models/visionad.py pyimgano/models/univad.py pyimgano/models/filopp.py pyimgano/models/adaclip.py pyimgano/models/aaclip.py pyimgano/models/one_to_normal.py pyimgano/models/logsad.py pyimgano/models/anogen.py tests/test_parallel_algorithm_registry_smoke.py
git commit -m "feat: reserve parallel algorithm family module slots"
```

---

### Task 2: Create Eight Feature Branches and Worktrees

**Files:**
- Modify: none
- Test: none

**Step 1: Create worktrees**

Run:

```bash
git worktree add .worktrees/visionad-search-fsad -b feat/visionad-search-fsad
git worktree add .worktrees/univad-unified-fsad -b feat/univad-unified-fsad
git worktree add .worktrees/filopp-vlm-localization -b feat/filopp-vlm-localization
git worktree add .worktrees/adaclip-hybrid-prompts -b feat/adaclip-hybrid-prompts
git worktree add .worktrees/aaclip-anomaly-aware -b feat/aaclip-anomaly-aware
git worktree add .worktrees/one-to-normal-personalization -b feat/one-to-normal-personalization
git worktree add .worktrees/logsad-logical-structural -b feat/logsad-logical-structural
git worktree add .worktrees/anogen-fewshot-generation -b feat/anogen-fewshot-generation
```

**Step 2: Verify each worktree starts clean**

Run in each worktree:

```bash
pytest tests/test_parallel_algorithm_registry_smoke.py -v
```

Expected: PASS in all worktrees.

**Step 3: Commit**

No commit; this is branch/worktree setup only.

---

### Task 3: Implement VisionAD Family

**Files:**
- Modify: `pyimgano/models/visionad.py`
- Test: `tests/test_visionad.py`

**Step 1: Write the failing test**

Create `tests/test_visionad.py`:

```python
import numpy as np
import pyimgano.models as models


class FakePatchSearchBackend:
    def fit(self, train_patches):
        self.train_centroid = np.mean(np.concatenate(train_patches, axis=0), axis=0)
        return self

    def score(self, patch_grid):
        delta = patch_grid - self.train_centroid[None, :]
        patch_scores = np.linalg.norm(delta, axis=1)
        return float(np.max(patch_scores)), patch_scores


def test_visionad_scores_anomaly_higher_than_normal():
    detector = models.create_model(
        "vision_visionad",
        search_backend=FakePatchSearchBackend(),
        embedder=lambda image: image,
    )
    train = [np.array([[0.0, 0.0], [0.1, 0.0]], dtype=np.float32)]
    test = [
        np.array([[0.0, 0.0], [0.1, 0.1]], dtype=np.float32),
        np.array([[5.0, 5.0], [5.1, 5.1]], dtype=np.float32),
    ]
    detector.fit(train)
    scores = detector.decision_function(test)
    assert scores[1] > scores[0]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_visionad.py -v`

Expected: FAIL.

**Step 3: Write minimal implementation**

Implement in `pyimgano/models/visionad.py`:

- injectable `embedder`
- injectable `search_backend`
- training-free patch-memory scoring path
- optional `predict_anomaly_map()` using patch-score upsampling when patch-grid metadata is available

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_visionad.py -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add pyimgano/models/visionad.py tests/test_visionad.py
git commit -m "feat: add visionad family adapter"
```

---

### Task 4: Implement UniVAD Family

**Files:**
- Modify: `pyimgano/models/univad.py`
- Test: `tests/test_univad.py`

**Step 1: Write the failing test**

Create a deterministic few-shot test that:

- fits on a small support set
- accepts injected feature extractor and multi-layer aggregator
- scores a shifted test sample higher than a nominal sample

Use exact file: `tests/test_univad.py`

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_univad.py -v`

Expected: FAIL.

**Step 3: Write minimal implementation**

Implement in `pyimgano/models/univad.py`:

- support-set `fit`
- injected `feature_extractor`
- layer-weighted feature fusion
- nearest-neighbor or prototype-based scoring for the first milestone

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_univad.py -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add pyimgano/models/univad.py tests/test_univad.py
git commit -m "feat: add univad family adapter"
```

---

### Task 5: Implement FiLo++ Family

**Files:**
- Modify: `pyimgano/models/filopp.py`
- Test: `tests/test_filopp.py`

**Step 1: Write the failing test**

Create `tests/test_filopp.py` with:

- injected image encoder
- injected prompt bank
- injected localization head
- assertion that anomaly prompts raise score and localization map has the expected shape

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_filopp.py -v`

Expected: FAIL.

**Step 3: Write minimal implementation**

Implement in `pyimgano/models/filopp.py`:

- prompt filtering hook
- fused text-description scoring
- optional deformable-localization adapter path
- offline-safe fallback when heavyweight VLM deps are unavailable

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_filopp.py -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add pyimgano/models/filopp.py tests/test_filopp.py
git commit -m "feat: add filopp family adapter"
```

---

### Task 6: Implement AdaCLIP Family

**Files:**
- Modify: `pyimgano/models/adaclip.py`
- Test: `tests/test_adaclip.py`

**Step 1: Write the failing test**

Create `tests/test_adaclip.py` with injected CLIP-like text/image features and assert:

- hybrid prompt fusion changes the score
- anomalous sample ranks above the nominal sample

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_adaclip.py -v`

Expected: FAIL.

**Step 3: Write minimal implementation**

Implement in `pyimgano/models/adaclip.py`:

- static + dynamic prompt fusion
- injected encoder hooks for offline tests
- clean `ImportError` path for missing optional CLIP backend

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_adaclip.py -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add pyimgano/models/adaclip.py tests/test_adaclip.py
git commit -m "feat: add adaclip family adapter"
```

---

### Task 7: Implement AA-CLIP Family

**Files:**
- Modify: `pyimgano/models/aaclip.py`
- Test: `tests/test_aaclip.py`

**Step 1: Write the failing test**

Create `tests/test_aaclip.py` asserting:

- anomaly-aware text anchor injection is accepted
- patch-level anomaly scores are returned
- anomalous sample ranks above the nominal sample

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_aaclip.py -v`

Expected: FAIL.

**Step 3: Write minimal implementation**

Implement in `pyimgano/models/aaclip.py`:

- anomaly-aware prompt/text-anchor path
- patchwise score aggregation
- optional `predict_anomaly_map()`

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_aaclip.py -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add pyimgano/models/aaclip.py tests/test_aaclip.py
git commit -m "feat: add aaclip family adapter"
```

---

### Task 8: Implement One-to-Normal Family

**Files:**
- Modify: `pyimgano/models/one_to_normal.py`
- Test: `tests/test_one_to_normal.py`

**Step 1: Write the failing test**

Create `tests/test_one_to_normal.py` with:

- injected `normalizer` callable that transforms anomalous inputs toward a normal manifold
- anomaly score defined from original-vs-normalized residual
- assertion that abnormal inputs have higher residual scores

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_one_to_normal.py -v`

Expected: FAIL.

**Step 3: Write minimal implementation**

Implement in `pyimgano/models/one_to_normal.py`:

- injected one-to-normal generator
- residual-based image score
- optional residual map output for localization

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_one_to_normal.py -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add pyimgano/models/one_to_normal.py tests/test_one_to_normal.py
git commit -m "feat: add one-to-normal family adapter"
```

---

### Task 9: Implement LogSAD Family

**Files:**
- Modify: `pyimgano/models/logsad.py`
- Test: `tests/test_logsad.py`

**Step 1: Write the failing test**

Create `tests/test_logsad.py` that uses injected:

- patch-token detector
- logical-rule matcher
- score calibrator

Assert that:

- structural anomaly increases image score
- logical-rule violation increases image score
- combined scoring remains deterministic

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_logsad.py -v`

Expected: FAIL.

**Step 3: Write minimal implementation**

Implement in `pyimgano/models/logsad.py`:

- multi-granularity score fusion
- injected logic engine
- calibration hook
- offline-safe implementation that does not require external MLLM at test time

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_logsad.py -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add pyimgano/models/logsad.py tests/test_logsad.py
git commit -m "feat: add logsad family adapter"
```

---

### Task 10: Implement AnoGen Family Adapter

**Files:**
- Modify: `pyimgano/models/anogen.py`
- Test: `tests/test_anogen.py`
- Optionally modify: `pyimgano/synthesis/__init__.py`

**Step 1: Write the failing test**

Create `tests/test_anogen.py` that:

- injects a synthetic anomaly generator
- produces generated anomalous variants from a normal image batch
- verifies that the adapter can emit training pairs or score outputs deterministically

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_anogen.py -v`

Expected: FAIL.

**Step 3: Write minimal implementation**

Implement in `pyimgano/models/anogen.py`:

- adapter around anomaly-driven generation
- integration hook into existing synthesis-oriented code
- deterministic fake-generator path for tests

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_anogen.py -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add pyimgano/models/anogen.py tests/test_anogen.py
git commit -m "feat: add anogen family adapter"
```

---

### Task 11: Integrate Shared Discovery, CLI, and Documentation

**Files:**
- Modify: `pyimgano/discovery.py`
- Modify: `pyimgano/cli.py`
- Modify: `pyimgano/infer_cli.py`
- Modify: `docs/MODEL_INDEX.md`
- Modify: `docs/SOTA_ALGORITHMS.md`
- Modify: `README.md`
- Test: `tests/test_cli_discovery.py`
- Test: `tests/test_pyim_cli.py`
- Test: `tests/test_infer_cli_discovery_and_model_presets_v16.py`

**Step 1: Write failing tests**

Extend discovery tests to assert the eight new models:

- appear in `--list-models`
- carry appropriate tags/family metadata
- surface through `pyim` and `pyimgano-infer --list-models`

**Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_cli_discovery.py tests/test_pyim_cli.py tests/test_infer_cli_discovery_and_model_presets_v16.py -v
```

Expected: FAIL.

**Step 3: Write minimal implementation**

- update discovery metadata if new family/type tags are needed
- ensure CLI and infer discovery output remain stable
- document status as adapter/native/experimental where appropriate

**Step 4: Run tests to verify they pass**

Run:

```bash
pytest tests/test_cli_discovery.py tests/test_pyim_cli.py tests/test_infer_cli_discovery_and_model_presets_v16.py -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add pyimgano/discovery.py pyimgano/cli.py pyimgano/infer_cli.py docs/MODEL_INDEX.md docs/SOTA_ALGORITHMS.md README.md tests/test_cli_discovery.py tests/test_pyim_cli.py tests/test_infer_cli_discovery_and_model_presets_v16.py
git commit -m "feat: integrate parallel algorithm families into discovery and docs"
```

---

### Task 12: Final Integration Verification

**Files:**
- Modify: none unless fixes are required
- Test: `tests/test_parallel_algorithm_registry_smoke.py`
- Test: `tests/test_visionad.py`
- Test: `tests/test_univad.py`
- Test: `tests/test_filopp.py`
- Test: `tests/test_adaclip.py`
- Test: `tests/test_aaclip.py`
- Test: `tests/test_one_to_normal.py`
- Test: `tests/test_logsad.py`
- Test: `tests/test_anogen.py`

**Step 1: Run focused verification**

Run:

```bash
pytest tests/test_parallel_algorithm_registry_smoke.py tests/test_visionad.py tests/test_univad.py tests/test_filopp.py tests/test_adaclip.py tests/test_aaclip.py tests/test_one_to_normal.py tests/test_logsad.py tests/test_anogen.py -v
```

Expected: PASS.

**Step 2: Run existing discovery smoke**

Run:

```bash
pytest tests/test_cli_smoke.py tests/test_cli_discovery.py tests/test_pyim_cli.py -v
```

Expected: PASS.

**Step 3: Merge all feature branches into the integration branch**

Run:

```bash
git checkout feat/algo-families-2026q1
git merge --no-ff feat/visionad-search-fsad
git merge --no-ff feat/univad-unified-fsad
git merge --no-ff feat/filopp-vlm-localization
git merge --no-ff feat/adaclip-hybrid-prompts
git merge --no-ff feat/aaclip-anomaly-aware
git merge --no-ff feat/one-to-normal-personalization
git merge --no-ff feat/logsad-logical-structural
git merge --no-ff feat/anogen-fewshot-generation
```

**Step 4: Re-run verification after merges**

Run the two pytest commands from Step 1 and Step 2 again.

Expected: PASS.

**Step 5: Commit merge-resolution fixes if needed**

```bash
git add <resolved-files>
git commit -m "merge: integrate parallel algorithm family branches"
```
