# Workbench Training Runtime Boundary Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce `pyimgano.workbench.runner` training coupling by moving detector fitting, micro-finetune invocation, and checkpoint persistence into a dedicated workbench runtime helper.

**Architecture:** Add a `pyimgano.workbench.training_runtime` module that receives the detector, config, train inputs, category, and optional run directory, then returns normalized training metadata. Keep threshold calibration and inference in `runner.py`; only the training/checkpoint branch moves out.

**Tech Stack:** Python, dataclasses, pytest, existing `pyimgano.training.runner`, checkpoint helpers, `WorkbenchConfig`.

---

### Task 1: Lock The Training Boundary With Failing Tests

**Files:**
- Create: `tests/test_workbench_training_runtime.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Write the failing test**

Add tests that prove:
- the helper runs micro-finetune and writes checkpoint metadata when training is enabled
- the helper falls back to `detector.fit(...)` when training is disabled
- `runner.py` no longer references `micro_finetune` or `save_checkpoint(...)` directly

**Step 2: Run test to verify it fails**

Run:

```bash
pytest --no-cov tests/test_workbench_training_runtime.py tests/test_architecture_boundaries.py -k "workbench_training_runtime or runner_uses_training_runtime_boundary" -v
```

Expected: FAIL because the helper module does not exist and `runner.py` still owns the training branch.

### Task 2: Add The Training Runtime Helper And Refactor Runner

**Files:**
- Create: `pyimgano/workbench/training_runtime.py`
- Modify: `pyimgano/workbench/runner.py`

**Step 1: Write minimal implementation**

- add a small dataclass describing normalized training artifacts
- add `run_workbench_training(...)`
- move fit-kwargs assembly, micro-finetune invocation, and checkpoint path bookkeeping into the helper
- keep threshold calibration and downstream reporting unchanged

**Step 2: Run focused tests to verify they pass**

Run:

```bash
pytest --no-cov tests/test_workbench_training_runtime.py tests/test_workbench_runner_checkpoints.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

### Task 3: Regression Sweep

**Files:**
- Modify: `pyimgano/workbench/training_runtime.py`
- Modify: `pyimgano/workbench/runner.py`
- Modify: `tests/test_workbench_training_runtime.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Run regression suite**

Run:

```bash
pytest --no-cov tests/test_workbench_training_runtime.py tests/test_workbench_runner_checkpoints.py tests/test_integration_workbench_train_then_infer.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

**Step 2: Check patch hygiene**

Run:

```bash
git diff --check -- docs/plans/2026-03-13-workbench-training-runtime-boundary.md pyimgano/workbench/training_runtime.py pyimgano/workbench/runner.py tests/test_workbench_training_runtime.py tests/test_architecture_boundaries.py
```

Expected: no output
