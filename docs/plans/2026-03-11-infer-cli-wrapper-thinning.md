# Infer CLI Wrapper Thinning Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Thin the remaining wrapper-composition logic in `pyimgano.infer_cli` by moving tiling and preprocessing wrapper setup into `pyimgano.services.infer_setup_service`.

**Architecture:** Keep `pyimgano.infer_cli` responsible for CLI-only concerns such as argument parsing, input discovery, checkpoint/context loading, calibration, defects export, and output emission. Extend `pyimgano.services.infer_setup_service` so detector materialization and wrapper composition live together, with the service resolving tiling defaults, applying `TiledDetector` and `PreprocessingDetector` in the correct order, and preserving restored thresholds across wrappers.

**Tech Stack:** Python, dataclasses, pytest, existing inference wrappers, model capability metadata.

---

### Task 1: Move Wrapper Composition Into Infer Setup Service

**Files:**
- Modify: `pyimgano/services/infer_setup_service.py`
- Modify: `pyimgano/services/__init__.py`
- Modify: `pyimgano/infer_cli.py`
- Test: `tests/test_infer_setup_service.py`
- Test: `tests/test_infer_cli_infer_config.py`

**Step 1: Write the failing tests**

Add one service test proving wrapper setup applies infer-config tiling defaults and preserves `threshold_`, and one CLI test proving `--infer-config` delegates wrapper composition to `infer_setup_service`.

**Step 2: Run tests to verify they fail**

Run:

```bash
pytest --no-cov tests/test_infer_setup_service.py tests/test_infer_cli_infer_config.py -k "wrapper_setup or delegates_wrapper_setup" -v
```

Expected: FAIL because wrapper composition is still implemented inline in `pyimgano.infer_cli`.

**Step 3: Write minimal implementation**

Extend `pyimgano.services.infer_setup_service` with wrapper request/result types and a function that:

- resolves effective tiling options from CLI values plus config payload defaults
- wraps with `TiledDetector` when enabled
- validates preprocessing compatibility for numpy-only preprocessing
- wraps with `PreprocessingDetector` around the tiled detector
- reapplies the restored threshold after each wrapper so downstream code sees the correct `threshold_`

Then update `pyimgano.infer_cli` to call the service instead of constructing wrappers inline.

**Step 4: Run tests to verify they pass**

Run:

```bash
pytest --no-cov tests/test_infer_setup_service.py tests/test_infer_cli_infer_config.py -v
```

Expected: PASS.
