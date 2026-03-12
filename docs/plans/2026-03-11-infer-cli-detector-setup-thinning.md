# Infer CLI Detector Setup Thinning Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Thin the repeated config-backed detector construction logic in `pyimgano.infer_cli` by moving `ConfigBackedInferContext -> detector` setup into `pyimgano.services.infer_setup_service`.

**Architecture:** Keep `pyimgano.infer_cli` responsible for CLI-only concerns such as argument parsing, input discovery, ONNX session-option sweeps, and output emission. Extend `pyimgano.services.infer_setup_service` so direct and config-backed inference both use service-owned detector materialization instead of duplicating model option resolution, checkpoint requirement enforcement, checkpoint loading, and threshold restoration in the CLI.

**Tech Stack:** Python, dataclasses, pytest, existing model registry and workbench checkpoint utilities.

---

### Task 1: Add Config-Backed Detector Setup Service

**Files:**
- Modify: `pyimgano/services/infer_setup_service.py`
- Modify: `pyimgano/services/__init__.py`
- Modify: `pyimgano/infer_cli.py`
- Test: `tests/test_infer_setup_service.py`
- Test: `tests/test_infer_cli_infer_config.py`

**Step 1: Write the failing tests**

Add a focused service test that proves config-backed setup restores checkpoints and thresholds, and a CLI test that proves the `--infer-config` branch delegates detector materialization to `infer_setup_service`.

**Step 2: Run tests to verify they fail**

Run:

```bash
pytest --no-cov tests/test_infer_setup_service.py tests/test_infer_cli_infer_config.py -k "config_backed or detector_setup_to_service" -v
```

Expected: FAIL because config-backed detector setup is still implemented inline in `pyimgano.infer_cli`.

**Step 3: Write minimal implementation**

Extend `pyimgano.services.infer_setup_service` with config-backed request/result types and a loader function that:

- accepts a prepared `ConfigBackedInferContext`
- resolves model options
- enforces checkpoint requirements using either `checkpoint_path` or restored trained checkpoint
- creates the detector through an injectable factory
- restores trained checkpoints and `threshold_`

Then update `pyimgano.infer_cli` so `--from-run` and `--infer-config` call the service after ONNX/session-option preparation instead of duplicating setup inline.

**Step 4: Run tests to verify they pass**

Run:

```bash
pytest --no-cov tests/test_infer_setup_service.py tests/test_infer_cli_infer_config.py -v
```

Expected: PASS.
