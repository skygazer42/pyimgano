# Infer CLI Runtime Planning Thinning Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Thin the remaining pre-inference planning logic in `pyimgano.infer_cli` by moving anomaly-map postprocess selection and defects pixel-threshold resolution into a dedicated runtime-planning service.

**Architecture:** Keep `pyimgano.infer_cli` responsible for CLI-only argument parsing, input discovery, detector/context setup, calibration, and output side effects. Introduce `pyimgano.services.infer_runtime_service` to own effective `include_maps` resolution, infer-config postprocess payload materialization, and defects pixel-threshold planning, including optional calibration-map collection from training inputs.

**Tech Stack:** Python, dataclasses, pytest, existing defects calibration helpers, workbench postprocess config, inference service.

---

### Task 1: Add Runtime Planning Service

**Files:**
- Create: `pyimgano/services/infer_runtime_service.py`
- Modify: `pyimgano/services/__init__.py`
- Modify: `pyimgano/infer_cli.py`
- Test: `tests/test_infer_runtime_service.py`
- Test: `tests/test_infer_cli_infer_config.py`

**Step 1: Write the failing tests**

Add a focused service test proving runtime planning can:

- imply `include_maps` when `defects` is enabled
- materialize an infer-config postprocess payload
- recover an infer-config pixel threshold and provenance

Add one CLI test proving the `--infer-config` path delegates runtime planning to `infer_runtime_service`.

**Step 2: Run tests to verify they fail**

Run:

```bash
pytest --no-cov tests/test_infer_runtime_service.py tests/test_infer_cli_infer_config.py -k "runtime_plan or delegates_runtime_plan" -v
```

Expected: FAIL because runtime planning still lives inline in `pyimgano.infer_cli`.

**Step 3: Write minimal implementation**

Create `pyimgano.services.infer_runtime_service` with request/result types and a `prepare_infer_runtime_plan(...)` function that:

- resolves effective `include_maps`
- builds anomaly-map postprocess from CLI or infer-config payload
- resolves pixel threshold and provenance for defects mode
- optionally runs calibration inference over training inputs when normal-pixel quantile calibration is needed

Then update `pyimgano.infer_cli` to use the service and remove the now-inline planning code.

**Step 4: Run tests to verify they pass**

Run:

```bash
pytest --no-cov tests/test_infer_runtime_service.py tests/test_infer_cli_infer_config.py -v
```

Expected: PASS.
