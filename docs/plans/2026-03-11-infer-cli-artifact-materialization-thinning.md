# Infer CLI Artifact Materialization Thinning Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Thin the remaining per-result artifact and defects-export logic in `pyimgano.infer_cli` by moving single-result materialization into a dedicated service.

**Architecture:** Keep `pyimgano.infer_cli` responsible for inference-loop control, batching, continue-on-error behavior, and writing JSONL lines. Introduce `pyimgano.services.infer_artifact_service` to own how one inference result becomes a record plus optional map/mask/overlay files and defects regions payload, including image-space bbox projection.

**Tech Stack:** Python, dataclasses, pytest, pathlib, numpy, existing inference/result/defects I/O helpers.

---

### Task 1: Add Infer Artifact Service

**Files:**
- Create: `pyimgano/services/infer_artifact_service.py`
- Modify: `pyimgano/services/__init__.py`
- Modify: `pyimgano/infer_cli.py`
- Test: `tests/test_infer_artifact_service.py`
- Test: `tests/test_infer_cli_smoke.py`

**Step 1: Write the failing tests**

Add one service test proving single-result materialization can:

- save anomaly maps and masks
- attach defects payload plus provenance
- emit regions payload
- project `bbox_xyxy_image` when requested
- save overlays

Add one CLI test proving result materialization delegates to `infer_artifact_service`.

**Step 2: Run tests to verify they fail**

Run:

```bash
pytest --no-cov tests/test_infer_artifact_service.py tests/test_infer_cli_smoke.py -k "artifact_service or delegates_artifact" -v
```

Expected: FAIL because per-result artifact materialization still lives inline in `pyimgano.infer_cli`.

**Step 3: Write minimal implementation**

Create `pyimgano.services.infer_artifact_service` with request/result dataclasses and a `materialize_infer_result_artifacts(...)` function that:

- serializes the inference result into a record
- saves anomaly maps when requested
- extracts defects and saves masks when enabled
- emits a regions payload for `--defects-regions-jsonl`
- saves overlays when requested

Then update `pyimgano.infer_cli` so `_process_ok_result` becomes a thin call into the service.

**Step 4: Run tests to verify they pass**

Run:

```bash
pytest --no-cov tests/test_infer_artifact_service.py tests/test_infer_cli_smoke.py -v
```

Expected: PASS.
