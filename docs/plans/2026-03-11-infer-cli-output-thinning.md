# Infer CLI Output Thinning Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extract `infer_cli` output target setup, JSONL writing, and error record creation into a dedicated service without rewriting the outer inference loop.

**Architecture:** Add a narrow `infer_output_service` that owns output file opening rules, JSONL write coordination, and error record formatting. Keep `infer_cli` responsible for loop control, counters, and exit-code behavior while delegating output-specific behavior through explicit request/result dataclasses.

**Tech Stack:** Python 3.10, pytest, dataclasses, pathlib, json

---

### Task 1: Add output service tests

**Files:**
- Create: `tests/test_infer_output_service.py`
- Modify: `tests/test_infer_cli_smoke.py`

**Step 1: Write the failing tests**

```python
def test_open_infer_output_targets_requires_defects_for_regions_jsonl():
    ...

def test_write_infer_output_payloads_writes_record_and_regions_and_flushes():
    ...

def test_infer_cli_smoke_delegates_output_writing_to_service():
    ...
```

**Step 2: Run test to verify it fails**

Run: `pytest --no-cov tests/test_infer_output_service.py tests/test_infer_cli_smoke.py -q`
Expected: FAIL because `pyimgano.services.infer_output_service` does not exist yet.

### Task 2: Implement output service and wire CLI

**Files:**
- Create: `pyimgano/services/infer_output_service.py`
- Modify: `pyimgano/services/__init__.py`
- Modify: `pyimgano/infer_cli.py`

**Step 1: Write minimal implementation**

```python
@dataclass(frozen=True)
class InferOutputTargetsRequest: ...

def open_infer_output_targets(...): ...
def write_infer_output_payloads(...): ...
def build_infer_error_record(...): ...
```

**Step 2: Refactor CLI to delegate**

```python
targets = infer_output_service.open_infer_output_targets(...)
write_result = infer_output_service.write_infer_output_payloads(...)
error_record = infer_output_service.build_infer_error_record(...)
```

**Step 3: Run tests to verify they pass**

Run: `pytest --no-cov tests/test_infer_output_service.py tests/test_infer_cli_smoke.py -q`
Expected: PASS

### Task 3: Run focused infer regression coverage

**Files:**
- Test: `tests/test_infer_cli_production_guardrails_v1.py`
- Test: `tests/test_infer_cli_defects_regions_jsonl.py`
- Test: `tests/test_infer_cli_infer_config.py`

**Step 1: Run focused regression suite**

Run: `pytest --no-cov tests/test_infer_output_service.py tests/test_infer_cli_smoke.py tests/test_infer_cli_production_guardrails_v1.py tests/test_infer_cli_defects_regions_jsonl.py tests/test_infer_cli_infer_config.py -q`
Expected: PASS

**Step 2: Check whitespace / patch hygiene**

Run: `git diff --check -- pyimgano/infer_cli.py pyimgano/services/__init__.py pyimgano/services/infer_output_service.py tests/test_infer_output_service.py tests/test_infer_cli_smoke.py docs/plans/2026-03-11-infer-cli-output-thinning.md`
Expected: no output
