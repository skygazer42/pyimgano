# Doctor CLI Thinning Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Thin `pyimgano.doctor_cli` by moving doctor payload assembly and environment checks into a reusable service module.

**Architecture:** Keep `pyimgano.doctor_cli` as an argparse and text/JSON formatting adapter. Introduce `pyimgano.services.doctor_service` to own optional-module checks, suite checks, accelerator checks, required-extras checks, and the final JSON-ready payload assembly so the behavior is testable outside the CLI.

**Tech Stack:** Python, argparse, pytest, importlib.metadata, optional dependency helpers, existing baseline/discovery utilities.

---

### Task 1: Add a Doctor Service Seam

**Files:**
- Create: `pyimgano/services/doctor_service.py`
- Modify: `pyimgano/services/__init__.py`
- Modify: `pyimgano/doctor_cli.py`
- Test: `tests/test_doctor_service.py`
- Test: `tests/test_doctor_cli.py`

**Step 1: Write the failing test**

Create `tests/test_doctor_service.py` with a focused payload test:

```python
from __future__ import annotations

from pyimgano.services.doctor_service import collect_doctor_payload


def test_collect_doctor_payload_returns_json_ready_shape() -> None:
    payload = collect_doctor_payload()
    assert payload["tool"] == "pyimgano-doctor"
    assert "optional_modules" in payload
    assert "baselines" in payload
```

Add one CLI delegation assertion in `tests/test_doctor_cli.py`:

```python
def test_doctor_cli_json_delegates_to_doctor_service(monkeypatch, capsys) -> None:
    from pyimgano.doctor_cli import main
    import pyimgano.services.doctor_service as doctor_service

    monkeypatch.setattr(
        doctor_service,
        "collect_doctor_payload",
        lambda **_kwargs: {"tool": "delegated-doctor", "python": {}, "platform": {}, "optional_modules": [], "baselines": {}},
    )

    rc = main(["--json"])
    assert rc == 0
    assert "delegated-doctor" in capsys.readouterr().out
```

**Step 2: Run test to verify it fails**

Run:

```bash
pytest --no-cov tests/test_doctor_service.py tests/test_doctor_cli.py -k "doctor_service or delegates_to_doctor_service" -v
```

Expected: FAIL because `pyimgano.services.doctor_service` does not exist and `pyimgano.doctor_cli` still builds payloads inline.

**Step 3: Write minimal implementation**

Create `pyimgano/services/doctor_service.py` with:

- `collect_doctor_payload(...)`
- `build_suite_checks(...)`
- `build_require_extras_check(...)`
- `build_accelerator_checks()`
- `check_module(...)`

Return plain dict/list payloads only. Reuse existing helpers from `pyimgano.utils.extras`, `pyimgano.utils.optional_deps`, baseline listing, and preset listing. Then migrate `pyimgano.doctor_cli` to call the service for payload assembly while keeping text rendering and exit-code handling in the CLI.

**Step 4: Run tests to verify they pass**

Run:

```bash
pytest --no-cov tests/test_doctor_service.py tests/test_doctor_cli.py -v
```

Expected: PASS.

### Task 2: Verification Sweep

**Files:**
- Modify: none
- Test: `tests/test_doctor_service.py`
- Test: `tests/test_doctor_cli.py`

**Step 1: Run focused verification**

Run:

```bash
pytest --no-cov tests/test_doctor_service.py tests/test_doctor_cli.py -v
```

Expected: PASS.
