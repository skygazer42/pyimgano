# Train CLI Thinning Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Thin `pyimgano.train_cli` by moving config-path execution, dry-run/preflight orchestration, and deploy bundle export into a reusable service module.

**Architecture:** Keep `pyimgano.train_cli` as an argparse and output adapter for recipe listing and config-driven execution modes. Introduce `pyimgano.services.train_service` to own config loading, workbench overrides, dry-run validation, preflight execution, recipe invocation, infer-config export, and deploy bundle packaging so that train flows are testable outside the CLI.

**Tech Stack:** Python, argparse, pytest, pathlib, json, shutil, existing workbench and recipe services.

---

### Task 1: Add a Train Service Seam

**Files:**
- Create: `pyimgano/services/train_service.py`
- Modify: `pyimgano/services/__init__.py`
- Modify: `pyimgano/train_cli.py`
- Test: `tests/test_train_service.py`
- Test: `tests/test_train_cli_dry_run.py`

**Step 1: Write the failing test**

Create `tests/test_train_service.py` with a focused dry-run payload test:

```python
from __future__ import annotations

import json

from pyimgano.services.train_service import TrainRunRequest, build_train_dry_run_payload


def test_build_train_dry_run_payload_returns_config_payload(tmp_path) -> None:
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(
        json.dumps(
            {
                "recipe": "industrial-adapt",
                "dataset": {"name": "custom", "root": "/tmp/data"},
                "model": {"name": "vision_patchcore"},
                "output": {"save_run": False},
            }
        ),
        encoding="utf-8",
    )

    payload = build_train_dry_run_payload(TrainRunRequest(config_path=str(cfg_path)))

    assert payload["config"]["recipe"] == "industrial-adapt"
```

Add one CLI delegation assertion in `tests/test_train_cli_dry_run.py`:

```python
def test_train_cli_dry_run_delegates_to_train_service(tmp_path, capsys, monkeypatch):
    from pyimgano.train_cli import main
    import pyimgano.services.train_service as train_service

    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text('{"recipe":"industrial-adapt","dataset":{"name":"custom","root":"/tmp/data"},"model":{"name":"vision_patchcore"},"output":{"save_run":false}}', encoding="utf-8")

    monkeypatch.setattr(
        train_service,
        "build_train_dry_run_payload",
        lambda _request: {"config": {"recipe": "delegated-train"}},
    )

    code = main(["--config", str(cfg_path), "--dry-run"])
    assert code == 0
    assert json.loads(capsys.readouterr().out)["config"]["recipe"] == "delegated-train"
```

**Step 2: Run test to verify it fails**

Run:

```bash
pytest --no-cov tests/test_train_service.py tests/test_train_cli_dry_run.py -k "train_service or delegates_to_train_service" -v
```

Expected: FAIL because `pyimgano.services.train_service` does not exist and `pyimgano.train_cli` still handles dry-run inline.

**Step 3: Write minimal implementation**

Create `pyimgano/services/train_service.py` with:

- `TrainRunRequest`
- `apply_train_overrides(...)`
- `load_train_config(...)`
- `build_train_dry_run_payload(...)`
- `run_train_preflight_payload(...)`
- `run_train_request(...)`

Keep deploy bundle helpers internal to the service. Migrate `pyimgano.train_cli` config-driven execution branches to call the service while keeping parser, recipe list/info formatting, JSON printing, and exit-code mapping in the CLI.

**Step 4: Run tests to verify they pass**

Run:

```bash
pytest --no-cov tests/test_train_service.py tests/test_train_cli_dry_run.py -v
```

Expected: PASS.

### Task 2: Verification Sweep

**Files:**
- Modify: none
- Test: `tests/test_train_service.py`
- Test: `tests/test_train_cli_dry_run.py`
- Test: `tests/test_train_cli_preflight.py`
- Test: `tests/test_train_cli_preprocessing_preset.py`
- Test: `tests/test_workbench_export_infer_config.py`

**Step 1: Run focused verification**

Run:

```bash
pytest --no-cov tests/test_train_service.py tests/test_train_cli_dry_run.py tests/test_train_cli_preflight.py tests/test_train_cli_preprocessing_preset.py tests/test_workbench_export_infer_config.py -v
```

Expected: PASS.
