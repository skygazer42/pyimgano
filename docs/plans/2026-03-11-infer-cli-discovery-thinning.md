# Infer CLI Discovery Thinning Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Thin `pyimgano.infer_cli` by moving discovery and preset payload assembly into `pyimgano.services.discovery_service`.

**Architecture:** Keep `pyimgano.infer_cli` as the argparse and output-format adapter for model discovery branches. Extend `pyimgano.services.discovery_service` so model listing, model info, preset listing, and preset info become reusable service calls shared across CLIs.

**Tech Stack:** Python, argparse, pytest, inspect, existing discovery/model preset utilities.

---

### Task 1: Add Preset Discovery Service APIs

**Files:**
- Modify: `pyimgano/services/discovery_service.py`
- Modify: `pyimgano/services/__init__.py`
- Test: `tests/test_discovery_service.py`
- Test: `tests/test_infer_cli_discovery_and_model_presets_v16.py`

**Step 1: Write the failing test**

Extend `tests/test_discovery_service.py` with:

```python
def test_build_model_preset_info_payload_returns_json_ready_shape() -> None:
    payload = build_model_preset_info_payload("industrial-structural-ecod")
    assert payload["name"] == "industrial-structural-ecod"
    assert "model" in payload
    assert "kwargs" in payload
```

Add one CLI delegation assertion in `tests/test_infer_cli_discovery_and_model_presets_v16.py`:

```python
def test_infer_cli_list_models_delegates_to_discovery_service(monkeypatch, capsys) -> None:
    from pyimgano.infer_cli import main as infer_main
    import pyimgano.services.discovery_service as discovery_service

    monkeypatch.setattr(
        discovery_service,
        "list_discovery_model_names",
        lambda **_kwargs: ["delegated-model"],
    )

    rc = infer_main(["--list-models"])
    assert rc == 0
    assert capsys.readouterr().out.strip().splitlines() == ["delegated-model"]
```

**Step 2: Run test to verify it fails**

Run:

```bash
pytest --no-cov tests/test_discovery_service.py tests/test_infer_cli_discovery_and_model_presets_v16.py -k "model_preset_info_payload or delegates_to_discovery_service" -v
```

Expected: FAIL because preset discovery APIs are not exposed by `discovery_service` and `infer_cli` still reads discovery utilities directly.

**Step 3: Write minimal implementation**

Extend `pyimgano.services.discovery_service` with:

- `list_model_preset_names(...)`
- `list_model_preset_infos_payload(...)`
- `build_model_preset_info_payload(...)`

Then migrate `pyimgano.infer_cli` discovery branches to call `discovery_service` for:

- `--list-models`
- `--model-info`
- `--list-model-presets`
- `--model-preset-info`

Keep parser validation and text/JSON formatting in the CLI.

**Step 4: Run tests to verify they pass**

Run:

```bash
pytest --no-cov tests/test_discovery_service.py tests/test_infer_cli_discovery_and_model_presets_v16.py -v
```

Expected: PASS.
