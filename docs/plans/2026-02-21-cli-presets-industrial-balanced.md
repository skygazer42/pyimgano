# CLI Presets (`--preset`) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `--preset industrial-balanced` to `pyimgano-benchmark` so users can run popular industrial AD models with sensible speed/accuracy-balanced defaults without hand-writing JSON kwargs.

**Architecture:** Extend `pyimgano/cli.py` with (1) a `--preset` flag, (2) a small preset resolver that returns model-specific default kwargs, and (3) a deterministic merge function where user kwargs override preset kwargs which override auto kwargs (device/contamination/pretrained). Optional FAISS support is detected via `optional_import` and never required.

**Tech Stack:** Python, `argparse`, `inspect`, `json`, NumPy; optional `faiss` behind `pyimgano[faiss]`.

---

### Task 1: Add `--preset` flag and a parser acceptance test

**Files:**
- Modify: `pyimgano/cli.py`
- Modify: `tests/test_cli_model_kwargs.py`

**Step 1: Write the failing test**

Append to `tests/test_cli_model_kwargs.py`:

```python
def test_cli_parser_accepts_preset_industrial_balanced():
    import pyimgano.cli as cli

    parser = cli._build_parser()
    try:
        parser.parse_args(
            [
                "--dataset",
                "mvtec",
                "--root",
                "/tmp",
                "--category",
                "bottle",
                "--preset",
                "industrial-balanced",
            ]
        )
    except SystemExit as exc:
        raise AssertionError(f"parser should accept --preset, got SystemExit({exc.code})") from exc
```

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_cli_model_kwargs.py::test_cli_parser_accepts_preset_industrial_balanced`
Expected: FAIL because `--preset` is an unknown argument.

**Step 3: Implement minimal CLI flag**

In `pyimgano/cli.py` add:
- `parser.add_argument("--preset", default=None, choices=["industrial-balanced"], help="...")`

**Step 4: Run test to verify it passes**

Run: `pytest -q tests/test_cli_model_kwargs.py::test_cli_parser_accepts_preset_industrial_balanced`
Expected: PASS.

**Step 5: Commit**

```bash
git add pyimgano/cli.py tests/test_cli_model_kwargs.py
git commit -m "feat: add pyimgano-benchmark preset flag"
```

---

### Task 2: Add preset resolver + tests (PatchCore defaults, FAISS fallback)

**Files:**
- Modify: `pyimgano/cli.py`
- Modify: `tests/test_cli_model_kwargs.py`

**Step 1: Write failing tests**

Append:

```python
def test_resolve_preset_kwargs_patchcore_prefers_sklearn_when_no_faiss(monkeypatch):
    import pyimgano.cli as cli

    monkeypatch.setattr(cli, "_faiss_available", lambda: False, raising=False)
    kwargs = cli._resolve_preset_kwargs("industrial-balanced", "vision_patchcore")
    assert kwargs["backbone"] == "resnet50"
    assert kwargs["knn_backend"] == "sklearn"


def test_resolve_preset_kwargs_patchcore_prefers_faiss_when_available(monkeypatch):
    import pyimgano.cli as cli

    monkeypatch.setattr(cli, "_faiss_available", lambda: True, raising=False)
    kwargs = cli._resolve_preset_kwargs("industrial-balanced", "vision_patchcore")
    assert kwargs["knn_backend"] == "faiss"
```

**Step 2: Run to verify it fails**

Run: `pytest -q tests/test_cli_model_kwargs.py::test_resolve_preset_kwargs_patchcore_prefers_sklearn_when_no_faiss`
Expected: FAIL because `_resolve_preset_kwargs` does not exist yet.

**Step 3: Implement preset resolver**

In `pyimgano/cli.py`:
- Add `_faiss_available()` using `optional_import("faiss")`
- Add `_resolve_preset_kwargs(preset, model_name)` returning the `industrial-balanced` kwargs

**Step 4: Run to verify it passes**

Run: `pytest -q tests/test_cli_model_kwargs.py::test_resolve_preset_kwargs_patchcore_prefers_faiss_when_available`
Expected: PASS.

**Step 5: Commit**

```bash
git add pyimgano/cli.py tests/test_cli_model_kwargs.py
git commit -m "feat: add industrial-balanced preset defaults"
```

---

### Task 3: Merge preset kwargs into model constructor kwargs (user overrides preset)

**Files:**
- Modify: `pyimgano/cli.py`
- Modify: `tests/test_cli_model_kwargs.py`

**Step 1: Write failing test**

Append:

```python
def test_build_model_kwargs_user_overrides_preset_values():
    from pyimgano.cli import _build_model_kwargs

    out = _build_model_kwargs(
        "vision_patchcore",
        user_kwargs={"coreset_sampling_ratio": 0.2},
        preset_kwargs={"coreset_sampling_ratio": 0.05, "n_neighbors": 5},
        auto_kwargs={"device": "cpu"},
    )
    assert out["coreset_sampling_ratio"] == 0.2
    assert out["n_neighbors"] == 5
    assert out["device"] == "cpu"
```

**Step 2: Run to verify it fails**

Run: `pytest -q tests/test_cli_model_kwargs.py::test_build_model_kwargs_user_overrides_preset_values`
Expected: FAIL because `_build_model_kwargs` does not accept `preset_kwargs` yet.

**Step 3: Implement merge logic**

Update `_build_model_kwargs` signature to accept `preset_kwargs`.

Implement merge order:
- start with filtered preset kwargs
- overlay user kwargs
- fill auto kwargs (only if missing) and only when accepted by the constructor

Wire it into `main()` so `--preset` influences `create_model(...)`.

**Step 4: Run to verify it passes**

Run: `pytest -q tests/test_cli_model_kwargs.py::test_build_model_kwargs_user_overrides_preset_values`
Expected: PASS.

**Step 5: Commit**

```bash
git add pyimgano/cli.py tests/test_cli_model_kwargs.py
git commit -m "feat: apply preset kwargs before auto kwargs"
```

---

### Task 4: End-to-end CLI smoke test (captures create_model kwargs)

**Files:**
- Modify: `tests/test_cli_model_kwargs.py`

**Step 1: Write failing test**

Append:

```python
def test_cli_applies_preset_for_patchcore(monkeypatch):
    import pyimgano.cli as cli

    captured: dict[str, object] = {}

    def fake_create_model(name: str, **kwargs):
        captured["name"] = name
        captured["kwargs"] = dict(kwargs)
        return object()

    monkeypatch.setattr(cli, "_faiss_available", lambda: False, raising=False)
    monkeypatch.setattr(cli, "load_benchmark_split", lambda *_a, **_k: object())
    monkeypatch.setattr(cli, "evaluate_split", lambda *_a, **_k: {"image_metrics": {"auroc": 0.0}})
    monkeypatch.setattr(cli, "create_model", fake_create_model)

    code = cli.main(
        [
            "--dataset",
            "mvtec",
            "--root",
            "/tmp",
            "--category",
            "bottle",
            "--model",
            "vision_patchcore",
            "--preset",
            "industrial-balanced",
            "--device",
            "cpu",
        ]
    )
    assert code == 0
    assert captured["name"] == "vision_patchcore"
    kwargs = captured["kwargs"]
    assert kwargs["backbone"] == "resnet50"
    assert kwargs["coreset_sampling_ratio"] == 0.05
    assert kwargs["knn_backend"] == "sklearn"
```

**Step 2: Run to verify it fails**

Run: `pytest -q tests/test_cli_model_kwargs.py::test_cli_applies_preset_for_patchcore`
Expected: FAIL until the preset is wired into `main()`.

**Step 3: Make it pass**

Ensure `main()` calls `_resolve_preset_kwargs(...)` and passes it into `_build_model_kwargs(...)`.

**Step 4: Run to verify it passes**

Run: `pytest -q tests/test_cli_model_kwargs.py::test_cli_applies_preset_for_patchcore`
Expected: PASS.

---

### Task 5: Docs update + full test run

**Files:**
- Modify: `README.md`
- Modify: `docs/QUICKSTART.md`

**Steps:**

1. Add a short section documenting `--preset industrial-balanced` + examples for PatchCore/FastFlow/CFlow.
2. Run tests: `pytest -q`
3. Commit + push:

```bash
git add README.md docs/QUICKSTART.md pyimgano/cli.py tests/test_cli_model_kwargs.py
git commit -m "docs: document pyimgano-benchmark industrial presets"
git push origin main
```

