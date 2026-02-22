# CLI Presets (`industrial-fast` / `industrial-accurate`) + Model Discovery Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add two new industrial presets and add `--list-models` / `--model-info` discovery modes to `pyimgano-benchmark`.

**Architecture:** Keep the existing single-command CLI shape. Make dataset args optional at parse time, but required at runtime unless a discovery mode is selected. Presets are resolved by name + model, then merged with user kwargs and auto kwargs with stable precedence.

**Tech Stack:** Python, `argparse`, `inspect`, `json`, pytest.

---

### Task 1: Add failing tests for new preset names

**Files:**
- Modify: `tests/test_cli_model_kwargs.py`

**Step 1: Write the failing test**

Add tests asserting the parser accepts:
- `--preset industrial-fast`
- `--preset industrial-accurate`

**Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest -q tests/test_cli_model_kwargs.py -k preset`
Expected: FAIL until parser choices are expanded.

**Step 3: Implement minimal code**

Update CLI parser `--preset` choices.

**Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest -q tests/test_cli_model_kwargs.py -k preset`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/test_cli_model_kwargs.py pyimgano/cli.py
git commit -m "test: cover new preset names"
```

---

### Task 2: Add failing tests for `industrial-fast` / `industrial-accurate` preset kwargs

**Files:**
- Modify: `tests/test_cli_model_kwargs.py`

**Step 1: Write the failing tests**

Add tests for representative models, e.g.:
- `vision_patchcore` (verify backbone/coreset/k/backend)
- `vision_anomalydino` (verify image_size/coreset/backend)
- `vision_reverse_dist` alias matches `vision_reverse_distillation`

Use `monkeypatch` to control FAISS availability.

**Step 2: Run to verify it fails**

Run: `.venv/bin/pytest -q tests/test_cli_model_kwargs.py`
Expected: FAIL until preset logic is implemented.

**Step 3: Implement minimal preset logic**

Implement in `pyimgano/cli.py:_resolve_preset_kwargs`.

**Step 4: Run to verify it passes**

Run: `.venv/bin/pytest -q tests/test_cli_model_kwargs.py`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/test_cli_model_kwargs.py pyimgano/cli.py
git commit -m "feat: add industrial-fast/accurate preset defaults"
```

---

### Task 3: Add failing tests for CLI discovery flags

**Files:**
- Create: `tests/test_cli_discovery.py`

**Step 1: Write the failing tests**

Tests:
- `main(["--list-models"])` returns `0` and prints `vision_patchcore` in stdout.
- `main(["--list-models","--json"])` returns `0` and prints valid JSON array.
- `main(["--model-info","vision_patchcore"])` returns `0` and prints “Signature”.
- `main(["--model-info","vision_patchcore","--json"])` returns `0` and prints JSON with `"name": "vision_patchcore"`.
- `main(["--list-models","--model-info","vision_patchcore"])` returns non-zero.

**Step 2: Run to verify it fails**

Run: `.venv/bin/pytest -q tests/test_cli_discovery.py`
Expected: FAIL until CLI flags exist and dataset args are optional at parse time.

**Step 3: Implement minimal CLI flags**

Update `pyimgano/cli.py`:
- Add parser args.
- Early-exit behavior in `main()`.
- Runtime validation for required dataset args in run mode.

**Step 4: Run to verify it passes**

Run: `.venv/bin/pytest -q tests/test_cli_discovery.py`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/test_cli_discovery.py pyimgano/cli.py
git commit -m "feat: add model discovery flags to CLI"
```

---

### Task 4: Update docs

**Files:**
- Modify: `README.md`
- Modify: `docs/QUICKSTART.md`

**Steps:**
- Document new preset names and their intent.
- Add usage snippets for `--list-models` and `--model-info` with `--json` mention.

**Commit**

```bash
git add README.md docs/QUICKSTART.md
git commit -m "docs: document new presets and discovery flags"
```

---

### Task 5: Final verification + push

**Steps:**

1) Format:
```bash
.venv/bin/black pyimgano tests
```

2) Run full tests:
```bash
.venv/bin/pytest -q
```

3) Push:
```bash
git push origin main
```

