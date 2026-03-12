# Service Package Unification Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Unify `pyimgano.services` into a consistent package shape by making the root package a lazy compatibility facade and standardizing source imports on direct module paths.

**Architecture:** Keep `pyimgano.services` as a public compatibility entrypoint, but stop treating it as an internal barrel import target. Replace eager root-package imports with a lazy `__getattr__` facade, then add AST guard tests so source modules consistently import `pyimgano.services.<module>` instead of `from pyimgano.services import ...`.

**Tech Stack:** Python 3.10, pytest, `ast`, `importlib`, `subprocess`, existing service modules.

---

### Task 1: Make `pyimgano.services` a Lazy Compatibility Facade

**Files:**
- Modify: `pyimgano/services/__init__.py`
- Test: `tests/test_services_package.py`

**Step 1: Write the failing test**

Add `tests/test_services_package.py` with tests that:
- prove importing `pyimgano.services` does not eagerly import `pyimgano.services.benchmark_service`
- prove exported names like `BenchmarkRunRequest` and `collect_doctor_payload` are still accessible through the root package

**Step 2: Run test to verify it fails**

Run: `pytest --no-cov tests/test_services_package.py -v`
Expected: FAIL because `pyimgano.services` eagerly imports service modules today.

**Step 3: Write minimal implementation**

- Replace eager imports in `pyimgano/services/__init__.py` with a lazy facade:
  - keep `__all__`
  - add `__getattr__`
  - resolve exported names from known service modules on first access
  - cache resolved attributes in module globals

**Step 4: Run tests to verify they pass**

Run: `pytest --no-cov tests/test_services_package.py -v`
Expected: PASS

### Task 2: Standardize Source Imports on Direct Service Module Paths

**Files:**
- Modify: `pyimgano/cli.py`
- Modify: `pyimgano/robust_cli.py`
- Create: `tests/test_service_import_style.py`

**Step 1: Write the failing test**

Create `tests/test_service_import_style.py` with an AST-based test that scans `pyimgano/**/*.py` and fails if any source file other than `pyimgano/services/__init__.py` imports from the root service package with:
- `from pyimgano.services import ...`
- `import pyimgano.services as ...`

**Step 2: Run test to verify it fails**

Run: `pytest --no-cov tests/test_service_import_style.py -v`
Expected: FAIL because `pyimgano/cli.py` and `pyimgano/robust_cli.py` still import via the root service package.

**Step 3: Write minimal implementation**

- Update `pyimgano/cli.py` to import the exact service modules it uses
- Update `pyimgano/robust_cli.py` to import the exact service modules it uses
- Leave test-only imports alone

**Step 4: Run tests to verify they pass**

Run: `pytest --no-cov tests/test_service_import_style.py tests/test_cli_smoke.py tests/test_robust_cli_smoke.py -v`
Expected: PASS

### Task 3: Run Focused Regression Coverage

**Files:**
- Test: `tests/test_services_package.py`
- Test: `tests/test_service_import_style.py`
- Test: `tests/test_cli_smoke.py`
- Test: `tests/test_robust_cli_smoke.py`
- Test: `tests/test_doctor_cli.py`
- Test: `tests/test_pyim_cli_model_presets.py`
- Test: `tests/test_architecture_boundaries.py`

**Step 1: Run regression suite**

Run: `pytest --no-cov tests/test_services_package.py tests/test_service_import_style.py tests/test_cli_smoke.py tests/test_robust_cli_smoke.py tests/test_doctor_cli.py tests/test_pyim_cli_model_presets.py tests/test_architecture_boundaries.py -v`
Expected: PASS

**Step 2: Check patch hygiene**

Run: `git diff --check -- pyimgano/services/__init__.py pyimgano/cli.py pyimgano/robust_cli.py tests/test_services_package.py tests/test_service_import_style.py docs/plans/2026-03-12-service-package-unification.md`
Expected: no output
