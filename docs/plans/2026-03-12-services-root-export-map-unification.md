# Services Root Export Map Unification Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `pyimgano.services` a clearer compatibility facade by replacing implicit module scanning with an explicit public export-to-module mapping.

**Architecture:** Keep `pyimgano.services` as a lazy compatibility layer, but declare a single mapping that says exactly which module owns each root export. Derive `__all__` from that mapping and update `__getattr__` to import only the owning module for a requested symbol, which preserves behavior while reducing accidental coupling and unrelated imports.

**Tech Stack:** Python 3.10, pytest, existing `pyimgano/services/__init__.py`, `tests/test_services_package.py`

---

### Task 1: Lock The Root Facade Behavior With Failing Tests

**Files:**
- Modify: `tests/test_services_package.py`

**Step 1: Write the failing test**

Add tests that prove:
- `pyimgano.services` maintains an explicit export-source mapping whose keys match `__all__`
- every mapped export is provided by the declared module
- resolving `services.resolve_model_options` loads `pyimgano.services.model_options` but does not load unrelated modules such as `pyimgano.services.benchmark_service`

**Step 2: Run tests to verify they fail**

Run: `pytest --no-cov tests/test_services_package.py -v`
Expected: FAIL because the root facade does not yet declare an explicit export-source mapping and symbol lookup still loads unrelated service modules while scanning.

### Task 2: Replace Module Scanning With Explicit Mapping

**Files:**
- Modify: `pyimgano/services/__init__.py`

**Step 1: Write minimal implementation**

- Introduce a single export-to-module mapping for the root service facade
- Derive `__all__` from the mapping in the existing public order
- Update `__getattr__` to import only the declared provider module for a requested symbol
- Preserve current public exports, lazy import behavior, error semantics, and compatibility with existing callers

**Step 2: Run focused tests to verify they pass**

Run: `pytest --no-cov tests/test_services_package.py -v`
Expected: PASS

### Task 3: Regressions And Hygiene

**Files:**
- Modify: `docs/plans/2026-03-12-services-root-export-map-unification.md`
- Modify: `pyimgano/services/__init__.py`
- Modify: `tests/test_services_package.py`

**Step 1: Run regression suite**

Run: `pytest --no-cov tests/test_services_package.py tests/test_service_import_style.py tests/test_architecture_boundaries.py tests/test_infer_continue_service.py -v`
Expected: PASS

**Step 2: Check patch hygiene**

Run: `git diff --check -- docs/plans/2026-03-12-services-root-export-map-unification.md pyimgano/services/__init__.py tests/test_services_package.py`
Expected: no output
