# Services Root Grouped Export Spec Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `pyimgano.services` easier to read and maintain by declaring its compatibility exports as grouped specifications instead of one flat block.

**Architecture:** Keep the root service facade lazy and compatibility-preserving, but represent its exports as ordered groups of `(export_name, module_name)` entries. Derive both `__all__` and `_SERVICE_EXPORT_SOURCES` from that grouped spec so the public surface, ownership, and ordering stay synchronized while the file becomes easier to scan by domain.

**Tech Stack:** Python 3.10, pytest, existing `pyimgano/services/__init__.py`, `tests/test_services_package.py`

---

### Task 1: Lock The Grouped Export Spec With Failing Tests

**Files:**
- Modify: `tests/test_services_package.py`

**Step 1: Write the failing test**

Add a test that proves:
- `pyimgano.services` defines a grouped export spec
- each group has a non-empty name and at least one export entry
- flattening the grouped spec reproduces `__all__`
- flattening the grouped spec reproduces `_SERVICE_EXPORT_SOURCES`

**Step 2: Run tests to verify they fail**

Run: `pytest --no-cov tests/test_services_package.py -v`
Expected: FAIL because the root facade does not yet declare a grouped export specification.

### Task 2: Refactor The Root Facade To Use The Grouped Spec

**Files:**
- Modify: `pyimgano/services/__init__.py`

**Step 1: Write minimal implementation**

- Introduce an ordered grouped export specification for the root service facade
- Derive `_SERVICE_EXPORT_SOURCES` and `__all__` from that grouped spec
- Preserve public exports, export ordering, lazy loading, and error behavior exactly

**Step 2: Run focused tests to verify they pass**

Run: `pytest --no-cov tests/test_services_package.py -v`
Expected: PASS

### Task 3: Regressions And Hygiene

**Files:**
- Modify: `docs/plans/2026-03-12-services-root-grouped-export-spec.md`
- Modify: `pyimgano/services/__init__.py`
- Modify: `tests/test_services_package.py`

**Step 1: Run regression suite**

Run: `pytest --no-cov tests/test_services_package.py tests/test_service_import_style.py tests/test_architecture_boundaries.py tests/test_infer_continue_service.py -v`
Expected: PASS

**Step 2: Check patch hygiene**

Run: `git diff --check -- docs/plans/2026-03-12-services-root-grouped-export-spec.md pyimgano/services/__init__.py tests/test_services_package.py`
Expected: no output
