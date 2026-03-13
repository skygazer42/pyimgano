# Sonar Cleanup Wave 1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove a first safe batch of SonarCloud issues without interfering with the in-progress workbench architecture refactor.

**Architecture:** Restrict this wave to low-risk, localized cleanups in tests and CI workflow files. Avoid broad production refactors such as cognitive-complexity reductions until the dirty worktree is smaller and each subsystem boundary change has landed.

**Tech Stack:** Python, pytest, GitHub Actions, SonarCloud

---

### Task 1: Confirm the first safe issue families

**Files:**
- Modify: `docs/plans/2026-03-13-sonar-cleanup-wave-1.md`
- Inspect: `.github/workflows/docs.yml`
- Inspect: `tests/test_benchmark_service.py`
- Inspect: `tests/test_cli_baseline_suites_v16.py`
- Inspect: `tests/test_cli_discovery.py`
- Inspect: `tests/test_cli_smoke.py`
- Inspect: `tests/test_infer_artifact_service.py`
- Inspect: `tests/test_infer_cli_discovery_and_model_presets_v16.py`
- Inspect: `tests/test_infer_cli_infer_config.py`
- Inspect: `tests/test_infer_cli_preprocessing_preset.py`
- Inspect: `tests/test_infer_cli_smoke.py`
- Inspect: `tests/test_infer_context_service.py`
- Inspect: `tests/test_infer_load_service.py`
- Inspect: `tests/test_infer_options_service.py`
- Inspect: `tests/test_infer_runtime_service.py`
- Inspect: `tests/test_infer_wrapper_service.py`
- Inspect: `tests/test_model_options_service.py`

**Step 1: Verify SonarCloud issue families to target**

Run:
```bash
zsh -lic 'curl -sf -u "$SONARQUBE_TOKEN:" "https://sonarcloud.io/api/issues/search?componentKeys=skygazer42_pyimgano&resolved=false&ps=1&facets=rules,severities,types" | jq .'
```

Expected: open issue totals and top rules, confirming the backlog is too large for one unsafe production refactor.

**Step 2: Limit this wave to low-risk issues**

Target:
- `python:S1244` in test assertions
- `python:S7504` unnecessary `list()` in tests
- `pythonbugs:S6466` possible `IndexError` in tests
- `python:S5655` mismatched test stub argument type
- GitHub Actions workflow dependency vulnerabilities if they can be resolved by version bumps alone

**Step 3: Avoid high-risk issue families in this wave**

Defer:
- `python:S3776` cognitive complexity in production modules
- package-wide naming/style rule sweeps
- shared service and workbench refactors already in flight

### Task 2: Fix test-only Sonar issues

**Files:**
- Modify: `tests/test_benchmark_service.py`
- Modify: `tests/test_cli_baseline_suites_v16.py`
- Modify: `tests/test_cli_discovery.py`
- Modify: `tests/test_cli_smoke.py`
- Modify: `tests/test_infer_artifact_service.py`
- Modify: `tests/test_infer_cli_discovery_and_model_presets_v16.py`
- Modify: `tests/test_infer_cli_infer_config.py`
- Modify: `tests/test_infer_cli_preprocessing_preset.py`
- Modify: `tests/test_infer_cli_smoke.py`
- Modify: `tests/test_infer_context_service.py`
- Modify: `tests/test_infer_load_service.py`
- Modify: `tests/test_infer_options_service.py`
- Modify: `tests/test_infer_runtime_service.py`
- Modify: `tests/test_infer_wrapper_service.py`
- Modify: `tests/test_model_options_service.py`

**Step 1: Replace direct float equality assertions with tolerant assertions**

Examples:
```python
assert value == pytest.approx(0.5)
```

**Step 2: Remove unnecessary `list()` wrappers around already iterable objects**

Examples:
```python
np.asarray([0.1 for _ in X], dtype=np.float32)
```

or, when possible:

```python
np.full(len(X), 0.1, dtype=np.float32)
```

**Step 3: Make test indexing safe before accessing `calls[0]`**

Examples:
```python
assert len(calls) == 1
assert calls[0] == expected
```

**Step 4: Fix mismatched argument type in the preprocessing preset test**

Adjust the test stub/object shape so the helper receives the argument type it expects.

### Task 3: Fix workflow dependency vulnerabilities if version-only

**Files:**
- Modify: `.github/workflows/docs.yml`

**Step 1: Inspect current action versions**

Run:
```bash
sed -n '1,220p' .github/workflows/docs.yml
```

**Step 2: Upgrade workflow actions only if the change is low-risk**

Acceptable:
- patch/minor updates to maintained `actions/*` references

Reject for this wave:
- behavior-changing workflow redesign

### Task 4: Verify the cleanup batch

**Files:**
- Test: targeted pytest modules touched in Task 2
- Test: workflow file syntax by inspection only if no local workflow linter exists

**Step 1: Run focused tests for edited test modules**

Run:
```bash
pytest --no-cov tests/test_benchmark_service.py tests/test_cli_baseline_suites_v16.py tests/test_cli_discovery.py tests/test_cli_smoke.py tests/test_infer_artifact_service.py tests/test_infer_cli_discovery_and_model_presets_v16.py tests/test_infer_cli_infer_config.py tests/test_infer_cli_preprocessing_preset.py tests/test_infer_cli_smoke.py tests/test_infer_context_service.py tests/test_infer_load_service.py tests/test_infer_options_service.py tests/test_infer_runtime_service.py tests/test_infer_wrapper_service.py tests/test_model_options_service.py -v
```

Expected: edited tests pass.

**Step 2: Check patch hygiene**

Run:
```bash
git diff --check -- .github/workflows/docs.yml tests/test_benchmark_service.py tests/test_cli_baseline_suites_v16.py tests/test_cli_discovery.py tests/test_cli_smoke.py tests/test_infer_artifact_service.py tests/test_infer_cli_discovery_and_model_presets_v16.py tests/test_infer_cli_infer_config.py tests/test_infer_cli_preprocessing_preset.py tests/test_infer_cli_smoke.py tests/test_infer_context_service.py tests/test_infer_load_service.py tests/test_infer_options_service.py tests/test_infer_runtime_service.py tests/test_infer_wrapper_service.py tests/test_model_options_service.py docs/plans/2026-03-13-sonar-cleanup-wave-1.md
```

Expected: no whitespace or merge-marker issues in edited files.
