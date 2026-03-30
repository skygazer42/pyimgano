# SonarCloud Push Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make SonarCloud run on every push to `main`, add a local reproduction command, add a focused Docker test image, and add a post-push Sonar issue fetch script.

**Architecture:** Keep Sonar in its dedicated GitHub Actions workflow rather than folding it into the main CI matrix. Add one repository shell entrypoint for local reproduction, one small Python API client for issue retrieval, and focused contract tests that lock the workflow, Docker, and docs behavior.

**Tech Stack:** GitHub Actions YAML, Python 3.9+, shell scripting, Docker, pytest, SonarCloud Web API, existing `sonar-project.properties`.

---

### Task 1: Add failing tests for the Sonar workflow contract

**Files:**
- Create: `tests/test_sonar_workflow_contract.py`
- Test: `.github/workflows/sonar.yml`

**Step 1: Write the failing test**

```python
def test_sonar_workflow_runs_on_push_without_repo_variable_gate():
    ...
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_sonar_workflow_contract.py -q`
Expected: FAIL because the workflow still depends on `ENABLE_SONARQUBE_CLOUD_SCAN`

**Step 3: Write minimal implementation**

```yaml
if: github.actor != 'dependabot[bot]' && ...
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_sonar_workflow_contract.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_sonar_workflow_contract.py .github/workflows/sonar.yml
git commit -m "test: lock sonar workflow push contract"
```

### Task 2: Make the Sonar workflow auto-run and wait for the quality gate

**Files:**
- Modify: `.github/workflows/sonar.yml`
- Test: `tests/test_sonar_workflow_contract.py`

**Step 1: Write the failing test**

```python
def test_sonar_workflow_waits_for_quality_gate():
    ...
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_sonar_workflow_contract.py -q`
Expected: FAIL because the workflow does not yet require quality-gate waiting

**Step 3: Write minimal implementation**

```yaml
with:
  args: >
    "-Dsonar.projectVersion=..."
    "-Dsonar.qualitygate.wait=true"
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_sonar_workflow_contract.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add .github/workflows/sonar.yml tests/test_sonar_workflow_contract.py
git commit -m "ci: run sonar on every push"
```

### Task 3: Add failing tests for the local Sonar runner and Docker recipe

**Files:**
- Create: `tests/test_sonar_local_runner.py`
- Test: `tools/run_sonar_local.sh`
- Test: `Dockerfile.sonar`

**Step 1: Write the failing test**

```python
def test_run_sonar_local_dry_run_prints_pytest_and_scanner_steps():
    ...
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_sonar_local_runner.py -q`
Expected: FAIL because the local runner and Dockerfile do not exist

**Step 3: Write minimal implementation**

```bash
#!/usr/bin/env bash
echo "pytest -v --cov=pyimgano --cov-report=xml --cov-report=term-missing"
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_sonar_local_runner.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_sonar_local_runner.py tools/run_sonar_local.sh Dockerfile.sonar
git commit -m "test: add local sonar runner contracts"
```

### Task 4: Implement the local Sonar runner and Docker image

**Files:**
- Create: `tools/run_sonar_local.sh`
- Create: `Dockerfile.sonar`
- Modify: `tests/test_sonar_local_runner.py`

**Step 1: Write the failing test**

```python
def test_run_sonar_local_requires_token_when_scan_enabled():
    ...
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_sonar_local_runner.py -q`
Expected: FAIL because token/error handling is not implemented yet

**Step 3: Write minimal implementation**

```bash
if [ "${skip_scan}" != "true" ] && [ -z "${SONAR_TOKEN:-}" ]; then
  echo "SONAR_TOKEN is required" >&2
  exit 1
fi
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_sonar_local_runner.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add tools/run_sonar_local.sh Dockerfile.sonar tests/test_sonar_local_runner.py
git commit -m "ci: add local sonar reproduction tooling"
```

### Task 5: Add failing tests for SonarCloud issue fetching

**Files:**
- Create: `tests/test_tools_fetch_sonar_issues.py`
- Test: `tools/fetch_sonar_issues.py`

**Step 1: Write the failing test**

```python
def test_fetch_sonar_issues_text_output_formats_quality_gate_and_issue_summary():
    ...
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_tools_fetch_sonar_issues.py -q`
Expected: FAIL with `ModuleNotFoundError` or missing script

**Step 3: Write minimal implementation**

```python
def main(argv=None):
    return 0
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_tools_fetch_sonar_issues.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_tools_fetch_sonar_issues.py tools/fetch_sonar_issues.py
git commit -m "test: add sonar issue fetch script contracts"
```

### Task 6: Implement SonarCloud issue fetching and CLI output

**Files:**
- Create: `tools/fetch_sonar_issues.py`
- Modify: `tests/test_tools_fetch_sonar_issues.py`

**Step 1: Write the failing test**

```python
def test_fetch_sonar_issues_json_output_includes_project_status_and_issues():
    ...
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_tools_fetch_sonar_issues.py -q`
Expected: FAIL because JSON formatting and HTTP plumbing are incomplete

**Step 3: Write minimal implementation**

```python
payload = {
    "project_status": project_status,
    "issues": issues,
}
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_tools_fetch_sonar_issues.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add tools/fetch_sonar_issues.py tests/test_tools_fetch_sonar_issues.py
git commit -m "feat: add sonar issue fetch helper"
```

### Task 7: Document the local and post-push Sonar workflow

**Files:**
- Modify: `CONTRIBUTING.md`
- Modify: `README.md`
- Modify: `tests/test_sonar_workflow_contract.py`

**Step 1: Write the failing test**

```python
def test_contributing_documents_local_sonar_and_issue_fetch_commands():
    ...
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_sonar_workflow_contract.py -q`
Expected: FAIL because docs do not yet mention the new commands

**Step 3: Write minimal implementation**

```markdown
./tools/run_sonar_local.sh --skip-scan
python3 tools/fetch_sonar_issues.py
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_sonar_workflow_contract.py tests/test_sonar_local_runner.py tests/test_tools_fetch_sonar_issues.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add CONTRIBUTING.md README.md tests/test_sonar_workflow_contract.py
git commit -m "docs: document sonar local and post-push workflow"
```

### Task 8: Run focused verification, Docker verification, and push-prep checks

**Files:**
- Modify: `.github/workflows/sonar.yml`
- Modify: `tools/run_sonar_local.sh`
- Modify: `tools/fetch_sonar_issues.py`
- Modify: `CONTRIBUTING.md`
- Modify: `README.md`

**Step 1: Run focused tests**

Run: `python3 -m pytest --no-cov tests/test_sonar_project_config.py tests/test_sonar_workflow_contract.py tests/test_sonar_local_runner.py tests/test_tools_fetch_sonar_issues.py -q`
Expected: PASS

**Step 2: Run local Sonar reproduction without scan**

Run: `bash tools/run_sonar_local.sh --skip-scan`
Expected: PASS and generate `coverage.xml`

**Step 3: Build and run Docker image**

Run: `docker build -f Dockerfile.sonar -t pyimgano-sonar-local .`
Expected: PASS

Run: `docker run --rm pyimgano-sonar-local --skip-scan`
Expected: PASS

**Step 4: Fetch Sonar issues**

Run: `SONAR_TOKEN=... python3 tools/fetch_sonar_issues.py --project-key skygazer42_pyimgano --limit 10`
Expected: PASS with quality-gate summary and issue list

**Step 5: Commit**

```bash
git add .github/workflows/sonar.yml Dockerfile.sonar tools/run_sonar_local.sh tools/fetch_sonar_issues.py CONTRIBUTING.md README.md tests/test_sonar_project_config.py tests/test_sonar_workflow_contract.py tests/test_sonar_local_runner.py tests/test_tools_fetch_sonar_issues.py docs/plans/2026-03-30-sonarcloud-push-integration-design.md docs/plans/2026-03-30-sonarcloud-push-integration.md
git commit -m "ci: automate sonar push verification"
```
