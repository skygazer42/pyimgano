# SonarCloud Push Integration (Design)

**Date:** 2026-03-30

## Goal

Make SonarCloud analysis run on every push and pull request to `main`, while also
providing a repeatable local validation path that can be exercised on the host,
inside Docker, and after push through the SonarCloud Web API.

## Context / Problem

The repository already has:

- [`.github/workflows/sonar.yml`](/data/pyimgano/.github/workflows/sonar.yml)
- [`sonar-project.properties`](/data/pyimgano/sonar-project.properties)
- Sonar-focused regression tests such as
  [`tests/test_sonar_project_config.py`](/data/pyimgano/tests/test_sonar_project_config.py)

But the workflow is effectively disabled by the repository variable gate
`ENABLE_SONARQUBE_CLOUD_SCAN == 'true'`. That means:

- pushes do not automatically produce SonarCloud analysis
- there is no repository-owned local command that reproduces the workflow steps
- post-push issue review depends on manual API calls
- there is no dedicated container recipe for CI/Sonar reproduction

The user explicitly wants a push-driven workflow with local verification first,
followed by push and issue retrieval.

## Non-goals

- No rewrite of the main CI workflow
- No change to the existing Sonar project key or organization
- No attempt to fix the entire current Sonar issue backlog in this pass
- No broad containerization of the whole repository beyond a focused Sonar test image

## Approaches Considered

### A) Keep a dedicated Sonar workflow and make it run on every push (recommended)

Keep Sonar in its own workflow, remove the repository-variable kill switch, and add
repo-owned local tooling for coverage, Docker reproduction, and Sonar API issue fetches.

Pros:

- preserves CI isolation from the main matrix workflow
- matches the user's "every push" requirement
- keeps Sonar-specific dependencies and debugging focused
- allows local host and Docker verification with the same commands

Cons:

- introduces a small amount of new tooling to maintain

### B) Fold Sonar into the main CI workflow

Pros:

- one workflow to look at

Cons:

- external SonarCloud instability would directly affect the main CI path
- larger workflow churn than necessary

### C) Keep Sonar opt-in and only add local tooling

Pros:

- smallest diff

Cons:

- does not satisfy the push-on-every-commit requirement

## Proposed Design

### 1) Make SonarCloud workflow push-driven again

Update [`.github/workflows/sonar.yml`](/data/pyimgano/.github/workflows/sonar.yml) so it
still triggers on `push`, `pull_request`, and `workflow_dispatch`, but no longer depends
on `ENABLE_SONARQUBE_CLOUD_SCAN`.

Keep the safer runtime guardrails:

- skip Dependabot runs
- skip fork PR scans that cannot safely access repository secrets

Also make the scan wait for the quality gate so the Sonar workflow result reflects
the server-side gate outcome rather than only local test execution.

### 2) Add a repo-owned local Sonar runner

Introduce a small shell entrypoint, `tools/run_sonar_local.sh`, that:

- runs from the repository root
- optionally installs the same Python dependency set used by Sonar CI
- runs the coverage-producing pytest command
- optionally launches the Sonar scanner through Docker when `SONAR_TOKEN` is present
- supports `--dry-run` so its contract can be tested cheaply in CI

This script becomes the canonical local reproduction path for Sonar CI.

### 3) Add a focused Docker recipe for Sonar reproduction

Introduce a dedicated Dockerfile, `Dockerfile.sonar`, that packages the system
dependencies and local script entrypoint needed to reproduce the Sonar test phase.

The intended workflow is:

- build the image locally
- run the image to execute the local Sonar script with `--skip-scan` or a real token

This gives a clean-room verification path that is independent of the host Python
environment.

### 4) Add a small SonarCloud API client for post-push checks

Introduce `tools/fetch_sonar_issues.py` to fetch:

- project quality gate status
- unresolved issues

from SonarCloud using `SONAR_TOKEN`.

The script should provide:

- compact human-readable output by default
- JSON output for automation
- paging controls sufficient for practical issue review

This becomes the standard "push then inspect Sonar" command.

### 5) Lock the behavior with lightweight contract tests

Add focused tests that cover:

- Sonar workflow auto-run contract and absence of the variable gate
- local Sonar runner dry-run output and required commands
- Dockerfile references to the local runner
- Sonar issue fetch script request/format behavior using monkeypatched HTTP calls
- contributor docs references for local Sonar and post-push inspection

## Testing Strategy

The implementation should verify the following layers:

1. unit/contract tests for workflow, scripts, and docs
2. local script dry-run
3. local script real test phase (`--skip-scan`)
4. Docker image build plus containerized `--skip-scan`
5. post-push Sonar API fetch using `SONAR_TOKEN`

## Risks / Mitigations

- SonarCloud availability can still fluctuate.
  Mitigation: keep Sonar isolated in its own workflow and support local reproduction.

- Local scanner setup is often brittle.
  Mitigation: use Dockerized scanner invocation from the local helper instead of
  requiring a host-installed scanner.

- Docker image builds may become slow.
  Mitigation: keep the image narrowly scoped to the Sonar reproduction path and
  avoid turning it into a general development container.
