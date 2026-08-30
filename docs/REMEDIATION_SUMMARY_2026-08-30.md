# Current remediation status (2026-08-30)

This is the short current-state companion to
`PAPER_CODE_AUDIT_2026-08-30.md`. The detailed report keeps historical finding
text for traceability; this file describes the post-remediation behavior.

## Current guarantees

- Paper/formula findings identified in the audit have code fixes, regression
  tests, or explicit external-backend boundaries.
- Legacy pickle/joblib loading is fail-closed and requires an explicit trust
  decision. Newly written joblib artifacts carry SHA-256 integrity sidecars.
- PatchCore-lite-map, Student-Teacher-lite, and compatible fitted detector states
  use the non-executable `pyimgano.safe-checkpoint` JSON/NumPy archive format.
  Deploy bundles therefore restore structured classical detector state without
  granting legacy joblib execution trust.
- The security-refreshed optional Python 3.10 environment is machine-pinned in
  `constraints/optional-py310-current.txt` and exercised by CI.
- The supported Python floor is 3.9; package metadata uses current SPDX/PEP 639
  licensing and builds cleanly with setuptools 77 or newer.
- The pinned profile is dependency-audited as a blocking CI gate. Its single
  temporary upstream exception is documented in `SECURITY.md`.
- CI separates compatibility smoke tests, one pinned full coverage run, optional
  backend integration, Semgrep, dependency audit, build, and deploy smoke. The
  deploy gate executes the exported bundle end to end.
- The golden MVTec AD entry point records dataset content hashes, environment,
  seeds, model settings, AUROC/AP/F1, pixel metrics, and AUPRO.

## Verified gates

- Full suite: 2887 passed, 0 skipped, 0 warnings, 0 failed.
- Official-weight GPU E2E: OpenCLIP and PatchCore passed on an RTX 3070 Ti
  under Torch 2.13.0+cu130, including finite maps and score direction.
- Registry metadata: 279 models, with no required, recommended, or invalid
  field issues; all 64 audited core models pass score-direction checks.
- Security: Semgrep ran 200 rules over 1083 targets with no findings; Bandit
  reported no medium/high findings. `pip-audit` has one documented upstream
  Lightning exception and fails on every additional advisory.

## Deliberate external boundary

The repository does not redistribute MVTec AD. Until a licensed dataset root is
provided, only the golden benchmark code path and synthetic smoke checks can be
verified; no full MVTec numerical result may be claimed. See
`GOLDEN_MVTEC_BENCHMARK.md` for the exact command.

## Release checklist

1. Run the clean-environment install under the named constraints profile.
2. Run the full test suite with warnings reported.
3. Run the optional integration tests and Semgrep profiles used by CI.
4. Run `tools/run_golden_mvtec_benchmark.py --check-only` against the licensed data.
5. Run the full golden benchmark and archive all three output artifacts.
