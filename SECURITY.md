# Security Policy

## Supported Versions

This project follows best-effort security support for recent releases.

- Latest released version on PyPI: supported
- Older versions: security fixes are not guaranteed

If you are using an older version, upgrade to the latest release before
reporting a vulnerability.

## Checkpoint Trust Boundary

Treat model artifacts as untrusted input unless you created them or verified
their provenance and digest. Pyimgano's built-in restore paths accept its
non-executable JSON/NumPy safe-checkpoint formats and safe Torch state-dict
payloads by default. Legacy pickle/joblib checkpoints require an explicit
`trusted=True`, `--trust-checkpoint`, or `--trust-detector` decision.

Do not enable a trust flag for files downloaded from an unknown or mutable
source. A SHA-256 digest detects accidental or malicious modification, but it
does not make executable serialization safe or establish who authored it.

## Temporary Dependency Audit Exception

The fully pinned optional-backend profile currently has one upstream exception:

- `lightning==2.6.5`: `PYSEC-2026-3624` / `CVE-2026-58659`
- Scope: installed transitively by the optional `anomalib` backend
- Exposure: Lightning's `load_from_checkpoint` can import attacker-controlled
  module names from a crafted checkpoint and execute code
- Mitigation: never load untrusted Lightning/Anomalib checkpoints; pyimgano's
  own checkpoint trust gate does not make a direct upstream Lightning API call
  safe
- Tracking: https://github.com/Lightning-AI/pytorch-lightning/issues/21913
- Review date: every dependency-profile refresh and before each release

CI runs `pip-audit` as a blocking check and ignores only the exact advisory ID
above. Any additional known vulnerability fails the pinned full-profile job.
Remove the exception as soon as Lightning publishes a fixed release.

## Reporting a Vulnerability

Please do not open a public GitHub issue for security vulnerabilities.

Instead, use one of the following private reporting channels:

- GitHub Security Advisories (preferred, if enabled for the repo)
- If advisories are not available, open an issue with minimal details and ask
  maintainers for a private follow-up

When reporting, include:

- A clear description of the issue and impact
- Steps to reproduce (or a proof-of-concept) if available
- Affected versions and environment details
- Any mitigation or workaround you have found

## Response Process

Maintainers will:

- Acknowledge receipt as soon as practical
- Assess severity and scope
- Prepare a fix and coordinate a release
- Credit reporters when appropriate (if desired)
