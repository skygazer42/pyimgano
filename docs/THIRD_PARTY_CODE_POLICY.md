# Third-Party Code Policy

This project (`pyimgano`) is MIT-licensed and aims to be a **self-contained** industrial anomaly
detection toolkit. We do study external industrial projects, but we must handle upstream code
carefully to keep licensing clean and the runtime dependency surface stable.

## Allowed

- **Shallow cloning external repos for study** into a local, gitignored cache directory
  (see `tools/clone_reference_repos.sh`). These clones must never be committed.
- **Re-implementing ideas** (architecture, algorithms, training recipes) using our own code and
  our own base contracts (`BaseDetector`, `BaseVisionDetector`, `BaseVisionDeepDetector`, etc.).
- **Copying small code snippets** only when:
  - The upstream license is compatible with MIT (e.g. MIT, Apache-2.0, BSD-2/3-Clause).
  - We preserve all required notices and attribution.
  - We document exactly what was copied and from where.

## Not Allowed

- Adding a runtime dependency on `pyod` or any other outlier library as a “thin wrapper”.
  Implementations must be native and centered on our own base classes.
- Copying code from **GPL** repositories into this MIT project (license incompatibility).
  GPL repos can be used for study only.
- Bundling model weights, large binary assets, or datasets into this repository.

## Required Process When Copying Code

1. **Create/Update notices** under `third_party/`:
   - Add the upstream project's license text.
   - Add a short entry in `third_party/NOTICE.md` describing:
     - Upstream repo name + commit hash (or tag)
     - File path(s) copied
     - What was modified

2. **Annotate copied source files** with a short header comment:

```python
# UPSTREAM: <repo> @ <commit>
# LICENSE: <license identifier>
# NOTES: <what was copied/changed>
```

3. **Run the audit tool** before final commit:

```bash
python tools/audit_third_party_notices.py
```

This tool checks that `UPSTREAM:` markers are accompanied by a notice entry under `third_party/`.

## Why This Policy Exists

- Keep the package legally clean and easy to redistribute.
- Keep runtime import surfaces stable (no surprise heavy dependencies).
- Preserve the core goal: build **our own** industrial AD package, not a thin wrapper.

