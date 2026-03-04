# Optional Dependencies (Extras)

`pyimgano` keeps the **core install** lightweight and pushes heavy runtimes behind
**extras** (optional dependencies).

Goals:

- ✅ `pip install pyimgano` should work for CPU baselines and basic CLI usage
- ✅ `import pyimgano` should not crash if you did **not** install extras
- ✅ When an extra is required, errors should be actionable (install hints)
- ✅ Offline-safe by default (no implicit weight downloads unless you opt in)

---

## Quick install map

| What you want | Install |
|--------------|---------|
| Deep models / torchvision backbones | `pip install "pyimgano[torch]"` |
| ONNX Runtime inference / ONNX export | `pip install "pyimgano[onnx]"` |
| OpenVINO inference | `pip install "pyimgano[openvino]"` |
| SSIM / phase-correlation / scikit-image baselines | `pip install "pyimgano[skimage]"` |
| Numba-accelerated baselines | `pip install "pyimgano[numba]"` |
| OpenCLIP backends | `pip install "pyimgano[clip]"` |
| Faster kNN (memory-bank methods) | `pip install "pyimgano[faiss]"` |
| anomalib checkpoint wrappers | `pip install "pyimgano[anomalib]"` |
| Common backends bundle | `pip install "pyimgano[backends]"` |
| Everything (dev/docs/viz + all backends) | `pip install "pyimgano[all]"` |

Notes:

- Some extras imply others (for example `pyimgano[clip]` includes `pyimgano[torch]`).
- Some third-party backends are not on PyPI (example: `patchcore-inspection`); install those separately.

---

## How to know which extra you need

### Suites / sweeps show `requires_extras`

Suite discovery outputs explicit extras requirements per baseline:

```bash
pyimgano-benchmark --suite-info industrial-v3 --json
```

Sweep discovery shows which baselines have variants:

```bash
pyimgano-benchmark --sweep-info industrial-template-small --json
```

### Missing extras are skipped (with install hints)

When running a suite, optional baselines are **skipped** if you didn’t install
the required extras. Skip reasons include install hints like:

- `pip install 'pyimgano[skimage]'`
- `pip install 'pyimgano[torch]'`

This is intentional: it keeps `pip install pyimgano` usable and avoids surprise
`ImportError` crashes at import time.

---

## Recommended combos (industrial)

| Environment | Recommendation |
|------------|----------------|
| CPU-only “template inspection” baselines | `pip install pyimgano` (+ `pyimgano[skimage]` if you want SSIM/phase-corr) |
| GPU anomaly maps (PatchCore / SoftPatch / DINO-based) | `pip install "pyimgano[torch]"` |
| Deployment runtime (ONNX) | `pip install "pyimgano[onnx]"` |
| Deployment runtime (OpenVINO) | `pip install "pyimgano[openvino]"` |
| Semantics-driven baselines | `pip install "pyimgano[clip]"` |

---

## Contributor install

For local development:

```bash
pip install -e ".[dev]"
```

If you are working on deep backends as well:

```bash
pip install -e ".[dev,torch]"
```
