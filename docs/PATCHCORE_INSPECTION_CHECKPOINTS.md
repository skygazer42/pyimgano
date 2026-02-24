# PatchCore-Inspection Checkpoints (amazon-science/patchcore-inspection)

PyImgAno provides an **optional** backend wrapper to evaluate models saved by:

- `amazon-science/patchcore-inspection` (PatchCore reference implementation)

This is useful if you want to:

- train with the official PatchCore-Inspection repo
- benchmark with PyImgAno’s unified CLI, metrics, and JSON reports

---

## 1) Install

The PatchCore-Inspection backend is optional.

```bash
pip install "pyimgano"
pip install "patchcore @ git+https://github.com/amazon-science/patchcore-inspection.git"
```

Note:
- PyPI does not allow published packages to declare VCS / direct-URL dependencies in metadata.
- Therefore `pyimgano` does not install PatchCore-Inspection automatically; install it explicitly as above.

---

## 2) Model name

Use:

- `vision_patchcore_inspection_checkpoint`

This wrapper expects a **saved model folder** created by PatchCore-Inspection (not a single `.pt` file).

---

## 3) Benchmark a checkpoint

```bash
pyimgano-benchmark \
  --dataset mvtec \
  --root /path/to/mvtec_ad \
  --category bottle \
  --model vision_patchcore_inspection_checkpoint \
  --checkpoint-path /path/to/patchcore_inspection/saved_model_dir \
  --device cuda \
  --pixel \
  --output runs/mvtec_bottle_patchcore_inspection.json
```

Notes:
- The wrapper uses ImageNet normalization and a `Resize(256) -> CenterCrop(224)` preprocessing by default.
- You can override these via `--model-kwargs`, e.g. `--model-kwargs '{"resize": 366, "imagesize": 320}'`.

---

## 4) References (upstream)

- PatchCore-Inspection repo: https://github.com/amazon-science/patchcore-inspection
