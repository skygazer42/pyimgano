# Anomalib Checkpoints (Train with anomalib, evaluate with PyImgAno)

PyImgAno includes **optional** wrappers that let you evaluate **anomalib-trained checkpoints** using the same:

- dataset loaders (MVTec AD / VisA)
- image-level metrics (AUROC/AP/F1, etc.)
- pixel-level metrics (pixel AUROC / pixel AP / AUPRO, when masks exist)
- JSON reporting via `--output`

This keeps `anomalib` as a *training / checkpoint export* dependency, while PyImgAno stays the **single** benchmarking CLI.

---

## 1) Install

```bash
# Adds anomalib as an optional backend dependency.
pip install "pyimgano[anomalib]"
```

If you already use multiple optional backends:

```bash
pip install "pyimgano[backends]"
```

---

## 2) Pick a wrapper model name

All anomalib checkpoint wrappers are registered models (so they work with `pyimgano-benchmark --model ...`).

### Generic wrapper

- `vision_anomalib_checkpoint` (generic; requires `checkpoint_path`)

### Alias wrappers (tags/metadata only)

These currently map to the same inference wrapper implementation, but provide clearer names for reports and filtering:

- `vision_patchcore_anomalib`
- `vision_padim_anomalib`
- `vision_stfpm_anomalib`
- `vision_draem_anomalib`
- `vision_fastflow_anomalib`
- `vision_reverse_distillation_anomalib`
- `vision_dfm_anomalib`
- `vision_cflow_anomalib`
- `vision_efficientad_anomalib`
- `vision_dinomaly_anomalib`
- `vision_cfa_anomalib`
- `vision_csflow_anomalib`
- `vision_dfkde_anomalib`
- `vision_dsr_anomalib`
- `vision_ganomaly_anomalib`
- `vision_rkde_anomalib`
- `vision_uflow_anomalib`
- `vision_winclip_anomalib`

Tip: you can list what your installed version exposes:

```bash
python -c "from pyimgano.models import list_models; print(list_models(tags=['anomalib']))"
```

---

## 3) Run benchmarking on a checkpoint

### CLI (recommended)

```bash
pyimgano-benchmark \
  --dataset mvtec \
  --root /path/to/mvtec_ad \
  --category bottle \
  --model vision_patchcore_anomalib \
  --checkpoint-path /path/to/anomalib/model.pt \
  --device cuda \
  --pixel \
  --output runs/mvtec_bottle_patchcore_anomalib.json
```

Notes:
- `--pixel` enables pixel metrics if the dataset split contains masks and the checkpoint returns anomaly maps.
- `--device` is passed to the wrapper if the underlying constructor supports it.
- The wrapper calibrates a threshold from the **train split** using `contamination` (default `0.1`).

### Passing advanced kwargs

You can pass additional constructor kwargs (as JSON):

```bash
pyimgano-benchmark \
  --dataset mvtec \
  --root /path/to/mvtec_ad \
  --category bottle \
  --model vision_anomalib_checkpoint \
  --checkpoint-path /path/to/anomalib/model.pt \
  --model-kwargs '{"contamination": 0.05}' \
  --device cuda
```

Rules:
- `--model-kwargs` must be a **JSON object**.
- `--checkpoint-path` and `--model-kwargs '{"checkpoint_path": "..."}'` must **match** (conflicts error out).
- For strict constructors (no `**kwargs`), unknown keys in `--model-kwargs` are rejected with a clear error.

---

## 4) Common pitfalls / troubleshooting

### `TRUST_REMOTE_CODE` requirement (anomalib v2)

Recent anomalib versions mark `TorchInferencer` as legacy and add a safety guard for checkpoint loading.
If you see an error mentioning `TRUST_REMOTE_CODE`, set:

```bash
export TRUST_REMOTE_CODE=1
```

### Security note (loading checkpoints)

PyTorch model loading uses Python pickle under the hood; only load checkpoints from sources you trust.

### Checkpoint format

PyImgAno’s wrapper uses anomalib’s `TorchInferencer(path=..., device=...)`.
In practice this is commonly a `.pt` exported weight file, but anomalib may also support other extensions depending on version.

---

## 5) References (anomalib upstream)

- Supported models / model reference: https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/models/
- Inference API (includes `TorchInferencer`): https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/deploy/
- Project repo: https://github.com/open-edge-platform/anomalib

