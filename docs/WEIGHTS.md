# Model Weights and Checkpoints

`pyimgano` intentionally avoids shipping large pretrained weights inside the wheel.
In production, weights should live on disk (or object storage) and be referenced explicitly.

## Weights manifest (optional, recommended for production)

If you manage local checkpoints/weights at scale, keep a small **manifest JSON**
so that paths, hashes, and licensing context are auditable.

Example `weights_manifest.json`:

```json
{
  "schema_version": 1,
  "entries": [
    {
      "name": "patchcore_wrn50_imagenet1k",
      "path": "checkpoints/patchcore_wrn50.pt",
      "sha256": "…",
      "license": "MIT",
      "source": "internal registry or upstream training run",
      "models": ["vision_patchcore"]
    }
  ]
}
```

Validate the manifest (never downloads anything):

```bash
pyimgano-weights validate ./weights_manifest.json --check-files --check-hashes --json
```

Generate a starter manifest:

```bash
pyimgano-weights template manifest > ./weights_manifest.json
```

Compute a file hash:

```bash
pyimgano-weights hash ./checkpoints/patchcore_wrn50.pt
```

Validate a model card JSON file:

```bash
pyimgano-weights validate-model-card ./model_card.json --json
```

Validate the model card plus the referenced checkpoint asset:

```bash
pyimgano-weights \
  validate-model-card ./model_card.json \
  --check-files \
  --check-hashes \
  --json
```

Validate the model card against a weights manifest too:

```bash
pyimgano-weights \
  validate-model-card ./model_card.json \
  --manifest ./weights_manifest.json \
  --check-files \
  --check-hashes \
  --json
```

Notes:

- `weights.path` is resolved relative to the model card file by default.
- Use `--base-dir DIR` to resolve relative paths against a different root.
- `--check-hashes` verifies `weights.sha256` when the referenced asset exists.
- `--manifest FILE` cross-checks the model card against a manifest entry.
- Set `weights.manifest_entry` in the model card when you want an explicit, stable link.

Recommended metadata per manifest entry:

- `source` — where the checkpoint came from
- `license` — redistributability / internal-use boundary
- `runtime` or `runtimes` — which runtime path(s) the asset supports (`torch`, `onnx`, `openvino`, ...)

## Deploy bundle manifests

If you export a deployable run bundle, `pyimgano` also writes
`deploy_bundle/bundle_manifest.json` next to the copied assets:

```bash
pyimgano-train --config ./train.json --export-deploy-bundle
```

That bundle manifest is separate from `weights_manifest.json`:

- `weights_manifest.json` tracks reusable checkpoint inventory across environments
- `deploy_bundle/bundle_manifest.json` tracks the exact files copied into one deploy bundle

The deploy bundle manifest records:

- relative destination paths inside the bundle
- file sizes
- sha256 digests
- source run metadata and environment fingerprint
- explicit roles for core files such as `infer_config.json`, `calibration_card.json`,
  `model_card.json`, and `weights_manifest.json`

If `model_card.json` or `weights_manifest.json` are present in the deploy bundle
root, bundle validation now checks them too instead of treating them as opaque
attachments.

See also: `docs/WORKBENCH.md` and `docs/CLI_REFERENCE.md`.

## Recommended patterns

### 1) Pass checkpoint paths explicitly (backend wrappers)

For checkpoint-backed models (e.g. anomalib wrappers), pass a checkpoint path:

```bash
pyimgano-benchmark \
  --dataset mvtec \
  --root /datasets/mvtec_ad \
  --category bottle \
  --model vision_patchcore_anomalib \
  --checkpoint-path /checkpoints/patchcore.ckpt \
  --device cuda
```

In Python:

```python
from pyimgano.models import create_model

det = create_model(
    "vision_anomalib_checkpoint",
    checkpoint_path="/checkpoints/model.ckpt",
    device="cuda",
)
```

### 2) Use cache directories for auto-downloaded weights

Some models/backbones download weights via upstream libraries (PyTorch, HuggingFace, OpenCLIP).
Prefer controlling cache locations in production:

- `TORCH_HOME` (torch / torchvision caches)
- `HF_HOME` / `TRANSFORMERS_CACHE` (HuggingFace)

Example:

```bash
export TORCH_HOME=/mnt/models/torch_cache
export HF_HOME=/mnt/models/hf_cache
```

## What not to do

- Don’t bake multi-GB checkpoints into Docker images unless you have a strong reason.
- Don’t commit weights into the git repo.
- Don’t expand `pyimgano` wheel size by vendoring upstream checkpoints.

See also: `docs/MODEL_CARDS.md`.
