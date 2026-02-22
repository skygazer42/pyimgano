# Model Weights and Checkpoints

`pyimgano` intentionally avoids shipping large pretrained weights inside the wheel.
In production, weights should live on disk (or object storage) and be referenced explicitly.

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

