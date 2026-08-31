# Trained Artifacts: Export, Import, and Inference

PyImgAno artifacts are relocatable, manifest-backed runtime packages for fitted
detectors. They carry the executable model state, tensor contracts, the
artifact-local inference policy, integrity metadata, and verification evidence.
A raw model file is not an artifact because tensor shapes alone do not define
anomaly-score semantics, preprocessing, thresholds, or map handling.

## Install the smallest runtime profile

| Task | Install |
|---|---|
| Native artifact runtime | `pip install pyimgano` plus any dependency required by the model |
| Import or run ONNX, without Torch | `pip install "pyimgano[onnx-runtime]"` |
| Create and run ONNX artifacts | `pip install "pyimgano[onnx-export]"` |
| Run OpenVINO artifacts, without Torch | `pip install "pyimgano[openvino-runtime]"` |
| Create and run OpenVINO artifacts | `pip install "pyimgano[openvino-export]"` |
| Complete creation/runtime toolchain | `pip install "pyimgano[deploy]"` |

`onnx` and `openvino` remain compatibility aliases for the 0.10 release line.
Use the explicit runtime/export profiles for new environments.

## Export a fitted detector from a run

The canonical post-run command is `pyimgano-export`. It restores the persisted
trained state; it never calls `fit()` or silently recreates a fresh detector.

```bash
pyimgano-export \
  --from-run runs/<run_dir> \
  --format native \
  --out ./exports
```

Repeat `--format` to request more than one certified representation:

```bash
pyimgano-export \
  --from-run runs/<run_dir> \
  --format native \
  --format onnx \
  --out ./exports
```

The transaction is strict by default: every requested format must be supported
and verified or no final export directory is published. `--non-strict` permits a
partial result containing only formats that passed. Reference parity is always
mandatory; `--verification-level end-to-end` strengthens it with the available
image corpus. `--overwrite` is required to replace an existing output.

Format support is model- and fitted-state-specific. Inspect the model's
`capabilities.trained_export` cells instead of assuming every detector supports
every format:

```bash
pyimgano-benchmark --model-info <model_name> --json
```

In the current schema-v1 certification scope, `ae_resnet_unet` is the only
full-format trained-artifact target. Two ECOD wrappers also have deliberately
source-locked composite paths. `conditional` means export still requires the
concrete fitted detector, a complete verified checkpoint, its exact local
embedding graph, and the declared dependencies.

| Model | Native | ONNX | TorchScript | OpenVINO |
|---|---|---|---|---|
| `ae_resnet_unet` | `supported` (`native_detector`) | `conditional` (`single_graph`) | `conditional` (`single_graph`) | `conditional` (`single_graph`) |
| `vision_onnx_ecod` | unsupported | `conditional` (`composite`) | unsupported | unsupported |
| `vision_torchscript_ecod` | unsupported | unsupported | `conditional` (`composite`) | unsupported |
| `vision_patchcore` | unsupported | unsupported | unsupported | unsupported |

An ECOD composite packages the exact embedding graph plus a non-executable,
complete fitted ECOD core; it never refits. `vision_onnx_ecod` can only preserve
its ONNX source (including its verified external-data closure), while
`vision_torchscript_ecod` can only preserve its TorchScript source. Schema v1
does not cross-convert these graphs and these ECOD artifacts expose image scores,
not anomaly maps. A custom extractor, a source graph that changed after
checkpoint certification, or an incomplete fitted core fails export.

The release-tested artifact matrix remains Ubuntu x86_64, Python 3.10, and CPU:
ONNX uses `CPUExecutionProvider`; TorchScript, native, and OpenVINO use CPU.
Manifests record the platform on which their export/runtime verification ran;
other project compatibility cells are not release-certified by this gate.

Training and export can also be one operation:

```bash
pyimgano-train \
  --config my_certified_config.json \
  --export-format native \
  --export-verification-level reference-parity
```

For the current reference path, `my_certified_config.json` must set
`model.name` to `ae_resnet_unet` and persist a complete verified checkpoint.

A multi-format export is indexed by `export_index.json` and normally has this
layout:

```text
exports/
├── export_index.json
└── <category>/
    └── <format>/
        ├── artifact_manifest.json
        ├── infer_config.json
        ├── model/ or state/
        └── verification/
```

The older `pyimgano-export-onnx` and `pyimgano-export-torchscript` commands are
embedding/backbone exporters. They do not package the fitted detector or produce
`artifact_manifest.json`; use `pyimgano-export` for trained-detector deployment.

## Run artifact inference from the CLI

An artifact directory or its manifest can be loaded directly:

```bash
pyimgano-infer \
  --artifact ./exports/bottle/native \
  --input ./test_images \
  --save-jsonl ./results.jsonl
```

An export root or deploy bundle can contain several choices. Select one by
category, format, and/or backend:

```bash
pyimgano-infer \
  --artifact ./exports \
  --artifact-category bottle \
  --artifact-format onnx \
  --artifact-backend onnxruntime \
  --input ./test_images \
  --save-jsonl ./results.jsonl
```

`--artifact-id sha256:...` is an exact content selector and cannot be combined
with category, format, or backend selectors. Ambiguous export roots fail and list
the available choices rather than choosing one silently.

For ONNX Runtime, provider order and options are explicit:

```bash
pyimgano-infer \
  --artifact ./exports/bottle/onnx \
  --onnx-providers CUDAExecutionProvider,CPUExecutionProvider \
  --onnx-provider-options '{"CUDAExecutionProvider":{"device_id":"0"}}' \
  --onnx-session-options '{"intra_op_num_threads":4}' \
  --input ./test_images \
  --save-jsonl ./results.jsonl
```

The requested providers must be permitted by the artifact manifest. Provider
options require `--onnx-providers`.

## Load an artifact from Python

`load_artifact()` returns a detector-compatible `ArtifactRuntime`. Use it as a
context manager so temporary verified staging and backend resources are released
deterministically.

```python
from pyimgano.inference import infer, load_artifact

with load_artifact(
    "./exports",
    category="bottle",
    format="native",
) as detector:
    scores = detector.decision_function(["images/a.png", "images/b.png"])
    results = infer(detector, ["images/a.png", "images/b.png"])
```

The runtime also exposes `predict()` and, when declared by the output contract,
`predict_anomaly_map()`. A score-only artifact without an operating threshold
supports `decision_function()` and `infer()` but deliberately rejects
`predict()` until a validated policy is bound.

For an exact ONNX provider configuration:

```python
from pyimgano.inference import load_artifact

with load_artifact(
    "./exports/bottle/onnx",
    providers=[
        {"name": "CUDAExecutionProvider", "options": {"device_id": "0"}},
        "CPUExecutionProvider",
    ],
    session_options={"intra_op_num_threads": 4},
) as detector:
    scores = detector.decision_function(["images/a.png"])
```

All TorchScript artifact layouts require an explicit executable-code trust
decision at load time:

```bash
pyimgano-infer \
  --artifact ./exports/bottle/torchscript \
  --trust-checkpoint \
  --input ./test_images \
  --save-jsonl ./results.jsonl
```

```python
from pyimgano.inference import load_artifact

with load_artifact(
    "./exports/bottle/torchscript",
    trust_checkpoint=True,
) as detector:
    scores = detector.decision_function(["images/a.png"])
```

This applies to both `single_graph` and `composite` TorchScript artifacts.
[PyTorch's `torch.jit.load` documentation](https://docs.pytorch.org/docs/stable/generated/torch.jit.load.html)
warns that untrusted model files can execute arbitrary code during loading. Set
the flag only after independently establishing provenance and integrity; an
artifact hash detects changes but does not make an untrusted graph safe.

Path inputs are decoded by the runtime. Direct NumPy inputs to
`ArtifactRuntime` must be canonical RGB `uint8` HWC arrays; use
`pyimgano.inference.infer(..., input_format=...)` or `infer_bgr()` for other
explicit NumPy formats.

## Import a third-party ONNX model

`load_artifact()` intentionally rejects a raw `.onnx` file. Import it once with a
versioned contract that declares preprocessing and output semantics:

```json
{
  "schema_family": "pyimgano-onnx-import",
  "schema_version": 1,
  "input": {
    "name": "input",
    "dtype": "float32",
    "layout": "NCHW",
    "color_space": "RGB",
    "size": [224, 224],
    "dynamic_batch": true,
    "dynamic_spatial": false,
    "resize": {
      "mode": "stretch",
      "interpolation": "bilinear"
    },
    "scale": {"divisor": 255.0},
    "normalize": {
      "mean": [0.0, 0.0, 0.0],
      "std": [1.0, 1.0, 1.0]
    }
  },
  "outputs": {
    "score": {
      "name": "score",
      "transform": "identity",
      "score_order": "higher_is_more_anomalous"
    }
  }
}
```

Then import and run it:

```bash
pyimgano-artifact import \
  --format onnx \
  --model ./model.onnx \
  --contract ./onnx-contract.json \
  --out ./imported-artifact

pyimgano-infer \
  --artifact ./imported-artifact \
  --input ./test_images \
  --save-jsonl ./results.jsonl
```

The importer checks the graph against the declared input/output names, dtypes,
shapes, and dynamic axes, contains external-data paths, and executes a fresh ONNX
Runtime smoke test. Its verification level is `runtime_smoke`, because a
third-party import has no PyImgAno reference detector for parity comparison.

The score transform may be `identity`, `select_index`, `sigmoid`,
`softmax_select`, or `negate`; index-selection transforms also require `axis` and
`index`. To expose an anomaly map, add this sibling to `outputs`:

```json
"anomaly_map": {
  "name": "anomaly_map",
  "layout": "NCHW",
  "channel": 0,
  "resize_to_source": true
}
```

Without `--policy`, import creates a score-only policy. Supply `--policy
infer-policy.json` during import, or clone the immutable runtime with a validated
policy later:

```bash
pyimgano-artifact bind-policy \
  --artifact ./imported-artifact \
  --policy ./infer-policy.json \
  --out ./production-artifact
```

`pyimgano-artifact inspect ARTIFACT --json` prints the manifest and
`pyimgano-artifact validate ARTIFACT --json` validates its manifest contract.
Loading the artifact additionally verifies referenced files and hashes before
backend construction.

## Package and run a deploy bundle

Request artifact export before deploy-bundle assembly so the bundle receives the
complete artifact roots and indexes them in `bundle_manifest.json.artifact_refs`:

```bash
pyimgano-train \
  --config my_certified_config.json \
  --export-format native \
  --export-infer-config \
  --export-deploy-bundle
```

Here too, the placeholder config denotes the certified `ae_resnet_unet` path;
do not add `--export-format` to an unrelated starter config unless its
`trained_export` capability cell reports support.

Validate and execute the relocatable bundle:

```bash
pyimgano-bundle validate runs/<run_dir>/deploy_bundle --check-hashes --json

pyimgano-bundle run runs/<run_dir>/deploy_bundle \
  --image-dir ./test_images \
  --output-dir ./bundle_run \
  --json
```

When a bundle contains multiple runtime artifacts, pass `--artifact-category`,
`--artifact-format`, or `--artifact-backend` to `run`/`watch`. A bundle with
`artifact_refs` always uses the selected artifact-local policy. Legacy bundles
without runtime artifact references continue to use their root
`infer_config.json` fallback.

Artifacts are relocatable: move the complete artifact root or bundle, not a model
file by itself. Relative component paths, declared sizes, SHA-256 digests,
symlink/path traversal checks, and the artifact ID are verified at load time.
Executable checkpoint deserialization remains disabled by default. Every
TorchScript graph, whether `single_graph` or one component of a `composite`,
requires `--trust-checkpoint` or
`load_artifact(..., trust_checkpoint=True)` after provenance review.

## Runtime support matrix

| Format | Runtime backend | Runtime extra | Creation extra | Important boundary |
|---|---|---|---|---|
| Native | `pyimgano` | Base/model-specific | Base/model-specific | Requires a certified state codec or an explicitly trusted checkpoint |
| ONNX | `onnxruntime` | `onnx-runtime` | `onnx-export` | Raw ONNX requires an explicit import contract |
| TorchScript | `torchscript` | `torch` | `torch` | Executable graph: explicit trust is required for both single-graph and composite loading |
| OpenVINO | `openvino` | `openvino-runtime` | `openvino-export` | Provider flags are not accepted; use the backend device selector |

This table describes runtime plumbing, not universal per-model export support.
The export adapter registry is authoritative for each model and fitted state.
