# Golden MVTec AD benchmark

This repository keeps one intentionally small, repeatable benchmark slice for
the two MVTec AD regimes most likely to expose implementation drift:

- `bottle` — object anomaly detection
- `carpet` — texture anomaly detection
- PatchCore, PaDiM, and OpenCLIP prompt scoring
- image AUROC/AP/F1 plus pixel AUROC/AP/AUPRO when maps are available

MVTec AD is not redistributed by this repository. Download it from the dataset
owner, accept its terms, and place the category directories under one root.

## 1. Reproduce the validated environment

Use a clean Python 3.10 virtual environment:

```bash
python -m venv .venv-golden
. .venv-golden/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -c constraints/optional-py310-current.txt \
  -e '.[dev,torch,onnx,skimage,numba,faiss,diffusion,anomalib,clip]'
python -m pip check
```

The constraints file is a tested compatibility profile, not a claim that newer
dependency combinations are unsupported.

## 2. Validate data and provenance

```bash
python tools/run_golden_mvtec_benchmark.py \
  --root /datasets/mvtec_ad \
  --output-dir runs/golden_mvtec_preflight \
  --check-only
```

This checks both categories and writes `golden_manifest.json` with content-based
dataset fingerprints and package/platform versions. Images and ground-truth
masks are included in the fingerprint.

## 3. Run the benchmark

```bash
python tools/run_golden_mvtec_benchmark.py \
  --root /datasets/mvtec_ad \
  --output-dir runs/golden_mvtec \
  --device cuda \
  --allow-download
```

`--allow-download` is mandatory because the benchmark uses official pretrained
weights and may need to fetch them. It is an explicit acknowledgement, not a
license grant. The output contains:

- `golden_manifest.json`
- `golden_summary.json`
- `golden_summary.csv`

For a non-publishable pipeline smoke test, add `--smoke`. Smoke output is marked
`publication_ready: false`.

Do not compare results when dataset fingerprints, resize, model settings, or the
named dependency profile differ. A successful preflight is not a paper-metric
reproduction; only a completed full run produces numerical evidence.
