# Deep Models Status (Recommended vs Experimental)

This repo contains a mix of deep anomaly detection implementations.

Some are **contract-aligned** and suitable for production-ish pipelines, while
others are kept mainly for research exploration and may have rough edges
(dependencies, speed, import-time cost, missing tiny mode, etc.).

This document is a living status page intended to help industrial users pick a
safe starting point.

## Recommended (Contract-Aligned, Test-Covered)

These models are intended to:

- follow `BaseVisionDeepDetector` semantics (`fit`, `decision_function`)
- avoid implicit weight downloads by default
- support `tiny=True` for unit tests (where applicable)

Recommended:

- `ae_resnet_unet` (lightweight conv AE reconstruction baseline)
- `vae_conv` (lightweight conv VAE reconstruction baseline)

## Industrial Baselines (Not End-to-End “Deep”, But Recommended)

These are often the best first-line industrial route:

- `vision_embedding_core` (embedding extractor + classical `core_*` detector)
- `vision_patchcore_lite` (embedding + memory bank NN distance, image-level)
- `vision_padim_lite` (embedding + Gaussian stats, image-level)
- `vision_student_teacher_lite` (teacher/student embedding regression residual)

They are easy to calibrate/ensemble and integrate well with manifests + synthesis.

## Experimental / Legacy / Research

Models in this group may:

- require extra dependencies or checkpoints
- be heavy (slow training/inference)
- not fully conform to the deep contract yet
- change behavior more often

Examples (non-exhaustive):

- various flow/distillation/diffusion/foundation-style models under `pyimgano/models/`
- models that depend on third-party backends (`anomalib`, `patchcore-inspection`, etc.)

When using these in industrial settings, prefer:

- explicit checkpoints
- CPU-only smoke validation first
- `--no-pretrained` / offline-safe settings

## How To Validate A Deep Model Quickly

Recommended minimal checks:

1. Construct with `device="cpu"`, `pretrained=False` (or equivalent).
2. Fit on 2–4 tiny images (`tiny=True` if supported).
3. Call `decision_function` on 1–2 images and ensure:
   - output shape `(N,)`
   - finite float scores
4. Ensure no network downloads occur (torchvision weight guard tests).

