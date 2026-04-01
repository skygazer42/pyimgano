# CLI Reference

PyImgAno provides the following CLIs:

- `pyimgano` — top-level umbrella CLI (`pyimgano --list`, `pyimgano list models`, `pyimgano train --help`)
- `pyim` — unified discovery shortcut for models, families, presets, and preprocessing schemes
- `pyimgano-benchmark` — one-click industrial benchmarking + run artifacts
- `pyimgano-runs` — inspect and compare saved benchmark/workbench runs
- `pyimgano-demo` — minimal offline demo (creates a tiny custom dataset + runs a suite/sweep)
- `pyimgano-train` — recipe-driven workbench runs (adaptation-first; optional micro-finetune)
- `pyimgano-infer` — JSONL inference over images/videos (path-driven)
- `pyimgano-defects` — standalone anomaly-map → mask → regions defects export
- `pyimgano-robust-benchmark` — robustness evaluation (clean + corruptions)
- `pyimgano-doctor` — environment + optional dependency (extras) sanity check
- `pyimgano-weights` — local weights/checkpoints manifest validation + hashing (never downloads)
- `pyimgano-manifest` — generate a JSONL manifest from a `custom`-layout dataset tree
- `pyimgano-datasets` — dataset converter discovery + metadata
- `pyimgano-synthesize` — anomaly synthesis + manifest generation
- `pyimgano-validate-infer-config` — validate an exported `infer_config.json` before deployment
- `pyimgano-features` — feature/embedding extractor discovery + extraction utilities
- `pyimgano-export-torchscript` — export a torchvision backbone to TorchScript (offline-safe by default)
- `pyimgano-export-onnx` — export a torchvision backbone to ONNX (offline-safe by default)

---

## `pyimgano`

`pyimgano` is the new top-level convenience entrypoint. It keeps the old
specialized CLIs intact, but gives users one command for discovery and command
navigation.

Common usage:

```bash
pyimgano --help
python -m pyimgano --help
pyimgano --list
pyimgano list models
pyimgano -- list models --json
pyimgano --list models --family patchcore
pyimgano train --help
pyimgano runs quality /path/to/run_dir --json
pyimgano runs acceptance /path/to/run_or_suite_export --json
```

Notes:

- Top-level `--list ...` forwards to the same discovery surface as `pyim --list ...`.
- `python -m pyimgano --help` is equivalent to the root CLI and is useful in environments where console scripts are not on `PATH`.
- `pyimgano --help` now includes a small guided workflow grouped into `discover`, `benchmark`, `train`, `infer`, `validate`, and `gate` stages.
- `pyimgano list ...` and `pyimgano -- list ...` are equivalent discovery aliases when you want a more shell-like command form.
- `pyimgano help COMMAND` is a shortcut for `pyimgano COMMAND --help`.
- Existing `pyimgano-*` entry points are still supported and remain the stable low-level interface.
- `pyim` is still available as the shorter discovery-only alias.

## `pyimgano-doctor`

`pyimgano-doctor` prints a lightweight environment report and checks which
optional extras are available.

Common usage:

```bash
pyimgano-doctor
pyimgano-doctor --json
pyimgano-doctor --profile first-run --json
pyimgano-doctor --profile benchmark --dataset-target ./_demo_custom_dataset --json
pyimgano-doctor --profile deploy --run-dir runs/<run_dir> --json
pyimgano-doctor --profile publish --publication-target /path/to/suite_export --json
pyimgano-doctor --suite industrial-v4 --json   # show which suite baselines will be skipped
pyimgano-doctor --require-extras torch,skimage --json   # CI/deploy gate: exit 1 if missing
pyimgano-doctor --recommend-extras --for-command export-onnx --json
pyimgano-doctor --recommend-extras --for-model vision_openclip_patch_map --json
pyimgano-doctor --accelerators --json   # runtime checks: torch CUDA/MPS, onnxruntime providers, openvino devices
pyimgano-doctor --run-dir /path/to/run_dir --json   # evaluate run readiness / acceptance-adjacent state
pyimgano-doctor --deploy-bundle /path/to/deploy_bundle --json   # validate deploy bundle readiness
```

Notes:
- `--profile first-run|benchmark|deploy|publish` emits a guided workflow profile and a readiness summary for that operator path.
- `--require-extras` accepts comma-separated values and is repeatable.
- `--recommend-extras` can be paired with `--for-command` or `--for-model` to turn extras discovery into a copy-pasteable install hint.
- For `--for-command benchmark`, the recommendation payload also surfaces `starter_configs`, `optional_baseline_count`, `starter_list_command`, and `starter_info_command` so you can see how much of the starter suite is gated behind optional extras and jump straight to the next benchmark command.
- Command recommendations also carry a `workflow_stage` hint so automation or UI layers can place the recommendation inside the broader `discover / benchmark / train / infer / validate / gate` flow.
- For `--for-command train|infer|runs`, the recommendation payload also surfaces `suggested_commands` so the text/JSON output tells you the next concrete command to run.
- Command recommendations also surface `next_step_commands` for the likely follow-up stage after the current command succeeds.
- For `--for-command train|infer|runs`, the payload also includes `artifact_hints` so the operator can see which files or directories are expected to matter after the command runs.
- For `--for-model NAME`, the recommendation payload now also includes `workflow_stage=discover`, `supports_pixel_map`, `tested_runtime`, and a `model_info_command` plus suggested follow-up commands.
- Export recommendations follow the same structure, including example outputs like `embed.onnx` and `embed.ts` plus the likely follow-up infer recommendation.
- When `--json` is set, the tool still prints JSON on missing extras, but exits with code `1`.
- `--accelerators` is best-effort and opt-in; it never raises, it only reports missing runtimes + install hints.
- `--run-dir` and `--deploy-bundle` surface readiness payloads for deployment-oriented checks without changing the underlying run/bundle artifacts.
- `--publication-target` accepts a suite export directory or `leaderboard_metadata.json` path and evaluates publication readiness through the same trust gate used by `pyimgano-runs publication`.

## `pyimgano-demo`

`pyimgano-demo` is a minimal **offline-safe** end-to-end smoke demo:

- writes a tiny `custom`-layout dataset under `--dataset-root`
- runs a baseline suite (and optional sweep) over it
- optionally runs a one-command **infer + defects** loop (no need to manually run `pyimgano-infer`)

Common usage:

```bash
pyimgano-demo
pyimgano-demo --smoke --summary-json /tmp/pyimgano_demo_summary.json --emit-next-steps
pyimgano-demo --scenario benchmark --summary-json /tmp/pyimgano_benchmark_summary.json
pyimgano-demo --scenario infer-defects --summary-json /tmp/pyimgano_infer_defects_summary.json
pyimgano-demo --export none --no-sweep
pyimgano-demo --infer-defects --export none --no-sweep   # writes <suite_dir>/infer/results.jsonl + masks/ + overlays/ + regions.jsonl
```

Notes:

- `--smoke` clamps the demo to a lightweight CPU-friendly path.
- `--scenario smoke|benchmark|infer-defects` applies a named preset so operators can rerun the same starter path without memorizing individual flags.
- `--summary-json PATH` writes a compact machine-friendly summary with `run_dir`, exported files, and suggested follow-up commands.
- `--emit-next-steps` prints a short copy-pasteable block after the demo completes.

## `pyim`

`pyim` is a lightweight discovery-first shortcut for the most common "what can I run?" questions.
It complements the heavier CLIs by exposing models, curated families, model presets, defects presets,
feature extractors, and named preprocessing schemes through one entrypoint.

If you want one top-level command for both discovery and execution, use `pyimgano`.
If you want the shortest discovery alias, keep using `pyim`.

Common usage:

```bash
pyim --list
pyim --goal first-run --json
pyim --goal deployable
pyim --list models --family patchcore
pyim --list models --year 2021 --type deep-vision
pyim --list models --type flow-based
pyim --list model-presets --family graph
pyim --list years --json
pyim --list types --json
pyim --list metadata-contract --json
pyim --audit-metadata --json
pyim --list preprocessing --deployable-only
pyim --list families --json
```

Notes:

- `--list` accepts: `all`, `models`, `families`, `types`, `years`, `features`, `model-presets`, `defects-presets`, `preprocessing`.
- `--list metadata-contract` prints the structured model metadata contract used by discovery and audits.
- `--audit-metadata` audits registry models against the metadata contract and returns a non-zero exit code when issues are present.
- `--family NAME` filters model and model-preset discovery using curated families or raw registry tags.
- `--type NAME` filters model discovery using curated high-level types such as `deep-vision`, `flow-based`, `one-class-svm`, `classical-core`, or raw registry tags.
- `--year VALUE` filters model discovery by publication year or timeline buckets such as `pre-2001` and `unknown`.
- `--tags a,b --tags c` works for model and feature discovery and is repeatable.
- `--deployable-only` restricts preprocessing output to infer/workbench-safe presets.
- `--goal first-run|cpu-screening|pixel-localization|deployable` emits task-oriented picks across models, recipes, and datasets in one payload.
- `--objective`, `--selection-profile`, and `--topk` add starter-pick guidance for `--list models` without changing the default discovery shape for other list kinds.
- In text output, `pyim` renders a `Selection Context` block ahead of starter picks so the chosen objective/profile/topk are visible in the terminal transcript.
- In text output, `pyim --goal ...` also renders `Goal Context` and `Goal Picks` blocks so the operator can see the chosen route and the concrete model/recipe/dataset recommendations together.
- When available, `pyim` also renders a `Suggested Commands` block with the next inspection commands for the top pick (for example `pyimgano-doctor --recommend-extras --for-model ...` and `pyimgano-benchmark --model-info ...`).
- In text output, starter picks now show compact hints like `runtime=numpy`, `pixel_map=yes|no`, `family=...`, `why=...`, and an install hint when extras are required.
- When starter picks are present in `--json` output, each pick includes lightweight deployment hints such as `supports_pixel_map`, `tested_runtime`, `deployment_family`, and `why_this_pick`.
- `--json` prints machine-friendly JSON payloads instead of text blocks.
- See `docs/MODEL_METADATA_CONTRACT.md` for field semantics and audit policy.

## `pyimgano-benchmark`

### Common Usage

```bash
pyimgano-benchmark \
  --dataset mvtec \
  --root /path/to/mvtec_ad \
  --category bottle \
  --model vision_patchcore \
  --preset industrial-balanced \
  --pretrained \
  --device cuda
```

Notes:
- CLIs default to **offline-safe** behavior (`--no-pretrained`). Use `--pretrained` explicitly when you want upstream weights (may download).
- `--model` can be either a registered model name (e.g. `vision_patchcore`) or a **model preset name**
  (e.g. `industrial-pixel-mad-map`). To discover presets, use `pyimgano-infer --list-model-presets`.

### Config Files (`--config`)

For reproducible benchmark runs, you can load flags from a JSON config:

```bash
pyimgano-benchmark --config benchmarks/configs/official_mvtec_industrial_v4_cpu_offline.json
pyimgano-benchmark --config official_mvtec_industrial_v4_cpu_offline.json
```

Rules:
- Config values are applied first.
- Explicit CLI flags override config values.
- Official configs can be discovered with `--list-official-configs` and inspected with
  `--official-config-info NAME --json`.
- Starter configs can be discovered with `--list-starter-configs` and inspected with
  `--starter-config-info NAME --json`.
- When the config name matches a built-in official preset, `--config` accepts the bare
  filename (for example `official_mvtec_industrial_v4_cpu_offline.json`) in addition to a full path.
- When `--config` is used, saved benchmark reports are stamped with a `benchmark_config`
  metadata block describing the source config and whether it matches an `official_*.json` preset.
- Config JSON can be:
  - a JSON object: keys are argparse dest names (example: `suite_sweep_max_variants`)
  - a JSON list of argv tokens: exact flags (example: `["--dataset","mvtec", ...]`)

### Discovery

- List models: `pyimgano-benchmark --list-models`
- Filter models by family/type/year: `pyimgano-benchmark --list-models --family patchcore`, `pyimgano-benchmark --list-models --type one-class-svm --year 2001`
- Load third-party plugins (entry points) before discovery: `pyimgano-benchmark --plugins --list-models`
- Filter models by tags: `pyimgano-benchmark --list-models --tags vision,deep`
- Model info (constructor signature + accepted kwargs): `pyimgano-benchmark --model-info vision_patchcore`
- List dataset categories: `pyimgano-benchmark --list-categories --dataset mvtec --root /path/to/mvtec_ad`
- List manifest categories: `pyimgano-benchmark --list-categories --dataset manifest --manifest-path /path/to/manifest.jsonl`
- List curated industrial baseline suites: `pyimgano-benchmark --list-suites`
- Suite contents (resolved baselines): `pyimgano-benchmark --suite-info industrial-v1`
- List curated suite sweep profiles (small grid searches): `pyimgano-benchmark --list-sweeps`
- Sweep contents (variants + overrides): `pyimgano-benchmark --sweep-info industrial-small`
- List official benchmark config presets: `pyimgano-benchmark --list-official-configs`
- Show official config metadata: `pyimgano-benchmark --official-config-info official_mvtec_industrial_v4_cpu_offline.json --json`
- List starter benchmark config presets: `pyimgano-benchmark --list-starter-configs`
- Show starter config metadata: `pyimgano-benchmark --starter-config-info official_mvtec_industrial_v4_cpu_offline.json --json`
- Starter config metadata also includes `starter_tier`, `optional_extras`, `optional_baseline_count`, `starter_info_command`, `starter_run_command`, and an `optional_extras_install_hint` so users know which optional backends broaden the suite, how the preset is positioned, and which command to run next.

### Baseline Suites (Industrial)

Suites are curated packs of **multiple model presets** intended for industrial algorithm selection.

Built-in suites: `industrial-ci`, `industrial-v1`, `industrial-v2`, `industrial-v3`, `industrial-v4` (use `--list-suites` for the full list).

Example:

```bash
pyimgano-benchmark \
  --dataset mvtec \
  --root /path/to/mvtec_ad \
  --category bottle \
  --suite industrial-v1 \
  --device cpu \
  --no-pretrained
```

Flags:

- `--suite NAME` — run a curated suite (instead of a single `--model`)
- `--suite-max-models N` — limit number of baselines (smoke/debug)
- `--suite-include NAME[,NAME]` — run only selected suite baselines (comma-separated or repeatable)
- `--suite-exclude NAME[,NAME]` — skip selected suite baselines (comma-separated or repeatable)
- `--suite-continue-on-error/--no-suite-continue-on-error` — keep running when a baseline errors or has missing optional deps
- `--suite-export csv|md|both` — write `leaderboard.*`, `best_by_baseline.*`, and `skipped.*` tables into the suite output directory (requires `--save-run`)
- `--suite-export-best-metric NAME` — metric used for `best_by_baseline.*` tables (default: `auroc`). Pixel metrics require `--pixel`.
- `--suite-sweep SPEC` — run a small parameter sweep (grid search) per baseline and rank variants.
  `SPEC` can be a built-in sweep name (discover with `--list-sweeps`), a JSON file path, or inline JSON.
- `--suite-sweep-max-variants N` — cap the number of sweep variants per baseline (excluding base). Example: `--suite-sweep-max-variants 1`

Suite artifacts (when `--save-run` is enabled):

- `<suite_dir>/report.json` — aggregated suite report (ranking + skipped baselines)
- `<suite_dir>/config.json`, `<suite_dir>/environment.json`
- `<suite_dir>/leaderboard_metadata.json` — compact benchmark metadata payload (suite, dataset, config provenance, environment fingerprint, run-artifact audit refs/digests, and exported leaderboard file digests)
  - official benchmark presets also stamp a small `citation` payload for publication workflows
- `<suite_dir>/leaderboard.csv`, `<suite_dir>/skipped.csv` (when `--suite-export csv|both`)
- `<suite_dir>/best_by_baseline.csv` (when `--suite-export csv|both`, best variant per baseline by AUROC; most useful with `--suite-sweep`)
- `<suite_dir>/leaderboard.md`, `<suite_dir>/skipped.md` (when `--suite-export md|both`)
- `<suite_dir>/best_by_baseline.md` (when `--suite-export md|both`, best variant per baseline by AUROC; most useful with `--suite-sweep`)
- `<suite_dir>/models/<baseline_name>/...` — per-baseline run artifacts
- `<suite_dir>/models/<baseline_name>/variants/<variant>/...` (when `--suite-sweep` is enabled)

Optional extras:

- Some suite entries are marked optional and are **skipped** when extras are not installed.
  Skip reasons include actionable install hints like `pip install 'pyimgano[skimage]'` or `pip install 'pyimgano[torch]'`.

#### Custom sweep JSON

You can pass a JSON sweep plan file to `--suite-sweep` (or prefix the path with `@`):

```json
{
  "name": "my-sweep",
  "description": "Tiny sweep for NCC window sizes",
  "variants_by_entry": {
    "industrial-template-ncc-map": [
      {"name": "win_7", "override": {"window_hw": [7, 7]}},
      {"name": "win_21", "override": {"window_hw": [21, 21]}}
    ]
  }
}
```

Run it:

```bash
pyimgano-benchmark \
  --dataset mvtec \
  --root /path/to/mvtec_ad \
  --category bottle \
  --suite industrial-ci \
  --suite-sweep ./my_sweep.json \
  --no-pretrained
```

## `pyimgano-runs`

Inspect and compare saved benchmark/workbench runs:

```bash
pyimgano-runs list --root runs
pyimgano-runs list --root runs --kind suite --dataset mvtec
pyimgano-runs list --root runs --min-quality reproducible --json
pyimgano-runs list --root runs --same-environment-as runs/run_a --json
pyimgano-runs list --root runs --same-target-as runs/run_a --json
pyimgano-runs list --root runs --kind robustness --same-robustness-protocol-as runs/run_a --json
pyimgano-runs latest --root runs --json
pyimgano-runs latest --root runs --min-quality audited --json
pyimgano-runs latest --root runs --same-environment-as runs/run_a --json
pyimgano-runs latest --root runs --same-target-as runs/run_a --json
pyimgano-runs latest --root runs --kind robustness --same-robustness-protocol-as runs/run_a --json
pyimgano-runs compare runs/run_a runs/run_b --json
pyimgano-runs compare runs/run_a runs/run_b --baseline runs/run_a --metric auroc --fail-on-regression --json
pyimgano-runs compare runs/run_a runs/run_b --baseline runs/run_a --require-same-split --json
pyimgano-runs compare runs/run_a runs/run_b --baseline runs/run_a --require-same-target --json
pyimgano-runs compare runs/run_a runs/run_b --baseline runs/run_a --require-same-environment --json
pyimgano-runs compare runs/run_a runs/run_b --baseline runs/run_a --require-same-robustness-protocol --json
pyimgano-runs acceptance /path/to/run_dir --require-status audited --json
pyimgano-runs acceptance /path/to/suite_export --json
pyimgano-runs publication /path/to/suite_export --json
```

See also: `docs/RUN_COMPARISON.md`

Notes:

- JSON output includes `split_comparison` metadata when comparing runs.
- JSON output includes `target_comparison` metadata when comparing runs.
- JSON output includes `environment_comparison` metadata when comparing runs.
- JSON output includes `robustness_protocol_comparison` when comparing robustness runs against a baseline.
- JSON output includes `trust_comparison` when a baseline is present so automation can
  read the normalized baseline trust gate/status/reason/degraded-by/audit refs
  without reconstructing them from `baseline_run.artifact_quality.trust_summary`.
- JSON `summary` for `compare` now also includes machine-readable
  `regression_gate`, `comparability_gates`, `blocking_flags`, `verdict`,
  `primary_metric*`, `primary_metric_statuses`, `primary_metric_deltas`, and
  `trust_*`, `candidate_verdicts`, `candidate_blocking_reasons`, and
  `candidate_comparability_gates` fields so CI can consume the same gate
  decision and baseline trust posture the plain-text output prints.
  When no `--baseline` is provided, `summary.baseline_checked=false`,
  `summary.regression_gate=unchecked`, and `summary.verdict=informational`.
- Plain-text `list` / `latest` output now also includes `quality=...`, `trust=...`,
  `primary_metric=...`, and the first `reason=...` so operators can spot reproducible
  vs audited runs and the main comparison metric without opening JSON.
- Plain-text `compare` output also includes `comparison_primary_metric=...`,
  baseline `quality` / `trust`, and per-candidate primary-metric status / delta
  plus a single `comparability_gates: ...` line before the deeper split, target,
  environment, and robustness compatibility blocks.
- The leading per-run lines in plain-text `compare` are also structured briefs now:
  each run shows `quality=...`, `trust=...`, `primary_metric=...`, and
  `primary_metric_status=...` instead of dumping the raw metric dict.
- Plain-text `compare` also prints `comparison_regression_gate=...`,
  `comparison_blocking_flags=...`, and `comparison_verdict=pass|blocked` so the
  operator can see which strict CLI flags would fail before reading the details.
- Plain-text `compare` also prints per-candidate `candidate_verdict.<run>=...`,
  `candidate_blocking_reasons.<run>=...`, and
  `candidate_comparability_gates.<run>=...` lines so shell logs can show
  exactly which run is blocked and whether it drifted on split, environment,
  target, or robustness protocol.
- Plain-text `compare` also prints `comparison_trust_gate=trusted|limited`,
  `comparison_trust_status=...`, `comparison_trust_reason=...`,
  `comparison_trust_degraded_by=...`, and `comparison_trust_ref.*=...` to
  separate “the gates pass” from “the baseline run is trusted enough for this
  comparison result to carry weight”.
- `--min-quality LEVEL` filters `list` and `latest` to runs whose `artifact_quality.status`
  is at least `reproducible`, `audited`, or `deployable`.
- `--same-environment-as RUN` filters `list` and `latest` to runs whose `environment.json`
  fingerprint matches the reference run.
- `--same-target-as RUN` filters `list` and `latest` to runs whose checked `dataset` /
  `category` target matches the reference run.
- `--same-robustness-protocol-as RUN` filters `list` and `latest` to robustness runs whose corruption mode, condition set, severity schedule, input mode, and resize match the reference run.
- `pyimgano-runs quality` validates the top-level `artifacts/calibration_card.json` when present, so an invalid threshold-audit card keeps a run at `reproducible` instead of `audited`.
- `pyimgano-runs quality` also inspects optional `deploy_bundle/model_card.json` and `deploy_bundle/weights_manifest.json`;
  if either file is present but invalid, the run does not qualify as `deployable`.
- `pyimgano-runs quality --require-status reproducible|audited|deployable` returns exit code `1`
  unless the run reaches at least the requested artifact quality level.
- `pyimgano-runs acceptance` is the aggregated handoff gate: it auto-detects run directories vs suite exports, reuses `quality` for runs, validates the selected `infer_config.json` (preferring `deploy_bundle/` when present), and only blocks on bundle weight metadata when `model_card.json` or `weights_manifest.json` actually exist.
- `pyimgano-runs publication` inspects `leaderboard_metadata.json` and returns exit code `1`
  unless the export is publication-ready; that now requires benchmark provenance
  (`benchmark_config.source` + `sha256`), `evaluation_contract`, `citation`,
  `audit_refs.report_json|config_json|environment_json`,
  matching `audit_digests.report_json|config_json|environment_json`, and any
  declared exported file digests to match the on-disk files, plus any
  `model_card_json` / `weights_manifest_json` exports to validate cleanly.
  Keep it when you want the dedicated suite-only alias.
- `--require-same-split` returns exit code `1` when the baseline split fingerprint is missing or any non-baseline run does not match it.
- `--require-same-target` returns exit code `1` when the baseline dataset/category is unavailable for checking or any non-baseline run does not match it.
- `--require-same-environment` returns exit code `1` when the baseline environment fingerprint is unavailable for checking or any non-baseline run does not match it.
- `--require-same-robustness-protocol` returns exit code `1` when the baseline robustness protocol is unavailable for checking or any non-baseline run changes corruption mode, conditions, severities, input mode, or resize.
- For robustness runs, the comparison contract defaults to the worst-condition metric surface (`worst_corruption_auroc`, then related robustness metrics) instead of the plain clean `auroc` view, so regression gates line up with drift robustness instead of only clean accuracy.

### Manifest dataset

Benchmark a manifest JSONL (paths mode):

```bash
pyimgano-benchmark \
  --dataset manifest \
  --root /path/to/fallback_root \
  --manifest-path /path/to/manifest.jsonl \
  --category bottle \
  --model vision_patchcore \
  --pretrained \
  --device cuda
```

Optional split policy knobs:

- `--manifest-test-normal-fraction 0.2`
- `--manifest-split-seed 123` (defaults to `--seed` or 0)

### Run Artifacts

By default, run artifacts are written to `runs/<timestamp>_<dataset>_<model>/`:

- `report.json`
- `config.json`
- `environment.json`
- `categories/<category>/report.json`
- `categories/<category>/per_image.jsonl`

Key flags:

- `--output-dir DIR` — write artifacts to a specific directory
- `--save-run/--no-save-run` — enable/disable artifact writing
- `--per-image-jsonl/--no-per-image-jsonl` — enable/disable per-image JSONL

### Inputs

- `--input-mode paths|numpy`
  - `paths`: pass file paths to detectors (default)
  - `numpy`: decode images into memory first (for numpy-first models)

### Reproducibility

- `--seed INT` — best-effort deterministic seeding (also passed as `random_seed/random_state` when supported)

### Threshold Calibration

- Default strategy: calibrate `threshold_` as a quantile of **train/normal** scores.
- Default quantile: `1 - contamination` when available, else `0.995`.
- Override quantile: `--calibration-quantile Q`
- Run artifacts include `threshold_provenance` (quantile + where it came from).

### Model Persistence (Classical Detectors Only)

- Save detector after fit: `--save-detector [PATH|auto]`
  - `auto` writes `<output-dir>/detector.pkl` (or `runs/.../detector.pkl` if `--output-dir` is omitted)
- Load detector and skip fitting: `--load-detector PATH`

Security note: never load pickle files from untrusted sources.

### Feature Cache (Path Inputs, Classical Detectors)

- `--cache-dir DIR` — cache extracted feature vectors on disk (speeds up repeated scoring)

---

## `pyimgano-infer`

Runs inference and emits one JSON record per input (stdout or JSONL file).

Example:

```bash
pyimgano-infer \
  --model vision_padim \
  --train-dir /path/to/train/good \
  --input /path/to/inputs \
  --save-jsonl out.jsonl
```

Notes:

- When `--train-dir` is provided and the detector does **not** set `threshold_` during `fit()`,
  `pyimgano-infer` auto-calibrates `threshold_` from train scores (same default quantile as
  `pyimgano-benchmark`: `1 - contamination` when available, else `0.995`).
- Pass `--calibration-quantile Q` to override the quantile explicitly.

### Model Presets (Shortcuts)

Presets are just **named (model + kwargs)** pairs that keep industrial command lines short while staying reproducible.

- List presets: `pyimgano-infer --list-model-presets`
- List models by publication year and type: `pyimgano-infer --list-models --year 2021 --type deep-vision`
- List classical one-class SVM style models from a verified publication year: `pyimgano-infer --list-models --year 2001 --type one-class-svm`
- Filter preset discovery by family/tag: `pyimgano-infer --list-model-presets --family graph`
- JSON preset discovery returns metadata (`name`, `model`, `kwargs`, `requires_extras`, `tags`):
  `pyimgano-infer --list-model-presets --family distillation --json`
- Show preset details (model/kwargs/description): `pyimgano-infer --model-preset-info industrial-pixel-mad-map`
- For a unified discovery view across models, presets, and preprocessing schemes, use `pyim --list`.

Example:

```bash
pyimgano-infer \
  --model-preset industrial-pixel-mad-map \
  --train-dir /path/to/train/normal \
  --input /path/to/inputs \
  --include-maps \
  --save-maps /tmp/pyimgano_maps \
  --save-jsonl out.jsonl
```

### Deployable Preprocessing Presets

For numpy-capable inference routes, `pyimgano-infer` can apply a named deployable preprocessing preset
before scoring. This is useful when you want consistent illumination/contrast normalization without
copying a long list of low-level knobs into every command.

- Discover preprocessing schemes: `pyim --list preprocessing --deployable-only`
- Apply a preprocessing preset directly:

```bash
pyimgano-infer \
  --model vision_patchcore \
  --preprocessing-preset illumination-contrast-balanced \
  --input /path/to/inputs
```

Notes:

- Current CLI support is limited to `preprocessing.illumination_contrast` presets.
- These presets require a numpy-capable model path. If the selected detector cannot consume numpy inputs,
  the CLI reports `PREPROCESSING_REQUIRES_NUMPY_MODEL`.

### Deployment-Friendly Embeddings

If you want the “embedding + core” industrial route without relying on upstream model registries at inference time,
use one of the deployment wrapper models and pass `--checkpoint-path`:

- TorchScript: `vision_torchscript_ecod`, `vision_torchscript_knn_cosine_calibrated`, ...
- ONNX Runtime: `vision_onnx_ecod`, `vision_onnx_knn_cosine_calibrated`, ...

Example (ONNX embeddings + ECOD):

```bash
pyimgano-infer \
  --model vision_onnx_ecod \
  --checkpoint-path /path/to/resnet18_embed.onnx \
  --train-dir /path/to/train/normal \
  --input /path/to/inputs \
  --save-jsonl out.jsonl
```

Optional:

- `--include-maps` + `--save-maps DIR` — write anomaly maps as `.npy`
- High-resolution tiling (optional; for 2K/4K inspection images):
  - `--tile-size N` — run tiled inference (wraps the detector in `TiledDetector`)
  - `--tile-stride N` — tile overlap stride (default: tile-size; smaller = more overlap = fewer seams)
  - `--tile-map-reduce max|mean|hann|gaussian` — blend overlapping tile maps (`hann`/`gaussian` reduce seams)
  - `--tile-score-reduce max|mean|topk_mean` + `--tile-score-topk` — aggregate tile scores into an image score
- `--seed INT` — best-effort deterministic seeding (also passed as `random_seed/random_state` when supported)
- `--batch-size N` — run inference in chunks (preserves output order; can reduce peak memory)
- ONNX Runtime CPU tuning (ONNX routes only):
  - `--onnx-session-options JSON` — pass onnxruntime `SessionOptions` knobs without nested `--model-kwargs`
    - Example: `--onnx-session-options '{"intra_op_num_threads":8,"inter_op_num_threads":1,"execution_mode":"sequential","graph_optimization_level":"all"}'`
  - `--onnx-sweep` — run a small grid search over `(intra_op_num_threads, graph_optimization_level)` and apply the best config
    - Optional knobs: `--onnx-sweep-intra 1,2,4,8`, `--onnx-sweep-opt-levels all,extended`, `--onnx-sweep-repeats N`, `--onnx-sweep-samples N`
    - `--onnx-sweep-json PATH` writes a machine-friendly sweep report (timings + chosen best)
- `--profile` — print stage timing summary to stderr (load model, fit/calibrate, infer, artifacts)
- `--profile-json PATH` — write a JSON profile payload (stable, machine-friendly)
- `--amp` — best-effort AMP/autocast for torch-backed models (requires torch + CUDA; otherwise runs without AMP)
- `--include-confidence` — include `label_confidence` in output records when the detector exposes confidence helpers
  - `label_confidence` is the confidence of the predicted binary label on `[0, 1]`, not an anomaly probability
  - best-effort: unsupported detectors simply omit the field
- `--reject-confidence-below FLOAT` — rewrite low-confidence predictions to a reject label
  - requires detector confidence support and threshold-based labels
  - rejection also emits `label_confidence`, `rejected`, and a stable `decision_summary` block
- Runtime metadata:
  - success records may also include a stable `postprocess_summary` block describing whether maps were requested/enabled, whether runtime postprocess ran, and which source supplied map/threshold defaults
  - this is especially useful for `--infer-config` and `--from-run`, where deploy defaults are restored before CLI overrides are applied
  - direct CLI runs can also emit `postprocess_summary` when maps/defects/postprocess/tiling or prediction-policy knobs materially affect runtime behavior
- `--reject-label INT` — label value used for rejected samples (default: `-2`; requires `--reject-confidence-below`)
- When running with `--infer-config` or `--from-run`, exported `prediction.reject_confidence_below`
  and `prediction.reject_label` are used as defaults when present. Explicit CLI flags override them.
- `--continue-on-error` — best-effort production mode: record per-input errors and keep going (exit code 1 if any errors)
- `--max-errors N` — stop early after N errors (only with `--continue-on-error`)
- `--flush-every N` — flush JSONL outputs every N records (stability vs performance)
  - Python service integrations also receive a `triage_summary` aggregate from `run_continue_on_error_inference(...)`
- `--include-anomaly-map-values` — embed raw anomaly-map values in JSONL (debug only; very large output)
- `--defects` — export industrial defect structures (binary mask + connected-component regions)
  - `--defects-preset industrial-defects-fp40` — FP reduction defaults (ROI/border/smoothing/hysteresis/shape filters)
  - `--defects-regions-jsonl PATH` — write per-image regions payloads to a dedicated JSONL file
  - `--save-masks DIR` + `--mask-format png|npy|npz` (`npz` is compressed numpy; good for large batches)
  - `--defects-mask-space roi|full` — when ROI is set, export ROI-only or full-size masks (regions are always ROI-gated)
  - `--defects-mask-dilate INT` — optional mask dilation for industrial fill/coverage
  - `--save-overlays DIR` — save per-image debugging overlays (original + heatmap + mask outline/fill)
  - `--defects-image-space` — add `bbox_xyxy_image` to regions when image size is available
  - Pixel threshold options:
    - `--pixel-threshold FLOAT` + `--pixel-threshold-strategy fixed`
    - `--pixel-threshold-strategy infer_config` (uses `defects.pixel_threshold` from `infer_config.json` / a workbench run)
    - `--pixel-threshold-strategy normal_pixel_quantile` (requires `--train-dir`; uses `--pixel-normal-quantile`)
      - If selected and `--train-dir` is provided, `pyimgano-infer` recalibrates from normal/train maps even if `infer_config.json` contains `defects.pixel_threshold`.
  - When running with `--infer-config` or `--from-run`, the exported `defects.*` settings are used as defaults
    (ROI, morphology, min-area, mask format, max regions, pixel threshold strategy/quantile, etc.). CLI flags override.
  - When running with `--infer-config` or `--from-run`, exported preprocessing defaults (e.g. `preprocessing.illumination_contrast`)
    are applied automatically for deploy consistency (when present).
    - Note: `preprocessing.illumination_contrast` requires a numpy-capable model (tag: `numpy`), otherwise you’ll see
      `PREPROCESSING_REQUIRES_NUMPY_MODEL`.
  - `--roi-xyxy-norm x1 y1 x2 y2` (optional; gates defects output only)
    - If ROI is set and you calibrate pixel threshold via `normal_pixel_quantile`, calibration uses ROI pixels only.
  - `--defect-border-ignore-px INT` (optional; ignores N pixels at the anomaly-map border for defects extraction)
  - Map smoothing (optional; reduces speckle before thresholding):
    - `--defect-map-smoothing none|median|gaussian|box`
    - `--defect-map-smoothing-ksize INT`
    - `--defect-map-smoothing-sigma FLOAT` (gaussian only)
  - Hysteresis thresholding (optional; keeps low regions connected to high seeds):
    - `--defect-hysteresis`
    - `--defect-hysteresis-low FLOAT`
    - `--defect-hysteresis-high FLOAT`
  - Shape filters (optional; useful to remove long thin strips / speckle fragments):
    - `--defect-min-fill-ratio FLOAT` — drop components whose `area / bbox_area` is below this threshold
    - `--defect-max-aspect-ratio FLOAT` — drop components whose bbox aspect ratio exceeds this threshold
    - `--defect-min-solidity FLOAT` — drop components whose solidity (contour / convex hull) is below this threshold
  - Region merge (optional; affects regions list only, mask unchanged):
    - `--defect-merge-nearby`
    - `--defect-merge-nearby-max-gap-px INT`
  - Output limiting (optional):
    - `--defect-max-regions INT`
    - `--defect-max-regions-sort-by score_max|score_mean|area`
  - Region-level filters (optional):
    - `--defect-min-score-max FLOAT` — drop components whose max anomaly score is below the threshold
    - `--defect-min-score-mean FLOAT` — drop components whose mean anomaly score is below the threshold
  - Mask morphology (optional):
    - `--defect-open-ksize INT` / `--defect-close-ksize INT`
    - `--defect-fill-holes`
- `--from-run RUN_DIR` — load model/threshold/checkpoint from a prior `pyimgano-train` workbench run
  - If the run contains multiple categories, pass `--from-run-category NAME`.
- `--infer-config PATH` — load model/threshold/checkpoint from an exported workbench infer-config
  - For example: `runs/.../artifacts/infer_config.json`
  - If the infer-config contains multiple categories, pass `--infer-category NAME`.

Defects export example:

```bash
pyimgano-infer \
  --model vision_patchcore \
  --pretrained \
  --train-dir /path/to/train/good \
  --input /path/to/inputs \
  --defects \
  --save-masks /tmp/pyimgano_masks \
  --mask-format png \
  --pixel-threshold 0.5 \
  --pixel-threshold-strategy fixed \
  --roi-xyxy-norm 0.1 0.1 0.9 0.9 \
  --save-jsonl out.jsonl
```

Each JSONL record includes a `defects` block when `--defects` is enabled:

```json
{
  "defects": {
    "pixel_threshold": 0.5,
    "pixel_threshold_provenance": {"method": "fixed", "source": "explicit"},
    "mask": {"path": "masks/000000_x.png", "shape": [256, 256], "dtype": "uint8", "encoding": "png"},
    "regions": [{"id": 1, "bbox_xyxy": [12, 34, 80, 120], "area": 1532, "centroid_xy": [45.2, 77.8]}]
  }
}
```

When `--include-confidence` is enabled and the detector supports confidence semantics,
records also include:

```json
{
  "label_confidence": 0.93,
  "decision_summary": {
    "decision": "normal",
    "threshold_applied": true,
    "has_confidence": true,
    "rejected": false,
    "requires_review": false
  }
}
```

When `--reject-confidence-below` is enabled, records also include explicit rejection status:

```json
{
  "label": -2,
  "label_confidence": 0.61,
  "rejected": true,
  "decision_summary": {
    "decision": "rejected_low_confidence",
    "threshold_applied": true,
    "has_confidence": true,
    "rejected": true,
    "requires_review": true
  }
}
```

---

## `pyimgano-validate-infer-config`

Validates an exported `infer_config.json` from the workbench before deployment.

```bash
pyimgano-validate-infer-config runs/.../artifacts/infer_config.json
```

Notes:

- For multi-category infer-configs, pass `--infer-category NAME`.
- To skip file existence checks (portable configs), pass `--no-check-files`.
- To print the normalized payload, pass `--json`.
- Exported infer-configs use `schema_version=1`.
- Legacy infer-configs without `schema_version` are accepted and normalized to `1` with a warning.
- Future infer-config schema versions are rejected with a clear compatibility error.
- The validator also normalizes and checks deploy-time `prediction.*` defaults
  (`reject_confidence_below` in `(0,1]`, `reject_label` as int).

---

## `pyimgano-weights`

Utilities for local checkpoint inventory and lightweight artifact validation.
This CLI never downloads weights.

Common usage:

```bash
pyimgano-weights hash ./checkpoints/model.pt
pyimgano-weights validate ./weights_manifest.json --check-files --check-hashes --json
pyimgano-weights validate-model-card ./model_card.json --json
pyimgano-weights validate-model-card ./model_card.json --check-files --check-hashes --json
pyimgano-weights validate-model-card ./model_card.json --manifest ./weights_manifest.json --check-files --check-hashes --json
pyimgano-weights audit-bundle ./deploy_bundle --check-hashes --json
pyimgano-weights template manifest
pyimgano-weights template model-card
```

Notes:

- `validate-model-card` performs schema validation by default.
- Add `--check-files` to verify that `weights.path` exists on disk.
- Add `--check-hashes` to verify `weights.sha256` when present.
- Add `--manifest FILE` to cross-check the model card against a weights manifest.
- `audit-bundle DIR` validates `DIR/model_card.json` and `DIR/weights_manifest.json` together and returns exit code `1` unless the bundle is delivery-ready.
- Relative `weights.path` values are resolved against the model card parent directory by default.
- Override the resolution root with `--base-dir DIR`.
- When possible, set `weights.manifest_entry` in the model card for an explicit manifest link.

---

## `pyimgano-bundle`

Offline deploy-bundle validation and execution for CPU-first QC bundles.

Common usage:

```bash
pyimgano-bundle validate ./deploy_bundle --json
pyimgano-bundle validate ./deploy_bundle --check-hashes --json
pyimgano-bundle run ./deploy_bundle --image-dir ./inputs --output-dir ./bundle_run --json
pyimgano-bundle run ./deploy_bundle --input-manifest ./input_manifest.jsonl --output-dir ./bundle_run --json
```

Notes:

- `validate` checks `infer_config.json`, `bundle_manifest.json`, and optional bundle weight audit files as one deploy-bundle contract.
- `validate` also checks `bundle_manifest.json` refs/roles/completeness flags and operator-contract digest consistency when those fields are present.
- `run` executes offline inference from the bundle and writes `results.jsonl` plus `run_report.json` under `--output-dir`.
- Batch gates such as `--max-anomaly-rate`, `--max-reject-rate`, `--max-error-rate`, and `--min-processed` only affect run verdicts, not the underlying bundle contract.
- Use `pyimgano-weights audit-bundle` when you want a weights/model-card-focused audit without running bundle validation or inference.

---

## `pyimgano-train`

Runs a **recipe-driven workbench** run from a JSON-first config file. This is the
recommended entrypoint for industrial “adaptation-first” workflows where you
want reproducible artifacts (config, environment snapshot, per-image JSONL, etc.).

See also: `docs/RECIPES.md`

### Discovery

- List recipes: `pyimgano-train --list-recipes`
- List recipes (JSON): `pyimgano-train --list-recipes --json`
- Recipe info: `pyimgano-train --recipe-info industrial-adapt`
- Recipe info (JSON): `pyimgano-train --recipe-info industrial-adapt --json`

### Run a recipe

```bash
pyimgano-train --config cfg.json
```

Common outputs:

- `--export-infer-config` — write `artifacts/infer_config.json` into the run directory
- `--export-infer-config` also writes `artifacts/calibration_card.json` when threshold provenance is available
- `--export-deploy-bundle` — write `deploy_bundle/` (includes `infer_config.json` + referenced checkpoints)
  - deploy bundles also include `calibration_card.json` when it exists in the run artifacts
  - Validate the bundle with: `pyimgano-validate-infer-config deploy_bundle/infer_config.json`
- See `docs/CALIBRATION_AUDIT.md` for how to review threshold provenance, score summaries, and split context.

### Artifact layout

Workbench runs follow a benchmark-compatible layout and add extra folders:

```
<run_dir>/
  report.json
  config.json
  environment.json
  categories/<cat>/report.json
  categories/<cat>/per_image.jsonl
  checkpoints/...
  artifacts/...
```

### Common overrides

Flags override the config (useful for quick experiments):

- `--dataset NAME` / `--root PATH` / `--category CAT`
- `--model MODEL_NAME` / `--device cpu|cuda`
- `--preprocessing-preset NAME` — override `preprocessing.illumination_contrast` with a deployable preset
  discovered via `pyim --list preprocessing --deployable-only`

### Preflight dataset validation

Validate dataset health and emit a machine-readable JSON report (no training):

```bash
pyimgano-train --config cfg.json --preflight
```

Behavior:

- Prints: `{"preflight": ...}` (JSON) to stdout.
- Returns exit code `0` when no `severity="error"` issues exist.
- Returns exit code `2` when any `severity="error"` issue exists.

This is intended for CI and pipeline orchestration (e.g. detect missing files,
duplicate paths, manifest group split conflicts, or incomplete anomaly masks
before starting a recipe run).

### Notes

- Training-enabled workbench runs persist checkpoints under `checkpoints/<category>/...` when supported.
- For reusing a run in deploy-style inference, see `pyimgano-infer --from-run` and `docs/RECIPES.md`.
- For manifest datasets, `pyimgano-train --dry-run` validates that `dataset.manifest_path` exists and is readable.

---

## `pyimgano-robust-benchmark`

Runs clean + corruption robustness evaluation (when supported by the selected model/input mode).

Example:

```bash
pyimgano-robust-benchmark \
  --dataset mvtec \
  --root /path/to/mvtec_ad \
  --category bottle \
  --model vision_patchcore \
  --pretrained \
  --device cuda
```

Notes:

- `--save-run` persists a run directory with `report.json`, `config.json`, and `environment.json`.
- `--output-dir /path/to/run_dir` selects that persisted run directory explicitly.
- Saved runs also include `artifacts/robustness_conditions.csv` and `artifacts/robustness_summary.json`.
- `robustness_conditions.csv` includes per-condition `drop_*` deltas relative to the clean baseline.
- `robustness_summary.json` includes aggregate clean/corruption accuracy retention plus latency drift ratios for run comparison.
- `robustness_summary.json` also carries audit refs/digests for `robustness_conditions.csv`, and saved `report.json`
  carries matching `robustness_trust.audit_refs|audit_digests` entries for both robustness artifacts.
- `pyimgano-runs list` / `latest` revalidate those saved robustness artifact digests before reporting `robustness_trust`.
- Saved robustness runs can be inspected with `pyimgano-runs list --kind robustness`.

---

## `pyimgano-manifest`

Generate a JSONL manifest for the built-in `custom` dataset layout.

Example:

```bash
pyimgano-manifest \
  --root /path/to/custom_dataset \
  --out /path/to/manifest.jsonl \
  --include-masks
```

Notes:

- Output is stable and sorted (useful for reproducible diffs).
- By default, paths are written relative to the output manifest directory.
- Use `--absolute-paths` to emit absolute paths when you need portability across working directories.

---

## `pyimgano-datasets`

Inspect dataset layouts, convert them to manifests, and summarize industrial-readiness signals.

Examples:

```bash
pyimgano-datasets list --json
pyimgano-datasets detect /path/to/dataset_root --json
pyimgano-datasets lint /path/to/dataset_root --dataset custom --json
pyimgano-datasets profile /path/to/dataset_root --json
```

Notes:

- `lint` and `profile` now surface a compact `readiness` payload in JSON and `readiness_status` / `issue_codes` in text output.
- Common issue codes include `FEWSHOT_TRAIN_SET`, `MISSING_TEST_ANOMALY`, and `PIXEL_METRICS_UNAVAILABLE`.
- `lint` still reports manifest/file validation under `validation`, but now also returns a non-zero exit code when industrial readiness is `error`.

---

## `pyimgano-export-torchscript`

Export a torchvision backbone (classification head stripped) as a TorchScript `.pt` file.

Requires:

- `pip install "pyimgano[torch]"`

Example (offline-safe):

```bash
pyimgano-export-torchscript \
  --backbone resnet18 \
  --out /tmp/resnet18_backbone.pt
```

Notes:

- `--pretrained` is **off by default** to avoid implicit weight downloads.
- Use `--method trace|script` depending on your deployment constraints (default: `trace`).

---

## `pyimgano-export-onnx`

Export a torchvision backbone (classification head stripped) as an ONNX `.onnx` file.

Requires:

- `pip install "pyimgano[torch]"` (export)
- `pip install "pyimgano[onnx]"` (recommended; needed for `--verify`, and required by newer `torch.onnx.export` flows via `onnxscript`)

Example (offline-safe):

```bash
pyimgano-export-onnx \
  --backbone resnet18 \
  --out /tmp/resnet18_backbone.onnx
```

Notes:

- `--pretrained` is **off by default** to avoid implicit weight downloads.
- By default, export uses `--dynamic-batch` (deploy-friendly).
- `--verify` (default: true) checks that the exported file loads in `onnx` and `onnxruntime`.

## Notes

- Many models have optional backends; when required dependencies are missing, error messages include install hints.
- For a full model catalog, see `docs/MODEL_INDEX.md` or use `pyimgano-benchmark --list-models`.
