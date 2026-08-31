from __future__ import annotations

import json
from pathlib import Path

from pyimgano.services.bundle_run_service import (
    BundleInferenceBatchRequest,
    build_bundle_infer_argv,
)


def _request(bundle: Path, output: Path, **kwargs) -> BundleInferenceBatchRequest:  # noqa: ANN003
    return BundleInferenceBatchRequest(
        bundle_dir=bundle,
        input_records=[{"resolved_input_path": "/inputs/a.png"}],
        results_jsonl=output,
        **kwargs,
    )


def test_bundle_inference_prefers_embedded_artifact_runtime(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "bundle_manifest.json").write_text(
        json.dumps(
            {
                "artifact_refs": [
                    {
                        "path": "artifacts/bottle/onnx/artifact_manifest.json",
                        "category": "bottle",
                        "format": "onnx",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    argv = build_bundle_infer_argv(
        _request(
            bundle,
            tmp_path / "results.jsonl",
            artifact_category="bottle",
            artifact_format="onnx",
            artifact_backend="onnxruntime",
            device="cpu",
            onnx_providers="CPUExecutionProvider",
            onnx_session_options='{"intra_op_num_threads":2}',
        )
    )

    assert argv[:2] == ["--artifact", str(bundle)]
    assert "--infer-config" not in argv
    assert argv[argv.index("--artifact-category") + 1] == "bottle"
    assert argv[argv.index("--artifact-format") + 1] == "onnx"
    assert argv[argv.index("--onnx-providers") + 1] == "CPUExecutionProvider"


def test_bundle_inference_keeps_legacy_infer_config_fallback(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "bundle_manifest.json").write_text(
        json.dumps({"schema_version": 1}), encoding="utf-8"
    )

    argv = build_bundle_infer_argv(_request(bundle, tmp_path / "results.jsonl"))

    assert argv[:2] == ["--infer-config", str(bundle / "infer_config.json")]


def test_deploy_bundle_manifest_indexes_runtime_artifact_manifests(tmp_path: Path) -> None:
    from pyimgano.reporting.deploy_bundle import build_deploy_bundle_manifest

    source = tmp_path / "run"
    bundle = tmp_path / "bundle"
    source.mkdir()
    bundle.mkdir()
    (source / "environment.json").write_text("{}", encoding="utf-8")
    artifact = bundle / "artifacts" / "bottle" / "onnx"
    artifact.mkdir(parents=True)
    (artifact / "artifact_manifest.json").write_text(
        json.dumps(
            {
                "artifact_id": "sha256:test",
                "layout": "single_graph",
                "model": {"category": "bottle"},
                "runtime": {"backend": "onnxruntime"},
                "components": [{"role": "runtime_model", "format": "onnx"}],
            }
        ),
        encoding="utf-8",
    )

    manifest = build_deploy_bundle_manifest(bundle_dir=bundle, source_run_dir=source)

    assert manifest["artifact_refs"] == [
        {
            "path": "artifacts/bottle/onnx/artifact_manifest.json",
            "category": "bottle",
            "format": "onnx",
            "backend": "onnxruntime",
            "artifact_id": "sha256:test",
        }
    ]
    assert manifest["artifact_roles"]["runtime_artifact_manifest"] == [
        "artifacts/bottle/onnx/artifact_manifest.json"
    ]


def test_copy_exported_artifacts_to_bundle_preserves_index_and_rejects_symlinks(
    tmp_path: Path,
) -> None:
    from pyimgano.services.train_export_helpers import copy_exported_artifacts_to_bundle

    exported = tmp_path / "exported"
    exported.mkdir()
    (exported / "export_index.json").write_text("{}", encoding="utf-8")
    artifact = exported / "bottle" / "native"
    artifact.mkdir(parents=True)
    (artifact / "artifact_manifest.json").write_text("{}", encoding="utf-8")
    bundle = tmp_path / "bundle"
    bundle.mkdir()

    destination = copy_exported_artifacts_to_bundle(
        {"output_dir": str(exported), "artifacts": [{"format": "native"}]},
        bundle_dir=bundle,
    )

    assert destination == bundle / "artifacts"
    assert (destination / "export_index.json").is_file()
    assert (destination / "bottle" / "native" / "artifact_manifest.json").is_file()
