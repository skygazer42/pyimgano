from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import pyimgano.infer_cli as infer_cli


def _write_png(path: Path) -> None:
    Image.fromarray(np.zeros((6, 7, 3), dtype=np.uint8), mode="RGB").save(path)


class _ArtifactRuntimeStub:
    model_name = "imported-score-model"
    threshold_ = 0.5
    infer_config = {
        "schema_family": "pyimgano-artifact-policy",
        "schema_version": 1,
        "threshold": 0.5,
        "postprocess": {
            "image_threshold": {
                "threshold": 0.5,
                "score_order": "higher_is_more_anomalous",
            }
        },
    }
    runtime_info = {
        "backend": "onnxruntime",
        "selected_provider": "CPUExecutionProvider",
    }

    def __init__(self) -> None:
        self.closed = False

    def decision_function(self, inputs):  # noqa: ANN001
        return np.linspace(0.1, 0.9, num=len(inputs), dtype=np.float32)

    def close(self) -> None:
        self.closed = True


def test_infer_cli_artifact_loads_selected_runtime_and_releases_it(
    tmp_path: Path, monkeypatch
) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    _write_png(input_dir / "a.png")
    _write_png(input_dir / "b.png")
    output = tmp_path / "results.jsonl"
    runtime = _ArtifactRuntimeStub()
    calls: list[tuple[object, dict[str, object]]] = []

    import pyimgano.inference as inference

    monkeypatch.setattr(
        inference,
        "load_artifact",
        lambda artifact, **kwargs: calls.append((artifact, dict(kwargs))) or runtime,
    )

    rc = infer_cli.main(
        [
            "--artifact",
            str(tmp_path / "exports"),
            "--artifact-category",
            "bottle",
            "--artifact-format",
            "onnx",
            "--artifact-backend",
            "onnxruntime",
            "--onnx-providers",
            "CUDAExecutionProvider,CPUExecutionProvider",
            "--onnx-provider-options",
            '{"CUDAExecutionProvider":{"device_id":"1"}}',
            "--onnx-session-options",
            '{"intra_op_num_threads":2}',
            "--input",
            str(input_dir),
            "--save-jsonl",
            str(output),
        ]
    )

    assert rc == 0
    assert runtime.closed is True
    assert calls == [
        (
            str(tmp_path / "exports"),
            {
                "category": "bottle",
                "format": "onnx",
                "backend": "onnxruntime",
                "artifact_id": None,
                "device": None,
                "providers": [
                    {
                        "name": "CUDAExecutionProvider",
                        "options": {"device_id": "1"},
                    },
                    "CPUExecutionProvider",
                ],
                "session_options": {"intra_op_num_threads": 2},
                "trust_checkpoint": False,
            },
        )
    ]
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [record["label"] for record in records] == [0, 1]


def test_infer_cli_artifact_is_mutually_exclusive_with_model() -> None:
    with pytest.raises(SystemExit):
        infer_cli._build_parser().parse_args(
            ["--artifact", "artifact", "--model", "vision_ecod", "--input", "image.png"]
        )


def test_infer_cli_artifact_rejects_training_and_reconstruction_overrides(
    tmp_path: Path, capsys
) -> None:
    image = tmp_path / "image.png"
    _write_png(image)

    rc = infer_cli.main(
        [
            "--artifact",
            str(tmp_path / "artifact"),
            "--train-dir",
            str(tmp_path),
            "--input",
            str(image),
        ]
    )

    assert rc == 2
    assert "Artifact manifests own model reconstruction" in capsys.readouterr().err


def test_infer_cli_provider_options_require_selected_providers(tmp_path: Path, capsys) -> None:
    image = tmp_path / "image.png"
    _write_png(image)

    rc = infer_cli.main(
        [
            "--artifact",
            str(tmp_path / "artifact"),
            "--onnx-provider-options",
            '{"CPUExecutionProvider":{}}',
            "--input",
            str(image),
        ]
    )

    assert rc == 2
    assert "requires --onnx-providers" in capsys.readouterr().err


def test_infer_cli_exact_artifact_id_cannot_be_combined_with_other_selectors(
    tmp_path: Path, capsys
) -> None:
    image = tmp_path / "image.png"
    _write_png(image)

    rc = infer_cli.main(
        [
            "--artifact",
            str(tmp_path / "exports"),
            "--artifact-id",
            f"sha256:{'a' * 64}",
            "--artifact-format",
            "onnx",
            "--input",
            str(image),
        ]
    )

    assert rc == 2
    assert "exact selector" in capsys.readouterr().err


def test_infer_cli_artifact_rejects_device_and_provider_override_together(
    tmp_path: Path, capsys
) -> None:
    image = tmp_path / "image.png"
    _write_png(image)

    rc = infer_cli.main(
        [
            "--artifact",
            str(tmp_path / "exports"),
            "--device",
            "cpu",
            "--onnx-providers",
            "CPUExecutionProvider",
            "--input",
            str(image),
        ]
    )

    assert rc == 2
    assert "cannot be combined" in capsys.readouterr().err


def test_infer_cli_profile_records_executed_artifact_backend_once(
    tmp_path: Path, monkeypatch
) -> None:
    image = tmp_path / "image.png"
    _write_png(image)
    profile = tmp_path / "profile.json"
    runtime = _ArtifactRuntimeStub()
    import pyimgano.inference as inference

    monkeypatch.setattr(inference, "load_artifact", lambda *_args, **_kwargs: runtime)

    rc = infer_cli.main(
        [
            "--artifact",
            str(tmp_path / "artifact"),
            "--input",
            str(image),
            "--profile-json",
            str(profile),
        ]
    )

    assert rc == 0
    payload = json.loads(profile.read_text(encoding="utf-8"))
    assert payload["runtime"] == {
        "backend": "onnxruntime",
        "selected_provider": "CPUExecutionProvider",
    }
