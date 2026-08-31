from __future__ import annotations

import inspect
import subprocess
import sys

from pyimgano.artifacts import bind_policy, export_run, import_onnx
from pyimgano.inference import ArtifactRuntime, load_artifact


def test_public_load_artifact_signature_is_stable() -> None:
    signature = inspect.signature(load_artifact)
    assert list(signature.parameters) == [
        "artifact",
        "category",
        "format",
        "backend",
        "artifact_id",
        "device",
        "providers",
        "session_options",
        "trust_checkpoint",
    ]
    assert signature.return_annotation in {ArtifactRuntime, "ArtifactRuntime"}


def test_public_artifact_runtime_exposes_detector_methods() -> None:
    assert callable(ArtifactRuntime.decision_function)
    assert callable(ArtifactRuntime.predict)
    assert callable(ArtifactRuntime.predict_anomaly_map)
    assert callable(ArtifactRuntime.score_and_maps)


def test_public_artifact_creation_facades_are_stable() -> None:
    assert list(inspect.signature(export_run).parameters) == [
        "run_dir",
        "formats",
        "out",
        "category",
        "verification_level",
        "strict",
        "trust_checkpoint",
        "overwrite",
    ]
    assert list(inspect.signature(import_onnx).parameters) == [
        "model",
        "contract",
        "out",
        "policy",
        "overwrite",
    ]
    assert list(inspect.signature(bind_policy).parameters) == [
        "source",
        "policy",
        "out",
        "probe",
        "trust_checkpoint",
    ]


def test_export_run_is_a_thin_service_facade(monkeypatch) -> None:
    import pyimgano.services.export_service as export_service

    calls = []
    monkeypatch.setattr(
        export_service,
        "export_from_run",
        lambda **kwargs: calls.append(dict(kwargs)) or {"status": "ok"},
    )

    result = export_run(
        "run",
        formats=["native", "onnx"],
        out="artifacts",
        category="bottle",
        verification_level="end_to_end",
        overwrite=True,
    )

    assert result == {"status": "ok"}
    assert calls == [
        {
            "run_dir": "run",
            "formats": ["native", "onnx"],
            "out_dir": "artifacts",
            "category": "bottle",
            "verification_level": "end_to_end",
            "strict": True,
            "trust_checkpoint": False,
            "overwrite": True,
        }
    ]


def test_artifact_load_service_has_no_fresh_process_import_cycle() -> None:
    completed = subprocess.run(
        [sys.executable, "-c", "import pyimgano.services.artifact_load_service"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
