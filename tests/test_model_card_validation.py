import json
from pathlib import Path
from types import MappingProxyType


def test_validate_model_card_file_accepts_minimal_payload(tmp_path):
    from pyimgano.weights.model_card import validate_model_card_file

    payload = {
        "schema_version": 1,
        "model_name": "patchcore_bottle_v1",
        "summary": {
            "purpose": "Bottle surface anomaly detection",
            "intended_inputs": "RGB bottle images",
            "output_contract": "image-level + pixel-level",
        },
        "weights": {
            "path": "checkpoints/patchcore_bottle.pt",
            "sha256": "0" * 64,
            "source": "internal training run",
            "license": "internal",
        },
        "deployment": {
            "runtime": "torch",
        },
    }
    path = tmp_path / "model_card.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report = validate_model_card_file(path)
    assert report.ok
    assert report.errors == ()
    assert report.normalized["model_name"] == "patchcore_bottle_v1"
    assert report.normalized["deployment"]["runtime"] == "torch"


def test_validate_model_card_file_can_check_weight_asset_and_hash(tmp_path):
    from pyimgano.weights.model_card import validate_model_card_file

    ckpt = tmp_path / "checkpoints" / "patchcore_bottle.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    data = b"dummy-model-card-weights"
    ckpt.write_bytes(data)

    import hashlib

    sha = hashlib.sha256(data).hexdigest()

    payload = {
        "schema_version": 1,
        "model_name": "patchcore_bottle_v1",
        "summary": {
            "purpose": "Bottle surface anomaly detection",
            "intended_inputs": "RGB bottle images",
            "output_contract": "image-level + pixel-level",
        },
        "weights": {
            "path": "checkpoints/patchcore_bottle.pt",
            "sha256": sha,
            "source": "internal training run",
            "license": "internal",
        },
        "deployment": {
            "runtime": "torch",
        },
    }
    path = tmp_path / "model_card.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report = validate_model_card_file(path, check_files=True, check_hashes=True)
    assert report.ok
    assert report.assets["weights"]["exists"] is True
    assert report.assets["weights"]["sha256_match"] is True
    assert (
        Path(report.assets["weights"]["resolved_path"])
        .as_posix()
        .endswith("checkpoints/patchcore_bottle.pt")
    )


def test_validate_model_card_file_rejects_missing_required_fields(tmp_path):
    from pyimgano.weights.model_card import validate_model_card_file

    path = tmp_path / "bad_model_card.json"
    path.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")

    report = validate_model_card_file(path)
    assert not report.ok
    assert any("model_name" in err for err in report.errors)
    assert any("summary" in err for err in report.errors)
    assert any("weights" in err for err in report.errors)
    assert any("deployment.runtime" in err for err in report.errors)


def test_validate_model_card_file_reports_missing_weight_asset_when_requested(tmp_path):
    from pyimgano.weights.model_card import validate_model_card_file

    payload = {
        "schema_version": 1,
        "model_name": "patchcore_bottle_v1",
        "summary": {
            "purpose": "Bottle surface anomaly detection",
            "intended_inputs": "RGB bottle images",
            "output_contract": "image-level + pixel-level",
        },
        "weights": {
            "path": "checkpoints/missing.pt",
            "source": "internal training run",
            "license": "internal",
        },
        "deployment": {
            "runtime": "torch",
        },
    }
    path = tmp_path / "model_card.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report = validate_model_card_file(path, check_files=True)
    assert not report.ok
    assert any("Missing weights file" in err for err in report.errors)


def test_validate_model_card_file_can_cross_check_manifest_by_entry_name(tmp_path):
    from pyimgano.weights.model_card import validate_model_card_file

    ckpt = tmp_path / "checkpoints" / "patchcore_bottle.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    data = b"dummy-model-card-weights"
    ckpt.write_bytes(data)

    import hashlib

    sha = hashlib.sha256(data).hexdigest()

    manifest = {
        "schema_version": 1,
        "entries": [
            {
                "name": "patchcore_bottle_v1",
                "path": "checkpoints/patchcore_bottle.pt",
                "sha256": sha,
                "source": "internal training run",
                "license": "internal",
                "runtime": "torch",
            }
        ],
    }
    manifest_path = tmp_path / "weights_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    payload = {
        "schema_version": 1,
        "model_name": "patchcore_bottle_v1",
        "summary": {
            "purpose": "Bottle surface anomaly detection",
            "intended_inputs": "RGB bottle images",
            "output_contract": "image-level + pixel-level",
        },
        "weights": {
            "path": "checkpoints/patchcore_bottle.pt",
            "sha256": sha,
            "source": "internal training run",
            "license": "internal",
            "manifest_entry": "patchcore_bottle_v1",
        },
        "deployment": {
            "runtime": "torch",
        },
    }
    path = tmp_path / "model_card.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report = validate_model_card_file(
        path,
        manifest_path=manifest_path,
        check_files=True,
        check_hashes=True,
    )
    assert report.ok
    assert report.assets["manifest"]["ok"] is True
    assert report.assets["manifest"]["matched_entry"] == "patchcore_bottle_v1"


def test_validate_model_card_file_can_cross_check_manifest_by_path(tmp_path):
    from pyimgano.weights.model_card import validate_model_card_file

    ckpt = tmp_path / "checkpoints" / "patchcore_bottle.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    data = b"dummy-model-card-weights"
    ckpt.write_bytes(data)

    import hashlib

    sha = hashlib.sha256(data).hexdigest()

    manifest = {
        "schema_version": 1,
        "entries": [
            {
                "name": "bottle_checkpoint",
                "path": "checkpoints/patchcore_bottle.pt",
                "sha256": sha,
                "source": "internal training run",
                "license": "internal",
                "runtime": "torch",
            }
        ],
    }
    manifest_path = tmp_path / "weights_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    payload = {
        "schema_version": 1,
        "model_name": "patchcore_bottle_v1",
        "summary": {
            "purpose": "Bottle surface anomaly detection",
            "intended_inputs": "RGB bottle images",
            "output_contract": "image-level + pixel-level",
        },
        "weights": {
            "path": "checkpoints/patchcore_bottle.pt",
            "sha256": sha,
            "source": "internal training run",
            "license": "internal",
        },
        "deployment": {
            "runtime": "torch",
        },
    }
    path = tmp_path / "model_card.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report = validate_model_card_file(
        path,
        manifest_path=manifest_path,
        check_files=True,
        check_hashes=True,
    )
    assert report.ok
    assert report.assets["manifest"]["matched_entry"] == "bottle_checkpoint"


def test_validate_model_card_file_reports_manifest_entry_mismatch(tmp_path):
    from pyimgano.weights.model_card import validate_model_card_file

    ckpt = tmp_path / "checkpoints" / "patchcore_bottle.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_bytes(b"dummy-model-card-weights")

    manifest = {
        "schema_version": 1,
        "entries": [
            {
                "name": "different_checkpoint",
                "path": "checkpoints/patchcore_bottle.pt",
                "source": "internal training run",
                "license": "internal",
                "runtime": "torch",
            }
        ],
    }
    manifest_path = tmp_path / "weights_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    payload = {
        "schema_version": 1,
        "model_name": "patchcore_bottle_v1",
        "summary": {
            "purpose": "Bottle surface anomaly detection",
            "intended_inputs": "RGB bottle images",
            "output_contract": "image-level + pixel-level",
        },
        "weights": {
            "path": "checkpoints/patchcore_bottle.pt",
            "source": "internal training run",
            "license": "internal",
            "manifest_entry": "patchcore_bottle_v1",
        },
        "deployment": {
            "runtime": "torch",
        },
    }
    path = tmp_path / "model_card.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report = validate_model_card_file(path, manifest_path=manifest_path, check_files=True)
    assert not report.ok
    assert any("manifest entry" in err for err in report.errors)


def test_validate_model_card_accepts_mapping_payloads() -> None:
    from pyimgano.weights.model_card import validate_model_card

    payload = MappingProxyType(
        {
            "schema_version": 1,
            "model_name": "patchcore_bottle_v1",
            "summary": MappingProxyType(
                {
                    "purpose": "Bottle surface anomaly detection",
                    "intended_inputs": "RGB bottle images",
                    "output_contract": "image-level + pixel-level",
                }
            ),
            "weights": MappingProxyType(
                {
                    "path": "checkpoints/patchcore_bottle.pt",
                    "sha256": "0" * 64,
                    "source": "internal training run",
                    "license": "internal",
                }
            ),
            "deployment": MappingProxyType({"runtime": "torch"}),
        }
    )

    report = validate_model_card(payload)

    assert report.ok
    assert report.errors == ()
