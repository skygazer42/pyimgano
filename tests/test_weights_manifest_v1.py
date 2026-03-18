import hashlib
import json
from pathlib import Path


def test_validate_weights_manifest_file_checks_files_and_hashes(tmp_path):
    from pyimgano.weights.manifest import validate_weights_manifest_file

    ckpt = tmp_path / "checkpoints" / "model.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    data = b"dummy-weights"
    ckpt.write_bytes(data)

    sha = hashlib.sha256(data).hexdigest()

    manifest = {
        "schema_version": 1,
        "entries": [
            {
                "name": "dummy",
                "path": "checkpoints/model.pt",
                "sha256": sha,
                "license": "internal",
                "source": "unit-test",
            }
        ],
    }
    manifest_path = tmp_path / "weights_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    report = validate_weights_manifest_file(
        manifest_path=str(manifest_path),
        check_files=True,
        check_hashes=True,
    )
    assert report.ok
    assert report.errors == ()
    assert len(report.entries) == 1
    resolved_posix = Path(report.entries[0]["resolved_path"]).as_posix()
    assert resolved_posix.endswith("checkpoints/model.pt")


def test_validate_weights_manifest_file_reports_sha_mismatch(tmp_path):
    from pyimgano.weights.manifest import validate_weights_manifest_file

    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"dummy-weights")

    manifest = {
        "schema_version": 1,
        "entries": [
            {
                "name": "dummy",
                "path": "model.pt",
                "sha256": "0" * 64,
            }
        ],
    }
    manifest_path = tmp_path / "weights_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    report = validate_weights_manifest_file(
        manifest_path=str(manifest_path),
        check_files=True,
        check_hashes=True,
    )
    assert not report.ok
    assert any("SHA256 mismatch" in e for e in report.errors)


def test_validate_weights_manifest_warns_for_missing_asset_metadata(tmp_path):
    from pyimgano.weights.manifest import validate_weights_manifest_file

    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"dummy-weights")

    manifest = {
        "schema_version": 1,
        "entries": [
            {
                "name": "dummy",
                "path": "model.pt",
            }
        ],
    }
    manifest_path = tmp_path / "weights_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    report = validate_weights_manifest_file(
        manifest_path=str(manifest_path),
        check_files=True,
        check_hashes=False,
    )
    assert report.ok
    assert any("source" in w for w in report.warnings)
    assert any("license" in w for w in report.warnings)
    assert any("runtime" in w for w in report.warnings)
