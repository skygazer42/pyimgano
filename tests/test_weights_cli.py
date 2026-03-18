import json


def test_weights_cli_template_manifest_emits_json(capsys):
    from pyimgano.weights_cli import main

    rc = main(["template", "manifest"])
    assert rc == 0

    out = json.loads(capsys.readouterr().out)
    assert out["schema_version"] == 1
    assert isinstance(out["entries"], list)
    assert out["entries"][0]["license"] == "internal-or-upstream-license"


def test_weights_cli_template_model_card_emits_json(capsys):
    from pyimgano.weights_cli import main

    rc = main(["template", "model-card"])
    assert rc == 0

    out = json.loads(capsys.readouterr().out)
    assert out["schema_version"] == 1
    assert out["model_name"] == "example_model_name"
    assert out["deployment"]["runtime"] == "torch"


def test_weights_cli_validate_model_card_json_success(tmp_path, capsys):
    from pyimgano.weights_cli import main

    payload = {
        "schema_version": 1,
        "model_name": "demo_model",
        "summary": {
            "purpose": "demo",
            "intended_inputs": "RGB",
            "output_contract": "image-level",
        },
        "weights": {
            "path": "checkpoints/demo.pt",
            "source": "unit-test",
            "license": "internal",
        },
        "deployment": {"runtime": "torch"},
    }
    path = tmp_path / "model_card.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    rc = main(["validate-model-card", str(path), "--json"])
    assert rc == 0

    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is True
    assert out["normalized"]["model_name"] == "demo_model"


def test_weights_cli_validate_model_card_can_check_files_and_hashes(tmp_path, capsys):
    from pyimgano.weights_cli import main

    ckpt = tmp_path / "checkpoints" / "demo.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    data = b"demo-weights"
    ckpt.write_bytes(data)

    import hashlib

    sha = hashlib.sha256(data).hexdigest()

    payload = {
        "schema_version": 1,
        "model_name": "demo_model",
        "summary": {
            "purpose": "demo",
            "intended_inputs": "RGB",
            "output_contract": "image-level",
        },
        "weights": {
            "path": "checkpoints/demo.pt",
            "sha256": sha,
            "source": "unit-test",
            "license": "internal",
        },
        "deployment": {"runtime": "torch"},
    }
    path = tmp_path / "model_card.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    rc = main(
        [
            "validate-model-card",
            str(path),
            "--check-files",
            "--check-hashes",
            "--json",
        ]
    )
    assert rc == 0

    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is True
    assert out["assets"]["weights"]["exists"] is True
    assert out["assets"]["weights"]["sha256_match"] is True


def test_weights_cli_validate_model_card_can_cross_check_manifest(tmp_path, capsys):
    from pyimgano.weights_cli import main

    ckpt = tmp_path / "checkpoints" / "demo.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    data = b"demo-weights"
    ckpt.write_bytes(data)

    import hashlib

    sha = hashlib.sha256(data).hexdigest()

    manifest = {
        "schema_version": 1,
        "entries": [
            {
                "name": "demo_model_entry",
                "path": "checkpoints/demo.pt",
                "sha256": sha,
                "source": "unit-test",
                "license": "internal",
                "runtime": "torch",
            }
        ],
    }
    manifest_path = tmp_path / "weights_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    payload = {
        "schema_version": 1,
        "model_name": "demo_model",
        "summary": {
            "purpose": "demo",
            "intended_inputs": "RGB",
            "output_contract": "image-level",
        },
        "weights": {
            "path": "checkpoints/demo.pt",
            "sha256": sha,
            "source": "unit-test",
            "license": "internal",
            "manifest_entry": "demo_model_entry",
        },
        "deployment": {"runtime": "torch"},
    }
    path = tmp_path / "model_card.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    rc = main(
        [
            "validate-model-card",
            str(path),
            "--manifest",
            str(manifest_path),
            "--check-files",
            "--check-hashes",
            "--json",
        ]
    )
    assert rc == 0

    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is True
    assert out["assets"]["manifest"]["matched_entry"] == "demo_model_entry"
    trust = out["trust_summary"]
    assert trust["status"] == "trust-signaled"
    assert trust["trust_signals"]["has_weights_sha256"] is True
    assert trust["trust_signals"]["has_manifest_link"] is True
    assert trust["trust_signals"]["has_cross_checked_manifest"] is True
    assert trust["audit_refs"]["model_card_json"] == str(path)
    assert trust["audit_refs"]["weights_manifest_json"] == str(manifest_path)


def test_weights_cli_validate_manifest_json_includes_trust_summary(tmp_path, capsys):
    from pyimgano.weights_cli import main

    ckpt = tmp_path / "checkpoints" / "demo.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    data = b"demo-weights"
    ckpt.write_bytes(data)

    import hashlib

    sha = hashlib.sha256(data).hexdigest()

    manifest = {
        "schema_version": 1,
        "entries": [
            {
                "name": "demo_model_entry",
                "path": "checkpoints/demo.pt",
                "sha256": sha,
                "source": "unit-test",
                "license": "internal",
                "runtime": "torch",
            }
        ],
    }
    manifest_path = tmp_path / "weights_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    rc = main(
        [
            "validate",
            str(manifest_path),
            "--check-files",
            "--check-hashes",
            "--json",
        ]
    )
    assert rc == 0

    out = json.loads(capsys.readouterr().out)
    trust = out["trust_summary"]
    assert trust["status"] == "trust-signaled"
    assert trust["trust_signals"]["all_entries_have_sha256"] is True
    assert trust["trust_signals"]["all_entries_have_source"] is True
    assert trust["trust_signals"]["all_entries_have_license"] is True
    assert trust["trust_signals"]["all_entries_have_runtime"] is True
    assert trust["audit_refs"]["weights_manifest_json"] == str(manifest_path)
