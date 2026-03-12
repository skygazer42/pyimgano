from __future__ import annotations


def test_emit_signature_payload_delegates_json_to_cli_output(monkeypatch) -> None:
    import pyimgano.cli_discovery_rendering as rendering

    calls = []
    monkeypatch.setattr(
        rendering,
        "cli_output",
        type(
            "_StubCliOutput",
            (),
            {
                "emit_jsonable": staticmethod(
                    lambda payload, **kwargs: calls.append((payload, kwargs)) or 7
                )
            },
        ),
        raising=False,
    )

    payload = {
        "name": "vision_ecod",
        "tags": ["vision", "classical"],
        "metadata": {"year": 2022},
        "signature": "(device='cpu')",
        "accepts_var_kwargs": False,
        "accepted_kwargs": ["device"],
    }

    rc = rendering.emit_signature_payload(payload, json_output=True)
    assert rc == 7
    assert calls == [(payload, {})]


def test_emit_signature_payload_renders_text(capsys) -> None:
    from pyimgano.cli_discovery_rendering import emit_signature_payload

    rc = emit_signature_payload(
        {
            "name": "vision_ecod",
            "tags": ["vision", "classical"],
            "metadata": {"year": 2022},
            "signature": "(device='cpu')",
            "accepts_var_kwargs": False,
            "accepted_kwargs": ["device", "seed"],
        },
        json_output=False,
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "Name: vision_ecod" in out
    assert "Tags: vision, classical" in out
    assert "Metadata:" in out
    assert "  year: 2022" in out
    assert "Signature:" in out
    assert "Accepts **kwargs: no" in out
    assert "  - device" in out
    assert "  - seed" in out


def test_emit_model_preset_payload_renders_text(capsys) -> None:
    from pyimgano.cli_discovery_rendering import emit_model_preset_payload

    rc = emit_model_preset_payload(
        {
            "name": "industrial-structural-ecod",
            "model": "vision_feature_pipeline",
            "kwargs": {"feature_extractor": "identity"},
            "optional": False,
            "tags": ["classical", "vision"],
            "description": "Structural baseline preset.",
        },
        json_output=False,
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "Name: industrial-structural-ecod" in out
    assert "Model: vision_feature_pipeline" in out
    assert "Kwargs:" in out
    assert "  feature_extractor: identity" in out
    assert "Optional: no" in out
    assert "Tags: classical, vision" in out
    assert "Description: Structural baseline preset." in out
