from __future__ import annotations


def test_emit_pyim_audit_payload_json_preserves_nonzero_exit(monkeypatch) -> None:
    import pyimgano.pyim_audit_rendering as rendering

    payload = {
        "summary": {
            "models_with_required_issues": 1,
            "models_with_recommended_issues": 0,
            "models_with_invalid_fields": 0,
        }
    }

    calls = []
    monkeypatch.setattr(
        rendering,
        "cli_output",
        type(
            "_StubCliOutput",
            (),
            {
                "emit_json": staticmethod(
                    lambda json_payload, **kwargs: calls.append((json_payload, kwargs)) or 0
                )
            },
        ),
        raising=False,
    )

    rc = rendering.emit_pyim_audit_payload(payload, json_output=True)

    assert rc == 1
    assert calls == [(payload, {})]


def test_emit_pyim_audit_payload_text_prints_summary(capsys) -> None:
    from pyimgano.pyim_audit_rendering import emit_pyim_audit_payload

    rc = emit_pyim_audit_payload(
        {
            "summary": {
                "models_with_required_issues": 0,
                "models_with_recommended_issues": 2,
                "models_with_invalid_fields": 0,
            }
        },
        json_output=False,
    )

    assert rc == 1
    out = capsys.readouterr().out
    assert "Metadata Audit" in out
    assert "required=0 recommended=2 invalid=0" in out
