from __future__ import annotations


def test_collect_pyim_audit_payload_delegates_to_model_registry(monkeypatch) -> None:
    import pyimgano.models.registry as model_registry
    from pyimgano.services.pyim_audit_service import collect_pyim_audit_payload

    payload = {
        "summary": {
            "models_with_required_issues": 0,
            "models_with_recommended_issues": 0,
            "models_with_invalid_fields": 0,
        }
    }
    monkeypatch.setattr(model_registry, "audit_model_metadata", lambda: payload)

    assert collect_pyim_audit_payload() == payload
