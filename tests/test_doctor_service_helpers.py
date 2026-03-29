from __future__ import annotations


def test_split_csv_args_preserves_repeatable_comma_syntax() -> None:
    from pyimgano.services.doctor_service_helpers import split_csv_args

    assert split_csv_args(["torch, skimage", "onnx"]) == ["torch", "skimage", "onnx"]
    assert split_csv_args(None) == []


def test_build_require_extras_check_reports_missing_and_install_hint(monkeypatch) -> None:
    import pyimgano.services.doctor_service_helpers as helpers

    monkeypatch.setattr(
        helpers,
        "extra_importable",
        lambda extra: str(extra) != "faiss",
    )
    monkeypatch.setattr(
        helpers,
        "extras_install_hint",
        lambda missing: "pip install 'pyimgano[faiss]'",
    )

    payload = helpers.build_require_extras_check(["torch,faiss"])

    assert payload == {
        "required": ["torch", "faiss"],
        "missing": ["faiss"],
        "ok": False,
        "install_hint": "pip install 'pyimgano[faiss]'",
    }


def test_build_accelerator_checks_returns_json_ready_shape() -> None:
    from pyimgano.services.doctor_service_helpers import build_accelerator_checks

    payload = build_accelerator_checks()

    assert isinstance(payload, dict)
    assert "torch" in payload
    assert "onnxruntime" in payload
    assert "openvino" in payload
