from __future__ import annotations

from types import SimpleNamespace

import pytest


def test_legacy_loader_requires_explicit_opt_in() -> None:
    from pyimgano.inference import load_legacy_artifact

    with pytest.raises(ValueError, match="allow_legacy=True"):
        load_legacy_artifact("run", kind="run")


def test_legacy_loader_requires_explicit_supported_kind() -> None:
    from pyimgano.inference import load_legacy_artifact

    with pytest.raises(ValueError, match="kind must be exactly"):
        load_legacy_artifact("model.onnx", kind="onnx", allow_legacy=True)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kind", "source", "expected_request"),
    [
        ("run", "old-run", "FromRunInferContextRequest"),
        ("infer_config", "infer.json", "InferConfigContextRequest"),
    ],
)
def test_legacy_loader_routes_only_through_named_legacy_contract(
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    source: str,
    expected_request: str,
) -> None:
    from pyimgano.inference import LegacyArtifactWarning, load_legacy_artifact
    from pyimgano.services import infer_context_service, infer_load_service

    requests: list[object] = []
    load_requests: list[object] = []
    context = SimpleNamespace()
    monkeypatch.setattr(
        infer_context_service,
        "prepare_from_run_context",
        lambda request: requests.append(request) or context,
    )
    monkeypatch.setattr(
        infer_context_service,
        "prepare_infer_config_context",
        lambda request: requests.append(request) or context,
    )
    detector = object()
    monkeypatch.setattr(
        infer_load_service,
        "load_config_backed_infer_detector",
        lambda request: load_requests.append(request) or SimpleNamespace(detector=detector),
    )

    with pytest.warns(LegacyArtifactWarning):
        actual = load_legacy_artifact(
            source,
            kind=kind,  # type: ignore[arg-type]
            allow_legacy=True,
            category="bottle",
            device="cpu",
            trust_checkpoint=True,
        )

    assert actual is detector
    assert type(requests[0]).__name__ == expected_request
    assert load_requests[0].trust_checkpoint is True
