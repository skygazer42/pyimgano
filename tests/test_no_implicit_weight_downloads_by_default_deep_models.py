from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")


def test_no_implicit_weight_downloads_by_default_for_selected_deep_models(
    monkeypatch,
) -> None:  # noqa: ANN001
    """Industrial guardrail: creating models must not trigger network downloads by default.

    Notes
    -----
    We hard-fail if torchvision/torch.hub attempts to download weights.
    """

    import torch.hub

    def _blocked(*_args, **_kwargs):  # noqa: ANN001, ANN201
        raise AssertionError("Implicit weight downloads are forbidden in unit tests.")

    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", _blocked, raising=True)
    monkeypatch.setattr(torch.hub, "load", _blocked, raising=True)

    from pyimgano.models import create_model

    # Models that should be constructible offline by default (pretrained=False).
    offline_ok = [
        "vision_patchcore",
        "vision_patchcore_lite_map",
        "vision_spade",
        "vision_padim",
        "vision_simplenet",
        "vision_oddoneout",
    ]
    for name in offline_ok:
        det = create_model(name, device="cpu")
        # Some models are lazy and only build backbones on first `fit()` / `extract()`.
        if name in {"vision_oddoneout", "vision_patchcore_lite_map"}:
            import numpy as np

            X = np.zeros((2, 64, 64, 3), dtype=np.uint8)
            det.fit(X)

    # Models that use torch.hub for foundation weights must require explicit opt-in.
    # By default they should fail-fast with a clear message instead of attempting a download.
    for name in ["vision_anomalydino", "vision_superad", "vision_softpatch", "vision_mambaad"]:
        try:
            _ = create_model(name, device="cpu")
        except Exception as exc:  # noqa: BLE001 - guardrail
            msg = str(exc).lower()
            assert "embedder" in msg or "pretrained" in msg or "hub" in msg
        else:
            raise AssertionError(
                f"Expected {name} to require explicit opt-in (embedder or pretrained)."
            )
