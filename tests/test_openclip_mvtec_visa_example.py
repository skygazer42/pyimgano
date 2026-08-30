from __future__ import annotations

import pytest

from examples import openclip_mvtec_visa


@pytest.mark.parametrize("allow_download", [False, True])
def test_openclip_example_forwards_explicit_download_gate(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    allow_download: bool,
) -> None:
    captured: dict[str, object] = {}
    split = object()

    def fake_create_model(name: str, **kwargs: object) -> object:
        captured["name"] = name
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(openclip_mvtec_visa, "create_model", fake_create_model)
    monkeypatch.setattr(openclip_mvtec_visa, "load_benchmark_split", lambda **_kwargs: split)
    monkeypatch.setattr(
        openclip_mvtec_visa,
        "evaluate_split",
        lambda detector, resolved_split, **_kwargs: {
            "detector_seen": detector is not None,
            "split_seen": resolved_split is split,
        },
    )

    argv = ["--dataset", "mvtec", "--root", "/dataset", "--category", "bottle"]
    if allow_download:
        argv.append("--allow-download")

    assert openclip_mvtec_visa.main(argv) == 0
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["allow_download"] is allow_download
    assert '"split_seen": true' in capsys.readouterr().out
