from __future__ import annotations

import numpy as np
from PIL import Image

import pyimgano.infer_cli as infer_cli


def _write_png(path) -> None:  # noqa: ANN001 - test helper
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    Image.fromarray(img, mode="RGB").save(path)


def test_infer_cli_applies_preprocessing_preset_for_direct_model_mode(
    tmp_path, monkeypatch
) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    _write_png(input_dir / "a.png")
    _write_png(input_dir / "b.png")

    class _NumpyOnlyDetector:
        def __init__(self):
            self.seen_numpy = False

        def decision_function(self, X):  # noqa: ANN001
            items = list(X)
            assert items
            assert all(isinstance(x, np.ndarray) for x in items)
            self.seen_numpy = True
            return np.linspace(0.0, 1.0, num=len(items), dtype=np.float32)

    det = _NumpyOnlyDetector()
    monkeypatch.setattr(infer_cli, "create_model", lambda name, **kwargs: det)

    rc = infer_cli.main(
        [
            "--model",
            "vision_patchcore",
            "--device",
            "cpu",
            "--no-pretrained",
            "--preprocessing-preset",
            "illumination-contrast-balanced",
            "--input",
            str(input_dir),
        ]
    )
    assert rc == 0
    assert det.seen_numpy is True


def test_infer_cli_preprocessing_preset_wrapper_delegates_to_infer_options_service(
    monkeypatch,
) -> None:
    from types import SimpleNamespace

    import pyimgano.infer_cli as infer_cli
    import pyimgano.services.infer_options_service as infer_options_service

    expected = object()
    calls: list[str] = []
    monkeypatch.setattr(
        infer_options_service,
        "resolve_preprocessing_preset_knobs",
        lambda name: calls.append(str(name)) or expected,
    )

    args = SimpleNamespace(preprocessing_preset="illumination-contrast-balanced")

    assert infer_cli._resolve_preprocessing_preset_knobs(args) is expected
    assert calls == ["illumination-contrast-balanced"]
