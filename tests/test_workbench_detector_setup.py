from __future__ import annotations

from pyimgano.workbench.config import WorkbenchConfig
from pyimgano.workbench.detector_setup import build_workbench_runtime_detector


def test_workbench_detector_setup_builds_detector_tiling_and_preprocessing(monkeypatch) -> None:
    import pyimgano.inference.preprocessing as preprocessing_module
    import pyimgano.services.workbench_service as workbench_service
    import pyimgano.workbench.detector_setup as detector_setup

    calls: list[tuple[str, object]] = []
    base_detector = object()
    tiled_detector = object()
    wrapped_detector = object()

    def _fake_create_workbench_detector(*, config):  # noqa: ANN001
        calls.append(("create", config))
        return base_detector

    def _fake_apply_tiling(detector, tiling):  # noqa: ANN001
        calls.append(("tiling", detector))
        assert detector is base_detector
        return tiled_detector

    class _FakePreprocessingDetector:
        def __init__(self, *, detector, illumination_contrast) -> None:  # noqa: ANN001
            calls.append(("preprocessing", detector))
            assert detector is tiled_detector
            self.detector = wrapped_detector

        def __getattr__(self, name: str):  # pragma: no cover - helper shim
            return getattr(self.detector, name)

    monkeypatch.setattr(workbench_service, "create_workbench_detector", _fake_create_workbench_detector)
    monkeypatch.setattr(detector_setup, "apply_tiling", _fake_apply_tiling)
    monkeypatch.setattr(preprocessing_module, "PreprocessingDetector", _FakePreprocessingDetector)

    cfg = WorkbenchConfig.from_dict(
        {
            "recipe": "industrial-adapt",
            "dataset": {"name": "custom", "root": "/tmp/data", "category": "custom"},
            "model": {"name": "vision_ecod"},
            "preprocessing": {"illumination_contrast": {"white_balance": "gray_world"}},
            "output": {"save_run": False},
        }
    )

    detector = build_workbench_runtime_detector(config=cfg)

    assert len(calls) == 3
    assert calls[0][0] == "create"
    assert calls[1] == ("tiling", base_detector)
    assert calls[2] == ("preprocessing", tiled_detector)
    assert detector is not tiled_detector


def test_workbench_detector_setup_skips_preprocessing_when_not_configured(monkeypatch) -> None:
    import pyimgano.services.workbench_service as workbench_service
    import pyimgano.workbench.detector_setup as detector_setup

    calls: list[str] = []
    base_detector = object()
    tiled_detector = object()

    def _fake_create_workbench_detector(*, config):  # noqa: ANN001
        calls.append("create")
        return base_detector

    def _fake_apply_tiling(detector, tiling):  # noqa: ANN001
        calls.append("tiling")
        assert detector is base_detector
        return tiled_detector

    monkeypatch.setattr(workbench_service, "create_workbench_detector", _fake_create_workbench_detector)
    monkeypatch.setattr(detector_setup, "apply_tiling", _fake_apply_tiling)

    cfg = WorkbenchConfig.from_dict(
        {
            "recipe": "industrial-adapt",
            "dataset": {"name": "custom", "root": "/tmp/data", "category": "custom"},
            "model": {"name": "vision_ecod"},
            "output": {"save_run": False},
        }
    )

    detector = build_workbench_runtime_detector(config=cfg)

    assert calls == ["create", "tiling"]
    assert detector is tiled_detector
