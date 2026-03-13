from __future__ import annotations

from pyimgano.workbench.config import WorkbenchConfig
from pyimgano.workbench.infer_config_payload import build_workbench_infer_config_payload


def test_workbench_infer_config_payload_delegates_to_service_boundary(monkeypatch) -> None:
    import pyimgano.workbench.infer_config_payload as payload_module

    calls: list[dict[str, object]] = []

    def _fake_build_infer_config_payload(*, config, report):  # noqa: ANN001 - test seam
        calls.append({"config": config, "report": dict(report)})
        return {"sentinel": True, "threshold": report.get("threshold")}

    monkeypatch.setattr(
        payload_module.workbench_service,
        "build_infer_config_payload",
        _fake_build_infer_config_payload,
    )

    cfg = WorkbenchConfig.from_dict(
        {
            "dataset": {"name": "custom", "root": ".", "category": "custom"},
            "model": {"name": "vision_ecod"},
            "output": {"save_run": False},
        }
    )

    payload = build_workbench_infer_config_payload(config=cfg, report={"threshold": 0.5})

    assert payload == {"sentinel": True, "threshold": 0.5}
    assert len(calls) == 1
    assert calls[0]["config"] == cfg
    assert calls[0]["report"] == {"threshold": 0.5}
