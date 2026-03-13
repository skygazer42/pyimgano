from __future__ import annotations

import json

import pytest


def test_train_cli_dry_run_applies_preprocessing_preset(tmp_path, capsys) -> None:
    from pyimgano.train_cli import main

    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(
        json.dumps(
            {
                "recipe": "industrial-adapt",
                "dataset": {"name": "custom", "root": "/tmp/data"},
                "model": {"name": "vision_patchcore"},
                "output": {"save_run": False},
            }
        ),
        encoding="utf-8",
    )

    code = main(
        [
            "--config",
            str(cfg_path),
            "--dry-run",
            "--preprocessing-preset",
            "illumination-contrast-balanced",
        ]
    )
    assert code == 0

    payload = json.loads(capsys.readouterr().out)
    pre = payload["config"]["preprocessing"]["illumination_contrast"]
    assert pre["white_balance"] == "gray_world"
    assert pre["clahe"] is True


def test_train_cli_dry_run_delegates_overrides_to_workbench_service(
    tmp_path, capsys, monkeypatch
) -> None:
    from pyimgano.train_cli import main
    import pyimgano.services.workbench_service as workbench_service

    calls: list[str] = []

    def _fake_resolve(name: str) -> dict[str, object]:
        calls.append(name)
        return {
            "white_balance": "max_rgb",
            "clahe": True,
            "clahe_clip_limit": 1.5,
        }

    monkeypatch.setattr(
        workbench_service,
        "resolve_preprocessing_preset_payload",
        _fake_resolve,
    )

    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(
        json.dumps(
            {
                "recipe": "industrial-adapt",
                "dataset": {"name": "custom", "root": "/tmp/data"},
                "model": {"name": "vision_patchcore"},
                "output": {"save_run": False},
            }
        ),
        encoding="utf-8",
    )

    code = main(
        [
            "--config",
            str(cfg_path),
            "--dry-run",
            "--preprocessing-preset",
            "illumination-contrast-balanced",
        ]
    )
    assert code == 0
    assert calls == ["illumination-contrast-balanced"]

    payload = json.loads(capsys.readouterr().out)
    pre = payload["config"]["preprocessing"]["illumination_contrast"]
    assert pre["white_balance"] == "max_rgb"
    assert pre["clahe"] is True
    assert pre["clahe_clip_limit"] == pytest.approx(1.5)


def test_train_cli_dry_run_delegates_preprocessing_preset_to_workbench_service(
    tmp_path, capsys, monkeypatch
) -> None:
    from pyimgano.train_cli import main

    import pyimgano.services.workbench_service as workbench_service

    calls: list[dict[str, object]] = []

    def _fake_apply_overrides(
        raw,
        *,
        dataset_name=None,
        root=None,
        category=None,
        model_name=None,
        device=None,
        preprocessing_preset=None,
    ):  # noqa: ANN001 - service seam
        calls.append(
            {
                "dataset_name": dataset_name,
                "root": root,
                "category": category,
                "model_name": model_name,
                "device": device,
                "preprocessing_preset": preprocessing_preset,
            }
        )
        out = dict(raw)
        out["preprocessing"] = {
            "illumination_contrast": {
                "white_balance": "gray_world",
                "clahe": True,
            }
        }
        return out

    monkeypatch.setattr(workbench_service, "apply_workbench_overrides", _fake_apply_overrides)

    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(
        json.dumps(
            {
                "recipe": "industrial-adapt",
                "dataset": {"name": "custom", "root": "/tmp/data"},
                "model": {"name": "vision_patchcore"},
                "output": {"save_run": False},
            }
        ),
        encoding="utf-8",
    )

    code = main(
        [
            "--config",
            str(cfg_path),
            "--dry-run",
            "--preprocessing-preset",
            "illumination-contrast-balanced",
        ]
    )
    assert code == 0
    assert len(calls) == 1
    assert calls[0]["preprocessing_preset"] == "illumination-contrast-balanced"

    payload = json.loads(capsys.readouterr().out)
    pre = payload["config"]["preprocessing"]["illumination_contrast"]
    assert pre["white_balance"] == "gray_world"
    assert pre["clahe"] is True
