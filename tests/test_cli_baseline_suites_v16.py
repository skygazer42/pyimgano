from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_custom_dataset(root: Path) -> None:
    import cv2
    import numpy as np

    for rel, value in [
        ("train/normal/train_0.png", 120),
        ("train/normal/train_1.png", 120),
        ("test/normal/good_0.png", 120),
        ("test/anomaly/bad_0.png", 240),
    ]:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        img = np.ones((16, 16, 3), dtype=np.uint8) * int(value)
        cv2.imwrite(str(p), img)


def test_benchmark_cli_can_list_suites_text(capsys) -> None:
    from pyimgano.cli import main as benchmark_main

    rc = benchmark_main(["--list-suites"])
    assert rc == 0

    out = capsys.readouterr().out.strip().splitlines()
    assert "industrial-ci" in out
    assert "industrial-parity-v1" in out
    assert "industrial-v1" in out
    assert "industrial-v2" in out
    assert "industrial-v3" in out
    assert "industrial-v4" in out


def test_benchmark_cli_can_list_suites_json(capsys) -> None:
    from pyimgano.cli import main as benchmark_main

    rc = benchmark_main(["--list-suites", "--json"])
    assert rc == 0

    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, list)
    assert "industrial-ci" in payload
    assert "industrial-parity-v1" in payload
    assert "industrial-v2" in payload
    assert "industrial-v3" in payload
    assert "industrial-v4" in payload


def test_benchmark_cli_can_list_sweeps_text(capsys) -> None:
    from pyimgano.cli import main as benchmark_main

    rc = benchmark_main(["--list-sweeps"])
    assert rc == 0

    out = capsys.readouterr().out.strip().splitlines()
    assert "industrial-small" in out
    assert "industrial-template-small" in out
    assert "industrial-feature-small" in out


def test_benchmark_cli_can_list_sweeps_json(capsys) -> None:
    from pyimgano.cli import main as benchmark_main

    rc = benchmark_main(["--list-sweeps", "--json"])
    assert rc == 0

    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, list)
    assert "industrial-small" in payload
    assert "industrial-feature-small" in payload


def test_benchmark_cli_can_list_official_configs_text(capsys) -> None:
    from pyimgano.cli import main as benchmark_main

    rc = benchmark_main(["--list-official-configs"])
    assert rc == 0

    out = capsys.readouterr().out.strip().splitlines()
    assert "official_manifest_industrial_v4_cpu_offline.json" in out
    assert "official_mvtec_industrial_v4_cpu_offline.json" in out
    assert "official_visa_industrial_v4_cpu_offline.json" in out


def test_benchmark_cli_can_list_official_configs_json(capsys) -> None:
    from pyimgano.cli import main as benchmark_main

    rc = benchmark_main(["--list-official-configs", "--json"])
    assert rc == 0

    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, list)
    by_name = {str(item["name"]): item for item in payload}
    mvtec = by_name["official_mvtec_industrial_v4_cpu_offline.json"]
    assert mvtec["official"] is True
    assert mvtec["dataset"] == "mvtec"
    assert mvtec["suite"] == "industrial-v4"
    assert mvtec["errors"] == []
    assert isinstance(mvtec["sha256"], str)
    assert mvtec["sha256"]


def test_benchmark_cli_can_list_starter_configs_json(capsys) -> None:
    from pyimgano.cli import main as benchmark_main

    rc = benchmark_main(["--list-starter-configs", "--json"])
    assert rc == 0

    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, list)
    by_name = {str(item["name"]): item for item in payload}
    mvtec = by_name["official_mvtec_industrial_v4_cpu_offline.json"]
    assert mvtec["starter"] is True
    assert mvtec["starter_tier"] == "starter"
    assert isinstance(mvtec["estimated_runtime"], str)
    assert mvtec["dataset"] == "mvtec"
    assert mvtec["optional_extras"] == ["clip", "skimage", "torch"]
    assert mvtec["optional_extras_install_hint"] == "pip install 'pyimgano[clip,skimage,torch]'"
    assert mvtec["optional_baseline_count"] == 11
    assert mvtec["starter_list_command"] == "pyimgano benchmark --list-starter-configs"
    assert "starter_info_command" in mvtec
    assert "starter_run_command" in mvtec


def test_benchmark_cli_can_list_starter_configs_text(capsys) -> None:
    from pyimgano.cli import main as benchmark_main

    rc = benchmark_main(["--list-starter-configs"])
    assert rc == 0

    out = capsys.readouterr().out
    assert "official_mvtec_industrial_v4_cpu_offline.json" in out
    assert "dataset=mvtec" in out
    assert "optional_extras=clip,skimage,torch" in out
    assert "optional_baselines=11" in out
    assert "list=pyimgano benchmark --list-starter-configs" in out
    assert "inspect=pyimgano benchmark --starter-config-info official_mvtec_industrial_v4_cpu_offline.json --json" in out


def test_benchmark_cli_starter_config_info_outputs_json(capsys) -> None:
    from pyimgano.cli import main as benchmark_main

    rc = benchmark_main(
        [
            "--starter-config-info",
            "official_mvtec_industrial_v4_cpu_offline.json",
            "--json",
        ]
    )
    assert rc == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["name"] == "official_mvtec_industrial_v4_cpu_offline.json"
    assert payload["starter"] is True
    assert payload["starter_tier"] == "starter"
    assert isinstance(payload["recommended_for"], list)
    assert payload["recommended_for"]
    assert payload["optional_extras"] == ["clip", "skimage", "torch"]
    assert payload["optional_baseline_count"] == 11
    assert payload["starter_list_command"] == "pyimgano benchmark --list-starter-configs"
    assert payload["starter_info_command"].endswith("--json")
    assert payload["starter_run_command"] == "pyimgano-benchmark --config official_mvtec_industrial_v4_cpu_offline.json"


def test_benchmark_cli_starter_config_info_text_surfaces_optional_extras(capsys) -> None:
    from pyimgano.cli import main as benchmark_main

    rc = benchmark_main(
        [
            "--starter-config-info",
            "official_mvtec_industrial_v4_cpu_offline.json",
        ]
    )
    assert rc == 0

    out = capsys.readouterr().out
    assert "Optional extras:" in out
    assert "clip, skimage, torch" in out
    assert "Optional baselines: 11" in out
    assert "Suggested commands:" in out
    assert "pyimgano benchmark --list-starter-configs" in out
    assert "pyimgano benchmark --starter-config-info official_mvtec_industrial_v4_cpu_offline.json --json" in out


def test_benchmark_cli_official_config_info_outputs_json(capsys) -> None:
    from pyimgano.cli import main as benchmark_main

    rc = benchmark_main(
        [
            "--official-config-info",
            "official_mvtec_industrial_v4_cpu_offline.json",
            "--json",
        ]
    )
    assert rc == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["name"] == "official_mvtec_industrial_v4_cpu_offline.json"
    assert payload["official"] is True
    assert payload["dataset"] == "mvtec"
    assert payload["suite"] == "industrial-v4"
    assert payload["kind"] == "file"
    assert payload["errors"] == []
    assert payload["source"].endswith("official_mvtec_industrial_v4_cpu_offline.json")
    assert payload["payload"]["dataset"] == "mvtec"


def test_benchmark_cli_config_can_resolve_official_name_without_full_path(capsys) -> None:
    from pyimgano.cli import main as benchmark_main

    rc = benchmark_main(
        [
            "--config",
            "official_mvtec_industrial_v4_cpu_offline.json",
            "--list-suites",
        ]
    )
    assert rc == 0

    out = capsys.readouterr().out.strip().splitlines()
    assert "industrial-v4" in out


def test_benchmark_cli_sweep_info_outputs_json(capsys) -> None:
    from pyimgano.cli import main as benchmark_main

    rc = benchmark_main(["--sweep-info", "industrial-small", "--json"])
    assert rc == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["name"] == "industrial-small"
    assert "variants_by_entry" in payload
    variants_by_entry = payload["variants_by_entry"]
    assert isinstance(variants_by_entry, dict)
    assert "industrial-template-ncc-map" in variants_by_entry
    assert isinstance(variants_by_entry["industrial-template-ncc-map"], list)


def test_benchmark_cli_suite_info_outputs_json(capsys) -> None:
    from pyimgano.cli import main as benchmark_main

    rc = benchmark_main(["--suite-info", "industrial-ci", "--json"])
    assert rc == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["name"] == "industrial-ci"
    assert "entries" in payload
    assert "baselines" in payload


def test_benchmark_cli_parity_suite_info_outputs_expected_entries(capsys) -> None:
    from pyimgano.cli import main as benchmark_main

    rc = benchmark_main(["--suite-info", "industrial-parity-v1", "--json"])
    assert rc == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["name"] == "industrial-parity-v1"
    entries = payload.get("entries", [])
    assert isinstance(entries, list)
    assert "industrial-template-ncc-map" in entries
    assert "industrial-structural-ecod" in entries
    assert "industrial-embedding-core-balanced" in entries
    assert "vision_patchcore" in entries
    assert "vision_patchcore_anomalib" in entries
    assert "vision_patchcore_inspection_checkpoint" in entries


def test_suite_info_includes_requires_extras_for_optional_baselines(capsys) -> None:
    from pyimgano.cli import main as benchmark_main

    rc = benchmark_main(["--suite-info", "industrial-v1", "--json"])
    assert rc == 0

    payload = json.loads(capsys.readouterr().out)
    baselines = payload.get("baselines", [])
    assert isinstance(baselines, list)

    by_name = {str(b.get("name")): b for b in baselines}
    ssim = by_name.get("industrial-ssim-template-map")
    assert ssim is not None
    assert ssim.get("optional") is True

    req = ssim.get("requires_extras")
    assert isinstance(req, list)
    assert "skimage" in req


def test_suite_info_marks_patchknn_baselines_as_optional_with_torch_extra(capsys) -> None:
    from pyimgano.cli import main as benchmark_main

    rc = benchmark_main(["--suite-info", "industrial-v3", "--json"])
    assert rc == 0

    payload = json.loads(capsys.readouterr().out)
    baselines = payload.get("baselines", [])
    assert isinstance(baselines, list)

    by_name = {str(b.get("name")): b for b in baselines}
    b = by_name.get("industrial-patchcore-lite-map")
    assert b is not None
    assert b.get("optional") is True

    req = b.get("requires_extras")
    assert isinstance(req, list)
    assert "torch" in req


def test_suite_info_marks_texture_baselines_as_optional_with_skimage_extra(capsys) -> None:
    from pyimgano.cli import main as benchmark_main

    rc = benchmark_main(["--suite-info", "industrial-v4", "--json"])
    assert rc == 0

    payload = json.loads(capsys.readouterr().out)
    baselines = payload.get("baselines", [])
    assert isinstance(baselines, list)

    by_name = {str(b.get("name")): b for b in baselines}
    hog = by_name.get("industrial-hog-ecod")
    assert hog is not None
    assert hog.get("optional") is True

    req = hog.get("requires_extras")
    assert isinstance(req, list)
    assert "skimage" in req


def test_benchmark_cli_suite_discovery_flags_are_mutually_exclusive(capsys) -> None:
    from pyimgano.cli import main as benchmark_main

    rc = benchmark_main(["--list-suites", "--list-models"])
    assert rc != 0

    err = capsys.readouterr().err.lower()
    assert "mutually" in err or "exclusive" in err


def test_benchmark_cli_official_config_discovery_flags_are_mutually_exclusive(capsys) -> None:
    from pyimgano.cli import main as benchmark_main

    rc = benchmark_main(["--list-official-configs", "--list-models"])
    assert rc != 0

    err = capsys.readouterr().err.lower()
    assert "mutually" in err or "exclusive" in err


def test_benchmark_cli_suite_mode_delegates_to_benchmark_service(monkeypatch, capsys) -> None:
    import pyimgano.services.benchmark_service as benchmark_service
    from pyimgano.cli import main as benchmark_main
    from pyimgano.services.benchmark_service import PixelPostprocessConfig

    calls = []
    monkeypatch.setattr(
        benchmark_service,
        "run_suite_request",
        lambda request: calls.append(request) or {"suite": request.suite, "rows": []},
    )

    rc = benchmark_main(
        [
            "--dataset",
            "mvtec",
            "--root",
            "/tmp/x",
            "--category",
            "bottle",
            "--suite",
            "industrial-v1",
            "--pixel",
            "--pixel-postprocess",
            "--pixel-post-gaussian-sigma",
            "1.25",
        ]
    )
    assert rc == 0
    assert len(calls) == 1
    assert isinstance(calls[0].pixel_postprocess, PixelPostprocessConfig)
    assert calls[0].pixel_postprocess.gaussian_sigma == pytest.approx(1.25)
    assert '"suite": "industrial-v1"' in capsys.readouterr().out


def test_benchmark_cli_can_run_suite_smoke(tmp_path: Path, capsys) -> None:
    from pyimgano.cli import main as benchmark_main

    root = tmp_path / "custom"
    _write_custom_dataset(root)

    out_dir = tmp_path / "suite_out"
    rc = benchmark_main(
        [
            "--dataset",
            "custom",
            "--root",
            str(root),
            "--suite",
            "industrial-ci",
            "--device",
            "cpu",
            "--no-pretrained",
            "--resize",
            "64",
            "64",
            "--limit-train",
            "2",
            "--limit-test",
            "2",
            "--output-dir",
            str(out_dir),
        ]
    )
    assert rc == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["suite"] == "industrial-ci"
    assert Path(payload["run_dir"]).exists()
    assert (out_dir / "report.json").exists()
    assert isinstance(payload.get("rows"), list)


def test_benchmark_cli_suite_include_filters_baselines(tmp_path: Path, capsys) -> None:
    from pyimgano.cli import main as benchmark_main

    root = tmp_path / "custom"
    _write_custom_dataset(root)

    rc = benchmark_main(
        [
            "--dataset",
            "custom",
            "--root",
            str(root),
            "--suite",
            "industrial-ci",
            "--suite-include",
            "industrial-template-ncc-map",
            "--device",
            "cpu",
            "--no-pretrained",
            "--resize",
            "64",
            "64",
            "--limit-train",
            "2",
            "--limit-test",
            "2",
            "--no-save-run",
        ]
    )
    assert rc == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload.get("baselines") == ["industrial-template-ncc-map"]
    rows = payload.get("rows", [])
    assert isinstance(rows, list)
    assert len(rows) == 1
    assert rows[0].get("base_name") == "industrial-template-ncc-map"


def test_benchmark_cli_suite_exclude_filters_baselines(tmp_path: Path, capsys) -> None:
    from pyimgano.cli import main as benchmark_main

    root = tmp_path / "custom"
    _write_custom_dataset(root)

    rc = benchmark_main(
        [
            "--dataset",
            "custom",
            "--root",
            str(root),
            "--suite",
            "industrial-ci",
            "--suite-exclude",
            "industrial-template-ncc-map",
            "--device",
            "cpu",
            "--no-pretrained",
            "--resize",
            "64",
            "64",
            "--limit-train",
            "2",
            "--limit-test",
            "2",
            "--no-save-run",
        ]
    )
    assert rc == 0

    payload = json.loads(capsys.readouterr().out)
    baselines = payload.get("baselines")
    assert isinstance(baselines, list)
    assert "industrial-template-ncc-map" not in baselines


def test_benchmark_cli_suite_include_unknown_baseline_errors(tmp_path: Path, capsys) -> None:
    from pyimgano.cli import main as benchmark_main

    root = tmp_path / "custom"
    _write_custom_dataset(root)

    rc = benchmark_main(
        [
            "--dataset",
            "custom",
            "--root",
            str(root),
            "--suite",
            "industrial-ci",
            "--suite-include",
            "not-a-baseline",
            "--device",
            "cpu",
            "--no-pretrained",
            "--resize",
            "64",
            "64",
            "--limit-train",
            "2",
            "--limit-test",
            "2",
            "--no-save-run",
        ]
    )
    assert rc != 0

    err = capsys.readouterr().err.lower()
    assert "unknown" in err
    assert "suite-include" in err


def test_benchmark_cli_suite_export_csv_writes_leaderboard(tmp_path: Path, capsys) -> None:
    import json

    from pyimgano.cli import main as benchmark_main

    root = tmp_path / "custom"
    _write_custom_dataset(root)

    out_dir = tmp_path / "suite_out"
    rc = benchmark_main(
        [
            "--dataset",
            "custom",
            "--root",
            str(root),
            "--suite",
            "industrial-ci",
            "--suite-export",
            "csv",
            "--device",
            "cpu",
            "--no-pretrained",
            "--resize",
            "64",
            "64",
            "--limit-train",
            "2",
            "--limit-test",
            "2",
            "--output-dir",
            str(out_dir),
        ]
    )
    assert rc == 0
    _ = capsys.readouterr()

    assert (out_dir / "leaderboard.csv").exists()
    assert (out_dir / "best_by_baseline.csv").exists()
    assert (out_dir / "skipped.csv").exists()
    metadata = json.loads((out_dir / "leaderboard_metadata.json").read_text(encoding="utf-8"))
    assert len(str(metadata["split_fingerprint"]["sha256"])) == 64
    assert metadata["dataset_readiness"]["status"] == "warning"
    assert metadata["dataset_readiness"]["issue_codes"] == [
        "PIXEL_METRICS_UNAVAILABLE",
        "FEWSHOT_TRAIN_SET",
    ]
    assert metadata["benchmark_context"]["dataset_readiness_status"] == "warning"
    assert metadata["benchmark_context"]["dataset_issue_codes"] == [
        "PIXEL_METRICS_UNAVAILABLE",
        "FEWSHOT_TRAIN_SET",
    ]


def test_benchmark_cli_suite_export_best_metric_requires_pixel(tmp_path: Path, capsys) -> None:
    from pyimgano.cli import main as benchmark_main

    root = tmp_path / "custom"
    _write_custom_dataset(root)

    out_dir = tmp_path / "suite_out"
    rc = benchmark_main(
        [
            "--dataset",
            "custom",
            "--root",
            str(root),
            "--suite",
            "industrial-ci",
            "--suite-export",
            "csv",
            "--suite-export-best-metric",
            "pixel_auroc",
            "--device",
            "cpu",
            "--no-pretrained",
            "--resize",
            "64",
            "64",
            "--limit-train",
            "2",
            "--limit-test",
            "2",
            "--output-dir",
            str(out_dir),
        ]
    )
    assert rc != 0

    err = capsys.readouterr().err.lower()
    assert "requires --pixel" in err


def test_benchmark_cli_can_run_suite_sweep_smoke(tmp_path: Path, capsys) -> None:
    from pyimgano.cli import main as benchmark_main

    root = tmp_path / "custom"
    _write_custom_dataset(root)

    rc = benchmark_main(
        [
            "--dataset",
            "custom",
            "--root",
            str(root),
            "--suite",
            "industrial-ci",
            "--suite-sweep",
            "industrial-small",
            "--device",
            "cpu",
            "--no-pretrained",
            "--resize",
            "64",
            "64",
            "--limit-train",
            "2",
            "--limit-test",
            "2",
            "--no-save-run",
        ]
    )
    assert rc == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload.get("suite") == "industrial-ci"
    sweep = payload.get("sweep")
    assert isinstance(sweep, dict)
    assert sweep.get("name") == "industrial-small"

    rows = payload.get("rows")
    assert isinstance(rows, list)
    assert len(rows) > 0


def test_benchmark_cli_can_run_suite_custom_sweep_json_file(tmp_path: Path, capsys) -> None:
    from pyimgano.cli import main as benchmark_main

    root = tmp_path / "custom"
    _write_custom_dataset(root)

    sweep_path = tmp_path / "my_sweep.json"
    sweep_path.write_text(
        json.dumps(
            {
                "name": "my-sweep",
                "description": "Tiny custom sweep for NCC window sizes",
                "variants_by_entry": {
                    "industrial-template-ncc-map": [
                        {"name": "win_7", "override": {"window_hw": [7, 7]}},
                        {"name": "win_21", "override": {"window_hw": [21, 21]}},
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    rc = benchmark_main(
        [
            "--dataset",
            "custom",
            "--root",
            str(root),
            "--suite",
            "industrial-ci",
            "--suite-sweep",
            str(sweep_path),
            "--device",
            "cpu",
            "--no-pretrained",
            "--resize",
            "64",
            "64",
            "--limit-train",
            "2",
            "--limit-test",
            "2",
            "--no-save-run",
        ]
    )
    assert rc == 0

    payload = json.loads(capsys.readouterr().out)
    sweep = payload.get("sweep")
    assert isinstance(sweep, dict)
    assert sweep.get("name") == "my-sweep"

    rows = payload.get("rows")
    assert isinstance(rows, list)
    assert len(rows) == 5  # 3 baselines + 2 variants on NCC baseline.

    names = {str(r.get("name")) for r in rows if isinstance(r, dict)}
    assert "industrial-template-ncc-map__win_7" in names
    assert "industrial-template-ncc-map__win_21" in names


def test_benchmark_cli_custom_sweep_info_outputs_json(tmp_path: Path, capsys) -> None:
    from pyimgano.cli import main as benchmark_main

    sweep_path = tmp_path / "my_sweep.json"
    sweep_path.write_text(
        json.dumps(
            {
                "name": "my-sweep",
                "description": "Tiny custom sweep for NCC window sizes",
                "variants_by_entry": {
                    "industrial-template-ncc-map": [
                        {"name": "win_7", "override": {"window_hw": [7, 7]}},
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    rc = benchmark_main(["--sweep-info", str(sweep_path), "--json"])
    assert rc == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["name"] == "my-sweep"
    assert "variants_by_entry" in payload
    assert "industrial-template-ncc-map" in payload["variants_by_entry"]


def test_benchmark_cli_suite_sweep_max_variants_caps_rows(tmp_path: Path, capsys) -> None:
    from pyimgano.cli import main as benchmark_main

    root = tmp_path / "custom"
    _write_custom_dataset(root)

    rc = benchmark_main(
        [
            "--dataset",
            "custom",
            "--root",
            str(root),
            "--suite",
            "industrial-ci",
            "--suite-sweep",
            "industrial-small",
            "--suite-sweep-max-variants",
            "1",
            "--device",
            "cpu",
            "--no-pretrained",
            "--resize",
            "64",
            "64",
            "--limit-train",
            "2",
            "--limit-test",
            "2",
            "--no-save-run",
        ]
    )
    assert rc == 0

    payload = json.loads(capsys.readouterr().out)
    rows = payload.get("rows")
    assert isinstance(rows, list)
    # industrial-ci has 3 baselines; cap=1 => base + 1 variant per baseline.
    assert len(rows) == 6

    by_base: dict[str, list[dict]] = {}
    for r in rows:
        base_name = str(r.get("base_name"))
        by_base.setdefault(base_name, []).append(r)
        assert r.get("variant") is not None

    assert set(by_base.keys()) == {
        "industrial-structural-ecod",
        "industrial-pixel-mean-absdiff-map",
        "industrial-template-ncc-map",
    }
    for _base_name, rs in by_base.items():
        variants = [str(r.get("variant")) for r in rs]
        assert variants.count("base") == 1
        assert len(rs) == 2
        assert any(v != "base" for v in variants)


def test_suite_skips_optional_baselines_when_extras_missing(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    import pyimgano.pipelines.run_suite as run_suite
    from pyimgano.cli import main as benchmark_main

    def _fake_can_import_root(root: str) -> bool:
        if root in ("skimage", "torch"):
            return False
        return True

    monkeypatch.setattr(run_suite, "_can_import_root", _fake_can_import_root)

    root = tmp_path / "custom"
    _write_custom_dataset(root)

    rc = benchmark_main(
        [
            "--dataset",
            "custom",
            "--root",
            str(root),
            "--suite",
            "industrial-v1",
            "--device",
            "cpu",
            "--no-pretrained",
            "--resize",
            "64",
            "64",
            "--limit-train",
            "2",
            "--limit-test",
            "2",
            "--no-save-run",
        ]
    )
    assert rc == 0

    payload = json.loads(capsys.readouterr().out)
    skipped = payload.get("skipped", {})
    assert isinstance(skipped, dict)

    assert "industrial-ssim-template-map" in skipped
    assert "pyimgano[skimage]" in str(skipped["industrial-ssim-template-map"].get("reason", ""))

    assert "industrial-embed-knn-cosine" in skipped
    assert "pyimgano[torch]" in str(skipped["industrial-embed-knn-cosine"].get("reason", ""))


def test_suite_torch_extra_requires_torchvision_for_skip_hints(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    """Suite skip hints should reflect the actual `pyimgano[torch]` extra (torch + torchvision)."""

    import pyimgano.pipelines.run_suite as run_suite
    from pyimgano.cli import main as benchmark_main

    def _fake_can_import_root(root: str) -> bool:
        # Simulate an environment where torch is present but torchvision is missing.
        if root == "torchvision":
            return False
        return True

    monkeypatch.setattr(run_suite, "_can_import_root", _fake_can_import_root)

    root = tmp_path / "custom"
    _write_custom_dataset(root)

    rc = benchmark_main(
        [
            "--dataset",
            "custom",
            "--root",
            str(root),
            "--suite",
            "industrial-v1",
            "--device",
            "cpu",
            "--no-pretrained",
            "--resize",
            "64",
            "64",
            "--limit-train",
            "2",
            "--limit-test",
            "2",
            "--no-save-run",
        ]
    )
    assert rc == 0

    payload = json.loads(capsys.readouterr().out)
    skipped = payload.get("skipped", {})
    assert isinstance(skipped, dict)

    assert "industrial-embed-knn-cosine" in skipped
    assert "pyimgano[torch]" in str(skipped["industrial-embed-knn-cosine"].get("reason", ""))


def test_suite_onnx_extra_requires_onnxscript_for_skip_hints(monkeypatch) -> None:
    """Suite skip hints should reflect the actual `pyimgano[onnx]` extra contents."""

    import pyimgano.pipelines.run_suite as run_suite
    from pyimgano.baselines.suites import Baseline

    def _fake_can_import_root(root: str) -> bool:
        if root == "onnxscript":
            return False
        return True

    monkeypatch.setattr(run_suite, "_can_import_root", _fake_can_import_root)

    b = Baseline(
        name="dummy",
        model="dummy",
        kwargs={},
        description="dummy",
        optional=True,
        requires_extras=("onnx",),
    )

    hint = run_suite._missing_extras_hint_for_baseline(b)
    assert "pyimgano[onnx]" in str(hint)
