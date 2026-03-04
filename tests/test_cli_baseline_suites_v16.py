from __future__ import annotations

import json
from pathlib import Path


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
    assert "industrial-v1" in out
    assert "industrial-v2" in out
    assert "industrial-v3" in out


def test_benchmark_cli_can_list_suites_json(capsys) -> None:
    from pyimgano.cli import main as benchmark_main

    rc = benchmark_main(["--list-suites", "--json"])
    assert rc == 0

    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, list)
    assert "industrial-ci" in payload
    assert "industrial-v2" in payload
    assert "industrial-v3" in payload


def test_benchmark_cli_can_list_sweeps_text(capsys) -> None:
    from pyimgano.cli import main as benchmark_main

    rc = benchmark_main(["--list-sweeps"])
    assert rc == 0

    out = capsys.readouterr().out.strip().splitlines()
    assert "industrial-small" in out
    assert "industrial-template-small" in out


def test_benchmark_cli_can_list_sweeps_json(capsys) -> None:
    from pyimgano.cli import main as benchmark_main

    rc = benchmark_main(["--list-sweeps", "--json"])
    assert rc == 0

    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, list)
    assert "industrial-small" in payload


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


def test_benchmark_cli_suite_discovery_flags_are_mutually_exclusive(capsys) -> None:
    from pyimgano.cli import main as benchmark_main

    rc = benchmark_main(["--list-suites", "--list-models"])
    assert rc != 0

    err = capsys.readouterr().err.lower()
    assert "mutually" in err or "exclusive" in err


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
    for base_name, rs in by_base.items():
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
