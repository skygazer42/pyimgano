from __future__ import annotations

import json
from pathlib import Path


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# comment line should be ignored\n")
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_manifest_cli_stats_json(tmp_path: Path, capsys) -> None:
    from pyimgano.manifest_cli import main as manifest_main

    abs_img = tmp_path / "abs.png"
    abs_img.write_bytes(b"\x89PNG\r\n\x1a\n")  # not a real image; stats should not decode

    manifest = tmp_path / "manifest.jsonl"
    _write_jsonl(
        manifest,
        [
            {"image_path": "rel/train0.png", "category": "A", "split": "train"},
            {"image_path": "rel/test_good.png", "category": "A", "split": "test", "label": 0},
            {
                "image_path": "rel/test_bad.png",
                "category": "A",
                "split": "test",
                "label": 1,
                "mask_path": "rel/test_bad_mask.png",
            },
            {"image_path": str(abs_img), "category": "B", "split": "test", "label": 1},
            {"image_path": "rel/other.png", "category": "B"},
        ],
    )

    code = manifest_main(["--stats", "--manifest", str(manifest), "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["total_records"] == 5
    assert payload["category_counts"] == {"A": 3, "B": 2}
    assert payload["split_counts"]["train"] == 1
    assert payload["split_counts"]["test"] == 3
    assert payload["split_counts"]["missing"] == 1
    assert payload["label_counts"]["0"] == 1
    assert payload["label_counts"]["1"] == 2
    assert payload["label_counts"]["missing"] == 2
    assert payload["mask_path_present_count"] == 1
    assert payload["image_path_absolute_count"] == 1


def test_manifest_cli_filter_writes_filtered_jsonl(tmp_path: Path) -> None:
    from pyimgano.manifest_cli import main as manifest_main

    manifest = tmp_path / "manifest.jsonl"
    _write_jsonl(
        manifest,
        [
            {"image_path": "rel/train0.png", "category": "A", "split": "train"},
            {"image_path": "rel/test_good.png", "category": "A", "split": "test", "label": 0},
            {
                "image_path": "rel/test_bad.png",
                "category": "A",
                "split": "test",
                "label": 1,
                "mask_path": "rel/test_bad_mask.png",
            },
            {"image_path": "rel/b_test_bad2.png", "category": "B", "split": "test", "label": 1},
        ],
    )

    out = tmp_path / "filtered.jsonl"
    code = manifest_main(
        [
            "--filter",
            "--manifest",
            str(manifest),
            "--out",
            str(out),
            "--category",
            "A",
            "--split",
            "test",
            "--label",
            "1",
            "--has-mask",
            "true",
        ]
    )
    assert code == 0
    assert out.exists()

    rows = [
        json.loads(line)
        for line in out.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert len(rows) == 1
    assert rows[0]["category"] == "A"
    assert rows[0]["split"] == "test"
    assert int(rows[0]["label"]) == 1
    assert "mask_path" in rows[0]

    # limit keeps stable order (first N matches)
    out2 = tmp_path / "limited.jsonl"
    code2 = manifest_main(
        [
            "--filter",
            "--manifest",
            str(manifest),
            "--out",
            str(out2),
            "--category",
            "A",
            "--limit",
            "2",
        ]
    )
    assert code2 == 0
    rows2 = [
        json.loads(line)
        for line in out2.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert len(rows2) == 2
    assert rows2[0]["image_path"] == "rel/train0.png"
    assert rows2[1]["image_path"] == "rel/test_good.png"
