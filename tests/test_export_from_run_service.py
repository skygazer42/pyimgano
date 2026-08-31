from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest


def _source(run_dir: Path, category: str) -> SimpleNamespace:
    return SimpleNamespace(run_dir=run_dir, category=category)


def test_export_from_run_publishes_all_categories_in_one_hashed_index(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import pyimgano.services.export_service as service
    from pyimgano.artifacts import category_slug, load_export_index

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    sources = (_source(run_dir, "bottle"), _source(run_dir, "轴承"))
    monkeypatch.setattr(service, "prepare_run_export_sources", lambda *_args, **_kwargs: sources)
    monkeypatch.setattr(
        service,
        "_static_capabilities",
        lambda _source, formats: ({name: object() for name in formats}, []),
    )
    restored: list[str] = []
    monkeypatch.setattr(
        service,
        "_load_restored_detector",
        lambda source, **_kwargs: restored.append(source.category) or object(),
    )

    def _export_one(*, source, format_name, target, **_kwargs):
        target.mkdir(parents=True)
        manifest = target / "artifact_manifest.json"
        manifest.write_text("{}", encoding="utf-8")
        digest_char = "a" if source.category == "bottle" else "b"
        return {
            "format": format_name,
            "backend": "pyimgano",
            "path": str(target),
            "manifest": str(manifest),
            "artifact_id": f"sha256:{digest_char * 64}",
        }

    monkeypatch.setattr(service, "_export_one", _export_one)
    destination = tmp_path / "portable"
    result = service.export_from_run(
        run_dir=run_dir,
        formats=("native",),
        out_dir=destination,
    )

    assert result["status"] == "ok"
    assert result["category"] is None
    assert result["categories"] == ["bottle", "轴承"]
    assert restored == ["bottle", "轴承"]
    index = load_export_index(destination / "export_index.json")
    assert [(item["category"], item["slug"]) for item in index["entries"]] == [
        ("bottle", "bottle"),
        ("轴承", category_slug("轴承")),
    ]


def test_strict_multi_category_preflight_never_loads_any_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import pyimgano.services.export_service as service

    sources = (_source(tmp_path, "a"), _source(tmp_path, "b"))
    monkeypatch.setattr(service, "prepare_run_export_sources", lambda *_args, **_kwargs: sources)

    def _capability(source, formats):
        if source.category == "b":
            return {}, [{"format": formats[0], "reason": "unsupported"}]
        return {formats[0]: object()}, []

    monkeypatch.setattr(service, "_static_capabilities", _capability)
    monkeypatch.setattr(
        service,
        "_load_restored_detector",
        lambda *_args, **_kwargs: pytest.fail("checkpoint loader must not run"),
    )

    with pytest.raises(service.ExportServiceError, match="before checkpoint loading"):
        service.export_from_run(run_dir=tmp_path, formats=("native",), strict=True)


def test_prepare_sources_discovers_every_report_category_before_loading(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import pyimgano.services.export_service as service
    from pyimgano.services import workbench_run_service

    monkeypatch.setattr(
        workbench_run_service,
        "load_report_from_run",
        lambda _path: {"per_category": {"轴承": {}, "bottle": {}}},
    )
    calls: list[str | None] = []
    monkeypatch.setattr(
        service,
        "prepare_run_export_source",
        lambda run_dir, *, category=None: calls.append(category)
        or _source(Path(run_dir), str(category)),
    )

    sources = service.prepare_run_export_sources(tmp_path)

    assert [source.category for source in sources] == ["bottle", "轴承"]
    assert calls == ["bottle", "轴承"]
