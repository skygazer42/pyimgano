from __future__ import annotations

import json
from pathlib import Path

import pytest


def _entry(
    *,
    category: str = "bottle",
    slug: str = "bottle",
    artifact: str = "bottle/native",
) -> dict[str, object]:
    return {
        "category": category,
        "slug": slug,
        "format": "native",
        "backend": "pyimgano",
        "artifact": artifact,
        "manifest": f"{artifact}/artifact_manifest.json",
        "artifact_id": f"sha256:{'a' * 64}",
    }


def test_export_index_is_content_addressed_and_supports_unicode_categories() -> None:
    from pyimgano.artifacts import build_export_index, category_slug, validate_export_index

    category = "轴承"
    payload = build_export_index(
        [
            _entry(
                category=category,
                slug=category_slug(category),
                artifact=f"{category_slug(category)}/native",
            )
        ]
    )

    assert payload["index_id"].startswith("sha256:")
    assert validate_export_index(payload) == payload
    payload["entries"][0]["backend"] = "onnxruntime"
    with pytest.raises(ValueError, match="index_id"):
        validate_export_index(payload)


def test_export_index_rejects_casefold_category_collisions() -> None:
    from pyimgano.artifacts import build_export_index

    second = _entry(category="BOTTLE", slug="BOTTLE", artifact="BOTTLE/native")
    second["artifact_id"] = f"sha256:{'b' * 64}"
    with pytest.raises(ValueError, match="collide"):
        build_export_index([_entry(), second])


@pytest.mark.parametrize("category", ["CON", "CON.txt", "com1.json", "LPT9.log"])
def test_category_slug_never_emits_windows_device_names(category: str) -> None:
    from pyimgano.artifacts import category_slug

    slug = category_slug(category)

    assert slug != category
    assert slug.split(".", 1)[0].upper() not in {"CON", "COM1", "LPT9"}


def test_export_index_loader_checks_contained_references(tmp_path: Path) -> None:
    from pyimgano.artifacts import load_export_index, write_export_index

    manifest = tmp_path / "bottle" / "native" / "artifact_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}", encoding="utf-8")
    path = write_export_index(tmp_path / "export_index.json", [_entry()])

    assert load_export_index(path)["entries"][0]["manifest"] == (
        "bottle/native/artifact_manifest.json"
    )
    manifest.unlink()
    with pytest.raises(ValueError, match="missing"):
        load_export_index(path)


def test_artifact_resolver_rejects_index_manifest_identity_conflict(tmp_path: Path) -> None:
    from pyimgano.artifacts import write_export_index
    from pyimgano.inference.artifact_runtime import ArtifactRuntimeError
    from pyimgano.services.artifact_load_service import _resolve_artifact_source

    manifest = tmp_path / "bottle" / "native" / "artifact_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "layout": "native_detector",
                "artifact_id": f"sha256:{'a' * 64}",
                "model": {"category": "cable"},
                "runtime": {"backend": "pyimgano"},
                "components": [],
            }
        ),
        encoding="utf-8",
    )
    write_export_index(tmp_path / "export_index.json", [_entry()])

    with pytest.raises(ArtifactRuntimeError, match="category.*conflicts"):
        _resolve_artifact_source(
            tmp_path,
            category="bottle",
            artifact_format="native",
            backend="pyimgano",
            artifact_id=None,
        )


@pytest.mark.parametrize("requested_format", ["openvino", "openvino-ir"])
def test_artifact_resolver_normalizes_openvino_index_format_alias(
    tmp_path: Path,
    requested_format: str,
) -> None:
    from pyimgano.artifacts import category_slug, write_export_index
    from pyimgano.services.artifact_load_service import _resolve_artifact_source

    artifact = "bottle/openvino"
    manifest = tmp_path / artifact / "artifact_manifest.json"
    manifest.parent.mkdir(parents=True)
    artifact_id = f"sha256:{'c' * 64}"
    manifest.write_text(
        json.dumps(
            {
                "layout": "single_graph",
                "artifact_id": artifact_id,
                "model": {"category": "bottle"},
                "runtime": {"backend": "openvino"},
                "components": [
                    {
                        "role": "runtime_model",
                        "format": "openvino-ir",
                        "path": "model/detector.xml",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    write_export_index(
        tmp_path / "export_index.json",
        [
            {
                "category": "bottle",
                "slug": category_slug("bottle"),
                "format": "openvino",
                "backend": "openvino",
                "artifact": artifact,
                "manifest": f"{artifact}/artifact_manifest.json",
                "artifact_id": artifact_id,
            }
        ],
    )

    assert (
        _resolve_artifact_source(
            tmp_path,
            category="bottle",
            artifact_format=requested_format,
            backend="openvino",
            artifact_id=None,
        )
        == manifest
    )
