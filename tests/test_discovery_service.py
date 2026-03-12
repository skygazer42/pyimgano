from __future__ import annotations

from pyimgano.services.discovery_service import (
    build_model_info_payload,
    build_model_preset_info_payload,
    list_discovery_model_names,
    list_model_preset_infos_payload,
    list_model_preset_names,
)


def test_list_discovery_model_names_supports_family_and_year_filters() -> None:
    names = list_discovery_model_names(family="one-to-normal", year="2025")
    assert isinstance(names, list)
    assert "vision_one_to_normal" in names


def test_build_model_info_payload_returns_json_ready_shape() -> None:
    payload = build_model_info_payload("vision_ecod")
    assert payload["name"] == "vision_ecod"
    assert "accepted_kwargs" in payload
    assert "constructor" in payload


def test_build_model_preset_info_payload_returns_json_ready_shape() -> None:
    payload = build_model_preset_info_payload("industrial-structural-ecod")
    assert payload["name"] == "industrial-structural-ecod"
    assert "model" in payload
    assert "kwargs" in payload


def test_list_model_preset_names_supports_family_filter() -> None:
    names = list_model_preset_names(family="graph")
    assert "industrial-structural-rgraph" in names
    assert "industrial-structural-lof" not in names


def test_list_model_preset_infos_payload_supports_family_filter() -> None:
    payload = list_model_preset_infos_payload(family="distillation")
    assert any(
        item["name"] == "industrial-reverse-distillation" and "distillation" in item["tags"]
        for item in payload
    )
