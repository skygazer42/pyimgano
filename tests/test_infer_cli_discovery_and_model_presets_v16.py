from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    Image.fromarray(img, mode="RGB").save(path)


class _DummyDetector:
    def __init__(self) -> None:
        self.threshold_ = 0.5

    def decision_function(self, x):  # noqa: ANN001, ANN201 - test stub
        return np.linspace(0.0, 1.0, num=len(x), dtype=np.float32)


def test_infer_cli_can_list_models(capsys) -> None:
    from pyimgano.infer_cli import main as infer_main

    rc = infer_main(["--list-models"])
    assert rc == 0

    out = capsys.readouterr().out.strip().splitlines()
    assert "vision_ecod" in out


def test_infer_cli_list_models_delegates_to_discovery_service(monkeypatch, capsys) -> None:
    import pyimgano.services.discovery_service as discovery_service
    from pyimgano.infer_cli import main as infer_main

    monkeypatch.setattr(
        discovery_service,
        "list_discovery_model_names",
        lambda **_kwargs: ["delegated-model"],
    )

    rc = infer_main(["--list-models"])
    assert rc == 0
    assert capsys.readouterr().out.strip().splitlines() == ["delegated-model"]


def test_infer_cli_can_list_models_by_year_and_type(capsys) -> None:
    from pyimgano.infer_cli import main as infer_main

    rc = infer_main(["--list-models", "--year", "2021", "--type", "deep-vision"])
    assert rc == 0

    out = capsys.readouterr().out.strip().splitlines()
    assert "cutpaste" in out
    assert "core_qmcd" not in out


def test_infer_cli_can_list_models_by_specific_method_type(capsys) -> None:
    from pyimgano.infer_cli import main as infer_main

    rc = infer_main(["--list-models", "--type", "one-class-svm", "--year", "2001"])
    assert rc == 0

    out = capsys.readouterr().out.strip().splitlines()
    assert "vision_ocsvm" in out
    assert "vision_lof" not in out


def test_infer_cli_can_list_models_by_density_estimation_type(capsys) -> None:
    from pyimgano.infer_cli import main as infer_main

    rc = infer_main(["--list-models", "--type", "density-estimation", "--year", "2019"])
    assert rc == 0

    out = capsys.readouterr().out.strip().splitlines()
    assert "vision_rkde_anomalib" in out


def test_infer_cli_can_list_models_by_new_parallel_family(capsys) -> None:
    from pyimgano.infer_cli import main as infer_main

    rc = infer_main(["--list-models", "--family", "one-to-normal", "--year", "2025"])
    assert rc == 0

    out = capsys.readouterr().out.strip().splitlines()
    assert "vision_one_to_normal" in out
    assert "vision_anogen_adapter" not in out


def test_infer_cli_can_list_model_presets(capsys) -> None:
    from pyimgano.infer_cli import main as infer_main

    rc = infer_main(["--list-model-presets"])
    assert rc == 0

    out = capsys.readouterr().out.strip().splitlines()
    assert "industrial-structural-ecod" in out


def test_infer_cli_can_list_model_presets_by_family(capsys) -> None:
    from pyimgano.infer_cli import main as infer_main

    rc = infer_main(["--list-model-presets", "--family", "graph"])
    assert rc == 0

    out = capsys.readouterr().out.strip().splitlines()
    assert "industrial-structural-rgraph" in out
    assert "industrial-structural-lof" not in out


def test_infer_cli_can_list_model_presets_by_family_as_json(capsys) -> None:
    import json

    from pyimgano.infer_cli import main as infer_main

    rc = infer_main(["--list-model-presets", "--family", "distillation", "--json"])
    assert rc == 0

    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, list)
    assert any(
        item["name"] == "industrial-reverse-distillation" and "distillation" in item["tags"]
        for item in payload
    )


def test_infer_cli_model_preset_listing_uses_shared_listing_helper(monkeypatch) -> None:
    import pyimgano.infer_cli as infer_cli
    import pyimgano.services.discovery_service as discovery_service

    monkeypatch.setattr(
        discovery_service,
        "list_model_preset_names",
        lambda **_kwargs: ["preset-a", "preset-b"],
    )
    monkeypatch.setattr(
        discovery_service,
        "list_model_preset_infos_payload",
        lambda **_kwargs: [{"name": "preset-a", "tags": ["graph"]}],
    )

    calls = []
    monkeypatch.setattr(
        infer_cli,
        "cli_listing",
        type(
            "_StubCliListing",
            (),
            {
                "emit_listing": staticmethod(
                    lambda items, **kwargs: calls.append((list(items), kwargs)) or 67
                )
            },
        ),
        raising=False,
    )

    rc = infer_cli.main(["--list-model-presets", "--json"])
    assert rc == 67
    assert calls == [
        (
            ["preset-a", "preset-b"],
            {
                "json_output": True,
                "json_payload": [{"name": "preset-a", "tags": ["graph"]}],
                "sort_keys": False,
            },
        )
    ]


def test_infer_cli_model_preset_listing_delegates_raw_filters_to_service(monkeypatch) -> None:
    import pyimgano.infer_cli as infer_cli
    import pyimgano.services.discovery_service as discovery_service

    service_calls: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        discovery_service,
        "list_model_preset_names",
        lambda **kwargs: service_calls.append(("names", dict(kwargs))) or ["preset-a"],
    )
    monkeypatch.setattr(
        discovery_service,
        "list_model_preset_infos_payload",
        lambda **kwargs: service_calls.append(("infos", dict(kwargs))) or [{"name": "preset-a"}],
    )
    monkeypatch.setattr(
        infer_cli,
        "cli_listing",
        type(
            "_StubCliListing",
            (),
            {"emit_listing": staticmethod(lambda items, **kwargs: 89)},
        ),
        raising=False,
    )

    rc = infer_cli.main(
        ["--list-model-presets", "--family", "graph", "--tags", "embeddings,gaussian", "--json"]
    )
    assert rc == 89
    assert service_calls == [
        ("names", {"tags": ["embeddings,gaussian"], "family": "graph"}),
        ("infos", {"tags": ["embeddings,gaussian"], "family": "graph"}),
    ]


def test_infer_cli_model_info_uses_shared_discovery_renderer(monkeypatch) -> None:
    import pyimgano.infer_cli as infer_cli
    import pyimgano.services.discovery_service as discovery_service

    monkeypatch.setattr(
        discovery_service,
        "build_model_info_payload",
        lambda _name: {
            "name": "delegated-model",
            "tags": ["vision"],
            "metadata": {},
            "signature": "()",
            "accepts_var_kwargs": False,
            "accepted_kwargs": [],
        },
    )

    calls = []
    monkeypatch.setattr(
        infer_cli,
        "cli_discovery_rendering",
        type(
            "_StubDiscoveryRendering",
            (),
            {
                "emit_signature_payload": staticmethod(
                    lambda payload, *, json_output: calls.append((payload, json_output)) or 47
                ),
                "emit_model_preset_payload": staticmethod(lambda payload, *, json_output: 0),
            },
        ),
        raising=False,
    )

    rc = infer_cli.main(["--model-info", "vision_patchcore"])
    assert rc == 47
    assert calls == [
        (
            {
                "name": "delegated-model",
                "tags": ["vision"],
                "metadata": {},
                "signature": "()",
                "accepts_var_kwargs": False,
                "accepted_kwargs": [],
            },
            False,
        )
    ]


def test_infer_cli_model_preset_info_uses_shared_discovery_renderer(monkeypatch) -> None:
    import pyimgano.infer_cli as infer_cli
    import pyimgano.services.discovery_service as discovery_service

    monkeypatch.setattr(
        discovery_service,
        "build_model_preset_info_payload",
        lambda _name: {
            "name": "delegated-preset",
            "model": "vision_ecod",
            "kwargs": {"device": "cpu"},
            "optional": False,
            "tags": ["graph"],
            "description": "Delegated preset.",
        },
    )

    calls = []
    monkeypatch.setattr(
        infer_cli,
        "cli_discovery_rendering",
        type(
            "_StubDiscoveryRendering",
            (),
            {
                "emit_signature_payload": staticmethod(lambda payload, *, json_output: 0),
                "emit_model_preset_payload": staticmethod(
                    lambda payload, *, json_output: calls.append((payload, json_output)) or 53
                ),
            },
        ),
        raising=False,
    )

    rc = infer_cli.main(["--model-preset-info", "industrial-structural-ecod"])
    assert rc == 53
    assert calls == [
        (
            {
                "name": "delegated-preset",
                "model": "vision_ecod",
                "kwargs": {"device": "cpu"},
                "optional": False,
                "tags": ["graph"],
                "description": "Delegated preset.",
            },
            False,
        )
    ]


def test_infer_cli_accepts_model_preset_name_as_model(tmp_path: Path, monkeypatch) -> None:
    import pyimgano.infer_cli as infer_cli
    from pyimgano.infer_cli import main as infer_main

    input_dir = tmp_path / "inputs"
    _write_png(input_dir / "a.png")

    out_jsonl = tmp_path / "out.jsonl"

    created: dict[str, object] = {}

    def _create_model(name: str, **kwargs):  # noqa: ANN001, ANN201 - test stub
        created["name"] = str(name)
        created["kwargs"] = dict(kwargs)
        return _DummyDetector()

    monkeypatch.setattr(infer_cli, "create_model", _create_model)

    rc = infer_main(
        [
            "--model",
            "industrial-structural-ecod",
            "--input",
            str(input_dir),
            "--save-jsonl",
            str(out_jsonl),
        ]
    )
    assert rc == 0
    assert created.get("name") == "vision_feature_pipeline"


def test_infer_cli_list_models_uses_shared_discovery_option_helper(monkeypatch) -> None:
    import pyimgano.infer_cli as infer_cli
    import pyimgano.services.discovery_service as discovery_service

    helper_calls = []

    monkeypatch.setattr(
        infer_cli,
        "cli_discovery_options",
        type(
            "_StubDiscoveryOptions",
            (),
            {
                "validate_mutually_exclusive_flags": staticmethod(
                    lambda flags: helper_calls.append(("flags", list(flags)))
                ),
                "resolve_model_list_discovery_options": staticmethod(
                    lambda **kwargs: helper_calls.append(("resolve", dict(kwargs)))
                    or SimpleNamespace(
                        tags=["normalized-tag"],
                        family="normalized-family",
                        algorithm_type="normalized-type",
                        year="2031",
                    )
                ),
            },
        ),
        raising=False,
    )

    discovery_calls = []
    monkeypatch.setattr(
        discovery_service,
        "list_discovery_model_names",
        lambda **kwargs: discovery_calls.append(dict(kwargs)) or ["delegated-model"],
    )
    monkeypatch.setattr(
        infer_cli,
        "cli_listing",
        type(
            "_StubCliListing",
            (),
            {"emit_listing": staticmethod(lambda items, **kwargs: 83)},
        ),
        raising=False,
    )

    rc = infer_cli.main(["--list-models"])
    assert rc == 83
    assert len(helper_calls) == 2
    assert helper_calls[0][0] == "flags"
    assert helper_calls[1] == (
        "resolve",
        {
            "list_models": True,
            "tags": None,
            "family": None,
            "algorithm_type": None,
            "year": None,
            "allow_family_without_list_models": False,
        },
    )
    assert len(discovery_calls) == 1
    assert discovery_calls == [
        {
            "tags": ["normalized-tag"],
            "family": "normalized-family",
            "algorithm_type": "normalized-type",
            "year": "2031",
        }
    ]
