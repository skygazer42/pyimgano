from __future__ import annotations

import ast
from pathlib import Path


def test_metadata_contract_lists_expected_fields() -> None:
    from pyimgano.models.metadata_contract import metadata_contract_fields

    fields = metadata_contract_fields()
    names = [item["name"] for item in fields]
    assert names == [
        "paper_fidelity",
        "implementation_status",
        "paper",
        "related_paper",
        "year",
        "family",
        "type",
        "supervision",
        "supports_pixel_map",
        "requires_checkpoint",
        "weights_source",
    ]


def test_audit_metadata_contract_flags_missing_and_invalid_fields() -> None:
    from pyimgano.models.metadata_contract import audit_metadata_contract
    from pyimgano.models.registry import ModelRegistry

    registry = ModelRegistry()

    class _PixelModel:
        def get_anomaly_map(self, x):  # noqa: ANN001, ANN201 - test helper
            return x

    registry.register(
        "toy_pixel_model",
        _PixelModel,
        tags=("vision", "deep", "patchcore", "memory_bank"),
        metadata={
            "paper": "Toy Paper",
            "year": 3024,
            "paper_fidelity": "core-aligned",
            "implementation_status": "test-double",
            "supervision": "mystery-mode",
            "requires_checkpoint": True,
        },
    )

    payload = audit_metadata_contract(registry)
    assert payload["summary"]["total_models"] == 1
    assert payload["required_missing_by_model"]["toy_pixel_model"] == ["weights_source"]
    assert "toy_pixel_model" not in payload["recommended_missing_by_model"]
    invalid = payload["invalid_fields_by_model"]["toy_pixel_model"]
    assert any(item["field"] == "year" for item in invalid)
    assert any(item["field"] == "supervision" for item in invalid)


def test_deep_proxy_must_not_claim_the_related_paper_as_implemented() -> None:
    from pyimgano.models.metadata_contract import audit_metadata_contract
    from pyimgano.models.registry import ModelRegistry

    registry = ModelRegistry()
    registry.register(
        "toy_proxy",
        object,
        tags=("vision", "deep"),
        metadata={
            "paper": "A Real Paper",
            "year": 2024,
            "paper_fidelity": "inspired",
            "implementation_status": "experimental-proxy",
        },
    )

    payload = audit_metadata_contract(registry)
    assert payload["required_missing_by_model"]["toy_proxy"] == ["related_paper"]
    invalid = payload["invalid_fields_by_model"]["toy_proxy"]
    assert any(item["field"] == "paper" for item in invalid)


def test_audit_metadata_contract_accepts_default_weights_source_for_known_wrappers() -> None:
    import pyimgano.models  # noqa: F401 - registry population side effects
    from pyimgano.models.metadata_contract import audit_metadata_contract
    from pyimgano.models.registry import MODEL_REGISTRY

    payload = audit_metadata_contract(
        MODEL_REGISTRY,
        names=[
            "vision_patchcore_anomalib",
            "vision_patchcore_inspection_checkpoint",
            "vision_onnx_ecod",
            "vision_torchscript_ecod",
        ],
    )
    assert payload["required_missing_by_model"] == {}


def test_metadata_contract_resolves_supervision_from_explicit_tags() -> None:
    import pyimgano.models  # noqa: F401 - registry population side effects
    from pyimgano.models.metadata_contract import resolve_metadata_contract_payload
    from pyimgano.models.registry import MODEL_REGISTRY

    assert (
        resolve_metadata_contract_payload(MODEL_REGISTRY.info("cutpaste"))["supervision"]
        == "self-supervised"
    )
    assert (
        resolve_metadata_contract_payload(MODEL_REGISTRY.info("vision_devnet"))["supervision"]
        == "weakly-supervised"
    )
    assert (
        resolve_metadata_contract_payload(MODEL_REGISTRY.info("core_ocsvm"))["supervision"]
        == "one-class"
    )


def test_every_registered_deep_model_declares_an_auditable_paper_relationship() -> None:
    import pyimgano.models  # noqa: F401 - registry population side effects
    from pyimgano.models.registry import MODEL_REGISTRY

    valid = {
        "core-aligned",
        "paper-adaptation",
        "partial",
        "inspired",
        "external-backend",
        "not-applicable",
    }
    for name in MODEL_REGISTRY.available(tags=("deep",)):
        entry = MODEL_REGISTRY.info(name)
        metadata = entry.metadata
        fidelity = metadata.get("paper_fidelity")
        assert fidelity in valid, name
        assert str(metadata.get("implementation_status", "")).strip(), name
        assert "sota" not in entry.tags, name

        if fidelity in {"core-aligned", "paper-adaptation"}:
            assert str(metadata.get("paper", "")).strip(), name
        elif fidelity in {"partial", "inspired"}:
            assert "paper" not in metadata, name
            assert str(metadata.get("related_paper", "")).strip(), name


def test_paper_fidelity_classification_covers_core_proxy_and_backend_paths() -> None:
    import pyimgano.models  # noqa: F401 - registry population side effects
    from pyimgano.models.registry import MODEL_REGISTRY

    expected = {
        "vision_patchcore": "core-aligned",
        "vision_padim": "core-aligned",
        "vision_stfpm": "core-aligned",
        "vision_reverse_distillation": "core-aligned",
        "vision_draem": "paper-adaptation",
        "vision_simplenet": "core-aligned",
        "vision_differnet": "paper-adaptation",
        "vision_ast": "paper-adaptation",
        "vision_promptad": "paper-adaptation",
        "vision_adaclip": "paper-adaptation",
        "vision_aaclip": "paper-adaptation",
        "vision_realnet": "paper-adaptation",
        "vision_regad": "paper-adaptation",
        "vision_winclip": "paper-adaptation",
        "vision_anomalydino": "paper-adaptation",
        "vision_patchcore_anomalib": "external-backend",
        "vision_fcdd": "paper-adaptation",
        "vision_memae": "paper-adaptation",
        "vision_riad": "paper-adaptation",
    }
    for name, fidelity in expected.items():
        assert MODEL_REGISTRY.info(name).metadata["paper_fidelity"] == fidelity


def test_all_deep_models_resolve_a_valid_supervision_contract() -> None:
    import pyimgano.models  # noqa: F401 - registry population side effects
    from pyimgano.models.metadata_contract import audit_metadata_contract
    from pyimgano.models.registry import MODEL_REGISTRY

    names = MODEL_REGISTRY.available(tags=("deep",))
    report = audit_metadata_contract(MODEL_REGISTRY, names=names)

    assert report["recommended_missing_by_model"] == {}
    assert report["invalid_fields_by_model"] == {}


def test_corrected_proxy_references_do_not_repeat_legacy_citation_errors() -> None:
    import pyimgano.models  # noqa: F401 - registry population side effects
    from pyimgano.models.registry import MODEL_REGISTRY

    anogen = MODEL_REGISTRY.info("vision_anogen_adapter").metadata
    riad = MODEL_REGISTRY.info("vision_riad").metadata

    assert anogen["year"] == 2024
    assert "Few-Shot Anomaly-Driven Generation" in anogen["related_paper"]
    assert "eccv_2024" in anogen["related_paper_url"]
    assert riad["paper_url"] == "https://doi.org/10.1016/j.patcog.2020.107706"
    assert riad["year"] == 2021
    assert "2108.11092" not in str(riad)


def test_unregistered_legacy_neural_modules_disclaim_unverified_paper_status() -> None:
    from pyimgano.models import bgad, csflow, dsr, intra, one_to_normal, pni, rdplusplus

    assert bgad.PAPER_FIDELITY == "not-applicable"
    assert csflow.PAPER_FIDELITY == "partial"
    assert dsr.PAPER_FIDELITY == "not-applicable"
    assert intra.PAPER_FIDELITY == "not-applicable"
    assert one_to_normal.IMPLEMENTATION_STATUS == "unregistered-incomplete-author-release"
    assert pni.PAPER_FIDELITY == "not-applicable"
    assert rdplusplus.PAPER_FIDELITY == "inspired"


def test_every_direct_deep_detector_is_registered_or_explicitly_disclaimed() -> None:
    models_dir = Path(__file__).parents[1] / "pyimgano" / "models"
    unregistered: set[str] = set()
    for path in models_dir.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            base_names = {ast.unparse(base).split(".")[-1] for base in node.bases}
            if "BaseVisionDeepDetector" not in base_names:
                continue
            is_registered = any(
                isinstance(decorator, ast.Call)
                and ast.unparse(decorator.func).split(".")[-1] == "register_model"
                for decorator in node.decorator_list
            )
            if not is_registered:
                unregistered.add(f"{path.stem}.{node.name}")

    assert unregistered == {
        "bgad.BGADDetector",
        "csflow.CSFlowDetector",
        "dsr.DSRDetector",
        "intra.InTraDetector",
        "pni.PNIDetector",
        "rdplusplus.RDPlusPlusDetector",
    }
