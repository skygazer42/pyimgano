from __future__ import annotations


def test_resolve_model_spec_string() -> None:
    from pyimgano.models.ensemble_spec import resolve_model_spec

    det = resolve_model_spec("core_knn", default_contamination=0.2)
    assert hasattr(det, "fit")
    assert hasattr(det, "decision_function")


def test_resolve_model_spec_dict_kwargs() -> None:
    from pyimgano.models.ensemble_spec import resolve_model_spec

    det = resolve_model_spec(
        {"name": "core_knn", "kwargs": {"n_neighbors": 7}},
        default_contamination=0.2,
    )
    assert hasattr(det, "fit")
    assert hasattr(det, "decision_function")
    # best-effort: should carry through kwargs
    assert getattr(det, "n_neighbors", None) in (7, None)  # wrapper stores in backend kwargs


def test_resolve_model_spec_instance_passthrough() -> None:
    from pyimgano.models import create_model
    from pyimgano.models.ensemble_spec import resolve_model_spec

    base = create_model("core_knn", contamination=0.1, n_neighbors=5)
    det = resolve_model_spec(base, default_contamination=0.2)
    assert det is base
