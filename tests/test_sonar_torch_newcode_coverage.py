import inspect

import numpy as np
import pytest

# These tests are only meaningful (and importable) when torch is installed.
# The SonarCloud workflow installs `pyimgano[torch]` to cover new-code guards
# added in torch-based detectors without forcing torch on all CI workflows.
pytest.importorskip("torch")


def _make_instance_without_init(cls):
    # Many detectors pull heavy deps / download weights in __init__.
    # For interface-level tests we bypass __init__ entirely.
    return cls.__new__(cls)


def _stub_scores_predict(n: int):
    return np.zeros((int(n),), dtype=np.float64)


def _assert_x_signature(method, expected_name: str = "x") -> None:
    params = list(inspect.signature(method).parameters)
    assert expected_name in params
    assert "X" not in params


def test_torch_detectors_predict_return_confidence_raises():
    from pyimgano.models.ast import VisionAST
    from pyimgano.models.bayesianpf import VisionBayesianPF
    from pyimgano.models.cflow import VisionCFlow
    from pyimgano.models.dfm import VisionDFM
    from pyimgano.models.draem import VisionDRAEM
    from pyimgano.models.dst import VisionDST
    from pyimgano.models.favae import VisionFAVAE
    from pyimgano.models.gcad import VisionGCAD
    from pyimgano.models.glad import VisionGLAD
    from pyimgano.models.inctrl import VisionInCTRL
    from pyimgano.models.oneformore import VisionOneForMore
    from pyimgano.models.panda import VisionPANDA
    from pyimgano.models.promptad import VisionPromptAD
    from pyimgano.models.realnet import VisionRealNet
    from pyimgano.models.regad import VisionRegAD
    from pyimgano.models.simplenet import VisionSimpleNet
    from pyimgano.models.stfpm import VisionSTFPM

    classes = (
        VisionAST,
        VisionBayesianPF,
        VisionCFlow,
        VisionDFM,
        VisionDRAEM,
        VisionDST,
        VisionFAVAE,
        VisionGCAD,
        VisionGLAD,
        VisionInCTRL,
        VisionOneForMore,
        VisionPANDA,
        VisionPromptAD,
        VisionRealNet,
        VisionRegAD,
        VisionSimpleNet,
        VisionSTFPM,
    )

    for cls in classes:
        inst = _make_instance_without_init(cls)
        with pytest.raises(NotImplementedError):
            cls.predict(inst, X=[], return_confidence=True)


def test_torch_detectors_use_lowercase_x_signatures():
    from pyimgano.models.anomalib_backend import VisionAnomalibCheckpoint
    from pyimgano.models.bayesianpf import VisionBayesianPF
    from pyimgano.models.cflow import VisionCFlow
    from pyimgano.models.dfm import VisionDFM
    from pyimgano.models.realnet import VisionRealNet
    from pyimgano.models.regad import VisionRegAD
    from pyimgano.models.simplenet import VisionSimpleNet
    from pyimgano.models.spade import VisionSPADEDetector

    cases = (
        (
            VisionBayesianPF,
            {
                "_preprocess": "x",
                "fit": "x",
                "predict": "x",
                "decision_function": "x",
                "predict_with_uncertainty": "x",
            },
        ),
        (
            VisionCFlow,
            {
                "fit": "x",
                "predict": "x",
                "decision_function": "x",
                "predict_anomaly_map": "x",
            },
        ),
        (VisionDFM, {"fit": "x", "predict": "x", "decision_function": "x"}),
        (
            VisionRealNet,
            {"_preprocess": "x", "fit": "x", "predict": "x", "decision_function": "x"},
        ),
        (
            VisionRegAD,
            {
                "_preprocess": "x",
                "fit": "x",
                "predict": "x",
                "decision_function": "x",
                "get_registration_map": "x",
            },
        ),
        (
            VisionSimpleNet,
            {
                "fit": "x",
                "_build_reference_features": "image_paths",
                "predict": "x",
                "decision_function": "x",
            },
        ),
        (
            VisionSPADEDetector,
            {
                "_iter_images": "x",
                "fit": "x",
                "decision_function": "x",
                "predict_proba": "x",
                "predict_anomaly_map": "x",
            },
        ),
        (
            VisionAnomalibCheckpoint,
            {"fit": "x", "decision_function": "x", "predict": "x", "predict_anomaly_map": "x"},
        ),
    )

    for cls, method_expectations in cases:
        for method_name, expected_name in method_expectations.items():
            _assert_x_signature(getattr(cls, method_name), expected_name=expected_name)


def test_draem_public_methods_accept_legacy_X_keyword():
    from pyimgano.models.draem import VisionDRAEM

    inst = _make_instance_without_init(VisionDRAEM)
    inst.get_anomaly_map = lambda _path: np.zeros((1, 1), dtype=np.float32)  # type: ignore[assignment]

    with pytest.raises(ValueError, match="Training set cannot be empty"):
        VisionDRAEM.fit(inst, X=[], y=None)

    with pytest.raises(NotImplementedError):
        VisionDRAEM.predict(inst, X=[], return_confidence=True)

    with pytest.raises(ValueError):
        VisionDRAEM.decision_function(inst, X=[], batch_size=0)

    maps = VisionDRAEM.predict_anomaly_map(inst, X=[object(), object()])
    assert isinstance(maps, np.ndarray)
    assert maps.shape == (2, 1, 1)


def test_stfpm_public_methods_accept_legacy_X_keyword():
    from pyimgano.models.stfpm import VisionSTFPM

    inst = _make_instance_without_init(VisionSTFPM)
    inst.get_anomaly_map = lambda _path: np.zeros((1, 1), dtype=np.float32)  # type: ignore[assignment]

    with pytest.raises(ValueError, match="Training set cannot be empty"):
        VisionSTFPM.fit(inst, X=[], y=None)

    with pytest.raises(NotImplementedError):
        VisionSTFPM.predict(inst, X=[], return_confidence=True)

    with pytest.raises(ValueError):
        VisionSTFPM.decision_function(inst, X=[], batch_size=0)

    maps = VisionSTFPM.predict_anomaly_map(inst, X=[object(), object()])
    assert isinstance(maps, np.ndarray)
    assert maps.shape == (2, 1, 1)


def test_torch_detectors_decision_function_alias_batch_size_paths():
    # These detectors implement decision_function as an alias to predict() and
    # accept an optional batch_size for interface compatibility.
    from pyimgano.models.ast import VisionAST
    from pyimgano.models.bayesianpf import VisionBayesianPF
    from pyimgano.models.dst import VisionDST
    from pyimgano.models.favae import VisionFAVAE
    from pyimgano.models.gcad import VisionGCAD
    from pyimgano.models.glad import VisionGLAD
    from pyimgano.models.inctrl import VisionInCTRL
    from pyimgano.models.oneformore import VisionOneForMore
    from pyimgano.models.panda import VisionPANDA
    from pyimgano.models.promptad import VisionPromptAD
    from pyimgano.models.realnet import VisionRealNet
    from pyimgano.models.regad import VisionRegAD

    classes = (
        VisionAST,
        VisionBayesianPF,
        VisionDST,
        VisionFAVAE,
        VisionGCAD,
        VisionGLAD,
        VisionInCTRL,
        VisionOneForMore,
        VisionPANDA,
        VisionPromptAD,
        VisionRealNet,
        VisionRegAD,
    )

    x = [object(), object(), object()]
    for cls in classes:
        inst = _make_instance_without_init(cls)
        inst.batch_size = 4

        def _predict_stub(_x, return_confidence: bool = False):  # noqa: ANN001
            assert return_confidence is False
            return _stub_scores_predict(len(_x))

        # Route decision_function -> predict without invoking heavy model code.
        inst.predict = _predict_stub  # type: ignore[assignment]

        scores = cls.decision_function(inst, X=x, batch_size=None)
        assert isinstance(scores, np.ndarray)
        assert scores.shape == (len(x),)
        assert int(inst.batch_size) == 4

        with pytest.raises(ValueError):
            cls.decision_function(inst, X=x, batch_size=0)

        scores2 = cls.decision_function(inst, X=x, batch_size=2)
        assert isinstance(scores2, np.ndarray)
        assert scores2.shape == (len(x),)
        assert int(inst.batch_size) == 4


def test_torch_detectors_batch_size_validation_only_paths():
    # Some detectors keep `batch_size` for interface compatibility but score
    # one item at a time. We can cover the validation branch without fitting.
    from pyimgano.models.cflow import VisionCFlow
    from pyimgano.models.dfm import VisionDFM
    from pyimgano.models.differnet import DifferNetDetector
    from pyimgano.models.draem import VisionDRAEM
    from pyimgano.models.simplenet import VisionSimpleNet
    from pyimgano.models.spade import VisionSPADEDetector
    from pyimgano.models.stfpm import VisionSTFPM

    # (cls, setup_fn): setup can stub internal methods to avoid heavy deps.
    cases = []

    def _no_setup(_inst):  # noqa: ANN001
        return None

    cases.extend(
        [
            (VisionCFlow, _no_setup),
            (VisionDFM, _no_setup),
            (VisionDRAEM, _no_setup),
            (VisionSimpleNet, _no_setup),
            (VisionSTFPM, _no_setup),
        ]
    )

    def _setup_differnet(inst):  # noqa: ANN001
        inst.predict_proba = lambda x: _stub_scores_predict(len(x))  # type: ignore[assignment]

    def _setup_spade(inst):  # noqa: ANN001
        inst._check_fitted = lambda: None  # type: ignore[assignment]
        inst._iter_images = lambda x: []  # type: ignore[assignment]

    cases.extend([(DifferNetDetector, _setup_differnet), (VisionSPADEDetector, _setup_spade)])

    for cls, setup in cases:
        inst = _make_instance_without_init(cls)
        setup(inst)

        with pytest.raises(ValueError):
            cls.decision_function(inst, X=[], batch_size=0)

        # Cover the "batch_size is None" / return path for cases where we
        # explicitly stubbed the downstream calls to avoid requiring a fit.
        if cls in (DifferNetDetector, VisionSPADEDetector):
            scores = cls.decision_function(inst, X=[], batch_size=None)
            assert isinstance(scores, np.ndarray)
