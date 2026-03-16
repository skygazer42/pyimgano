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

    X = [object(), object(), object()]
    for cls in classes:
        inst = _make_instance_without_init(cls)
        inst.batch_size = 4

        def _predict_stub(_x, return_confidence: bool = False):  # noqa: ANN001
            assert return_confidence is False
            return _stub_scores_predict(len(_x))

        # Route decision_function -> predict without invoking heavy model code.
        inst.predict = _predict_stub  # type: ignore[assignment]

        scores = cls.decision_function(inst, X=X, batch_size=None)
        assert isinstance(scores, np.ndarray)
        assert scores.shape == (len(X),)
        assert int(inst.batch_size) == 4

        with pytest.raises(ValueError):
            cls.decision_function(inst, X=X, batch_size=0)

        scores2 = cls.decision_function(inst, X=X, batch_size=2)
        assert isinstance(scores2, np.ndarray)
        assert scores2.shape == (len(X),)
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
        inst.predict_proba = lambda X: _stub_scores_predict(len(X))  # type: ignore[assignment]

    def _setup_spade(inst):  # noqa: ANN001
        inst._check_fitted = lambda: None  # type: ignore[assignment]
        inst._iter_images = lambda X: []  # type: ignore[assignment]

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


def test_realnet_init_random_state_does_not_reset_numpy_global_rng():
    from pyimgano.models.realnet import VisionRealNet

    np.random.seed(2024)
    expected_first = np.random.rand()
    expected_second = np.random.rand()

    np.random.seed(2024)
    actual_first = np.random.rand()
    _ = VisionRealNet(random_state=123, epochs=1, device="cpu")
    actual_second = np.random.rand()

    assert actual_first == pytest.approx(expected_first)
    assert actual_second == pytest.approx(expected_second)


def test_ast_init_random_state_does_not_reset_numpy_global_rng():
    from pyimgano.models.ast import VisionAST

    np.random.seed(2025)
    expected_first = np.random.rand()
    expected_second = np.random.rand()

    np.random.seed(2025)
    actual_first = np.random.rand()
    _ = VisionAST(random_state=123, epochs=1, device="cpu")
    actual_second = np.random.rand()

    assert actual_first == pytest.approx(expected_first)
    assert actual_second == pytest.approx(expected_second)


def test_ast_random_state_makes_synthetic_anomalies_repeatable():
    import torch

    from pyimgano.models.ast import VisionAST

    images = torch.arange(2 * 3 * 16 * 16, dtype=torch.float32).reshape(2, 3, 16, 16)
    model_a = VisionAST(random_state=123, epochs=1, device="cpu")
    model_b = VisionAST(random_state=123, epochs=1, device="cpu")

    anomalous_a, masks_a = model_a._generate_synthetic_anomalies(images)
    anomalous_b, masks_b = model_b._generate_synthetic_anomalies(images)

    assert torch.equal(masks_a, masks_b)
    assert torch.allclose(anomalous_a, anomalous_b)


def test_realnet_anomaly_generator_random_state_is_repeatable():
    import torch

    from pyimgano.models.realnet import AnomalyGenerator

    image = torch.arange(3 * 16 * 16, dtype=torch.float32).reshape(3, 16, 16)
    generator_a = AnomalyGenerator(random_state=123)
    generator_b = AnomalyGenerator(random_state=123)

    anomalous_a, mask_a = generator_a.generate_anomaly(image, anomaly_type="intensity")
    anomalous_b, mask_b = generator_b.generate_anomaly(image, anomaly_type="intensity")

    assert torch.equal(mask_a, mask_b)
    assert torch.allclose(anomalous_a, anomalous_b)
