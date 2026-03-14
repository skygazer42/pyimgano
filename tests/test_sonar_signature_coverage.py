import pytest


def _make_instance_without_init(cls):
    # Many vision detectors pull optional heavy deps (e.g., torch) during __init__.
    # For these interface-level tests we bypass __init__ and only exercise the
    # early-guard branches added for SonarCloud signature compatibility.
    return cls.__new__(cls)


def test_predict_return_confidence_raises_without_optional_deps():
    from pyimgano.models.mambaad import VisionMambaAD
    from pyimgano.models.padim import VisionPaDiM
    from pyimgano.models.patchcore import VisionPatchCore

    for cls in (VisionMambaAD, VisionPaDiM, VisionPatchCore):
        inst = _make_instance_without_init(cls)
        with pytest.raises(NotImplementedError):
            cls.predict(inst, X=[], return_confidence=True)


def test_decision_function_rejects_non_positive_batch_size():
    from pyimgano.models.mambaad import VisionMambaAD
    from pyimgano.models.padim import VisionPaDiM
    from pyimgano.models.patchcore import VisionPatchCore

    for cls in (VisionMambaAD, VisionPaDiM, VisionPatchCore):
        inst = _make_instance_without_init(cls)
        with pytest.raises(ValueError):
            cls.decision_function(inst, X=[], batch_size=0)


def test_decision_function_accepts_positive_batch_size_on_empty_input():
    import numpy as np

    from pyimgano.models.mambaad import VisionMambaAD
    from pyimgano.models.padim import VisionPaDiM
    from pyimgano.models.patchcore import VisionPatchCore

    # VisionMambaAD: empty input should short-circuit before touching heavy deps.
    mamba = _make_instance_without_init(VisionMambaAD)
    scores = VisionMambaAD.decision_function(mamba, X=[], batch_size=1)
    assert isinstance(scores, np.ndarray)
    assert scores.shape == (0,)

    # VisionPaDiM: avoid heavy deps by bypassing __init__ and setting fitted markers.
    padim = _make_instance_without_init(VisionPaDiM)
    padim.means = np.zeros((1, 1), dtype=np.float32)
    padim.inv_covs = np.zeros((1, 1, 1), dtype=np.float32)
    padim.patch_shape = (1, 1)
    scores = VisionPaDiM.decision_function(padim, X=[], batch_size=1)
    assert isinstance(scores, np.ndarray)
    assert scores.shape == (0,)

    # VisionPatchCore: cover the post-guard path without fitting by triggering the
    # "not fitted" error after setting the minimal attributes it expects.
    patchcore = _make_instance_without_init(VisionPatchCore)
    patchcore._np = np
    patchcore.memory_bank = None
    patchcore.nn_index = None
    with pytest.raises(RuntimeError):
        VisionPatchCore.decision_function(patchcore, X=[], batch_size=1)
