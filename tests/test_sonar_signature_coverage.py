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

