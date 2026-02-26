
def test_basevisiondetector_inherits_native_base_detector() -> None:
    from pyimgano.models.base_detector import BaseDetector
    from pyimgano.models.baseml import BaseVisionDetector

    assert issubclass(BaseVisionDetector, BaseDetector)
