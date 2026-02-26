
def test_basevisiondeepdetector_inherits_native_deep_base() -> None:
    from pyimgano.models.baseCv import BaseVisionDeepDetector
    from pyimgano.models.base_deep import BaseDeepLearningDetector

    assert issubclass(BaseVisionDeepDetector, BaseDeepLearningDetector)
