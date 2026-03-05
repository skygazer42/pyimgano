def test_basevisiondeepdetector_inherits_native_deep_base() -> None:
    from pyimgano.models.base_deep import BaseDeepLearningDetector
    from pyimgano.models.baseCv import BaseVisionDeepDetector

    assert issubclass(BaseVisionDeepDetector, BaseDeepLearningDetector)
