from __future__ import annotations

import pyimgano.models as models
from pyimgano.models import crossmad


def test_false_crossmad_proxies_are_not_registered() -> None:
    assert {"core_crossmad", "vision_crossmad"}.isdisjoint(models.list_models())
    assert crossmad.IMPLEMENTATION_STATUS == "unregistered-incompatible-cross-modal-contract"
    assert crossmad.AUTHOR_REPOSITORY_COMMIT == "c2b8fe7e060a642a247beaec24f1582c54d02cdf"
    assert crossmad.PAPER_LOCAL_FEATURE_LAYERS == (5, 15, 25)
    assert crossmad.PAPER_IMAGE_SIZE == 224
    assert crossmad.PAPER_SUPPORT_SHOTS == (1, 2, 4, 8)
    assert crossmad.PAPER_BATCH_SIZE == 16
    assert crossmad.PAPER_TRAINING_ITERATIONS == 1500
    assert crossmad.AUTHOR_CODE_GLOBAL_PROTOTYPE_SHAPE == (2, 1024)
    assert crossmad.AUTHOR_CODE_LOCAL_PROTOTYPE_SHAPE == (2, 1280)
    assert crossmad.AUTHOR_CODE_HARMONIZATION_TEMPERATURE == 0.04
    assert crossmad.AUTHOR_CODE_ZERO_SHOT_FUSION == {"global": 0.8, "local": 0.2}
    assert crossmad.AUTHOR_RELEASE_HAS_FEATURE_EXTRACTION_PIPELINE is False
    assert crossmad.AUTHOR_RELEASE_HAS_CHECKPOINTS is False
