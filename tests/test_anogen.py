from __future__ import annotations

import pyimgano.models as models
from pyimgano.models import anogen


def test_anogen_workflow_is_not_registered_as_a_detector() -> None:
    assert "vision_anogen_adapter" not in models.list_models()
    assert anogen.IMPLEMENTATION_STATUS == "unregistered-workflow-not-detector"
    assert anogen.AUTHOR_REPOSITORY_COMMIT == "11ade1bd89ec3bb89646d70b6b95f2c69053f973"
    assert (anogen.PAPER_TEXT_ENCODER, anogen.PAPER_EMBEDDING_DIM) == ("CLIP", 768)
    assert (anogen.PAPER_SUPPORT_SHOTS, anogen.PAPER_EMBEDDING_TRAIN_STEPS) == (3, 6000)
    assert anogen.PAPER_EMBEDDING_LEARNING_RATE == 0.005
    assert (anogen.PAPER_MASKS_PER_NORMAL_IMAGE, anogen.PAPER_IMAGES_PER_MASK) == (2, 2)
    assert anogen.PAPER_NORMAL_CONFIDENCE_THRESHOLD == 0.9
    assert anogen.PAPER_GENERATED_SAMPLE_PROBABILITY == 0.5
    assert anogen.PAPER_GENERATED_DATASET_SIZE == 70_760
    assert (anogen.AUTHOR_CODE_TEXT_ENCODER, anogen.AUTHOR_CODE_CONTEXT_DIM) == (
        "BERTEmbedder",
        1280,
    )
    assert anogen.AUTHOR_CODE_IMAGE_SIZE == 256
    assert anogen.AUTHOR_CODE_MAX_STEPS == 6100
    assert anogen.AUTHOR_RELEASE_HAS_ANOMALY_EMBEDDINGS is True
    assert anogen.AUTHOR_RELEASE_HAS_DETECTOR_CHECKPOINT is False
