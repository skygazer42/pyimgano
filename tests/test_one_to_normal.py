from __future__ import annotations

import pyimgano.models as models
from pyimgano.models import one_to_normal


def test_incomplete_one_to_normal_release_is_not_registered_as_the_paper() -> None:
    assert "vision_one_to_normal" not in models.list_models()
    assert one_to_normal.PAPER_FIDELITY == "not-applicable"
    assert one_to_normal.IMPLEMENTATION_STATUS == "unregistered-incomplete-author-release"
    assert one_to_normal.AUTHOR_ARTIFACT_COMMIT == ("1faca331bf876a66f105a8f5aa095e399c21e44d")
    assert one_to_normal.PAPER_DIFFUSION_BACKBONE == "Stable Diffusion v1.5 + DreamBooth"
    assert one_to_normal.PAPER_CLIP_BACKBONE == "ViT-L/14"
    assert one_to_normal.PAPER_IMAGE_SIZE == 240
    assert one_to_normal.PAPER_SHOTS == (2, 4, 8)
    assert one_to_normal.PAPER_TIMESTEP_RATIO == 0.3
    assert one_to_normal.PAPER_MEMORY_SIZE == 30
    assert (one_to_normal.PAPER_ALPHA, one_to_normal.PAPER_BETA) == (1.0, 0.5)
