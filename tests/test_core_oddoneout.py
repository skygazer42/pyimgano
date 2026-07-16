from __future__ import annotations

import pyimgano.models as models
from pyimgano.models import core_oddoneout


def test_false_oddoneout_vector_proxies_are_not_registered() -> None:
    removed = {
        "core_oddoneout",
        "vision_oddoneout",
        "vision_onnx_oddoneout",
        "vision_resnet18_oddoneout",
        "vision_torchscript_oddoneout",
    }
    assert removed.isdisjoint(models.list_models())
    assert core_oddoneout.IMPLEMENTATION_STATUS == "unregistered-incompatible-scene-contract"
    assert core_oddoneout.AUTHOR_REPOSITORY_COMMIT == ("5200c918e80628288c4bdc46c5afd036d1e79482")
    assert core_oddoneout.PAPER_NUM_VIEWS == 5
    assert core_oddoneout.PAPER_IMAGE_SIZE == 256
    assert core_oddoneout.PAPER_VOXEL_GRID == (96, 96, 16)
    assert core_oddoneout.PAPER_DENSITY_THRESHOLD == 0.2
    assert core_oddoneout.PAPER_ATTENTION_HEADS == 8
    assert core_oddoneout.PAPER_ATTENTION_TOPK == 20
    assert core_oddoneout.PAPER_STAGE_EPOCHS == (50, 50)
    assert core_oddoneout.AUTHOR_CODE_DINO_MODEL == "dinov2_vits14"
    assert core_oddoneout.AUTHOR_CODE_VOXEL_DIMS == {
        "feature": 32,
        "hidden": 128,
        "projection": 384,
    }
