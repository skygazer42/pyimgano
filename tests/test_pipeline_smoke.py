import pytest

from pyimgano.pipelines.mvtec_visa import build_default_detector

pytest.importorskip("torch")
pytest.importorskip("torchvision")


def test_build_default_detector():
    det = build_default_detector(model="vision_patchcore", device="cpu", pretrained=False)
    assert det is not None
