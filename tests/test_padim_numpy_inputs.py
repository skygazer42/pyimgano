import numpy as np
import pytest

import pyimgano.models.padim as padim_module
from pyimgano.models import create_model

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")


def test_padim_accepts_numpy_images_for_fit_scoring_and_maps():
    det = create_model(
        "vision_padim",
        pretrained=False,
        device="cpu",
        image_size=32,
        d_reduced=4,
        covariance_eps=0.1,
    )

    assert det.resize_size == 37
    assert [type(item).__name__ for item in det.transform.transforms] == [
        "ToPILImage",
        "Resize",
        "CenterCrop",
        "ToTensor",
        "Normalize",
    ]
    assert det.transform.transforms[1].interpolation.name == "BICUBIC"

    imgs = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(2)]
    det.fit(imgs)

    scores = det.decision_function(imgs)
    assert scores.shape == (2,)

    anomaly_map = det.get_anomaly_map(imgs[0])
    assert anomaly_map.shape == (32, 32)
    assert anomaly_map.dtype == np.float32
    assert np.isfinite(anomaly_map).all()


def test_padim_uses_fixed_random_channel_subset_not_projection(monkeypatch):
    det = create_model(
        "vision_padim",
        pretrained=False,
        device="cpu",
        image_size=32,
        d_reduced=2,
        covariance_eps=0.1,
        random_state=7,
    )
    features = np.arange(24, dtype=np.float32).reshape(6, 4)

    def fake_extract(_image):
        det.patch_shape = (2, 3)
        return features.copy()

    monkeypatch.setattr(det, "_extract_patch_features", fake_extract)

    images = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(2)]
    det.fit(images)

    expected_indices = np.sort(np.random.default_rng(7).choice(4, size=2, replace=False))
    np.testing.assert_array_equal(det.feature_indices_, expected_indices)
    np.testing.assert_allclose(det.means, features[:, expected_indices])


def test_padim_paper_backbone_defaults_and_blockwise_feature_correspondence(monkeypatch):
    class TinyBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = torch.nn.Identity()
            self.layer2 = torch.nn.AvgPool2d(2)
            self.layer3 = torch.nn.AvgPool2d(2)

        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x)
            return self.layer3(x)

    built = []

    def build_backbone(name, *, pretrained):
        built.append((name, pretrained))
        return TinyBackbone()

    monkeypatch.setattr(padim_module, "_build_torchvision_backbone", build_backbone)

    r18 = padim_module.VisionPaDiM(image_size=32)
    wr50 = padim_module.VisionPaDiM(backbone="wide_resnet50", image_size=32)
    assert r18.d_reduced == 100
    assert wr50.backbone_name == "wide_resnet50_2"
    assert wr50.d_reduced == 550
    assert built == [("resnet18", False), ("wide_resnet50_2", False)]

    image = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    wr50.transform = lambda value: torch.from_numpy(value.copy()).permute(2, 0, 1).float()
    actual = wr50._extract_patch_features(image)

    level1 = wr50.transform(image).unsqueeze(0)
    level2 = torch.nn.functional.avg_pool2d(level1, 2)
    level3 = torch.nn.functional.avg_pool2d(level2, 2)
    expected = torch.cat(
        [
            level1,
            level2.repeat_interleave(2, -2).repeat_interleave(2, -1),
            level3.repeat_interleave(4, -2).repeat_interleave(4, -1),
        ],
        dim=1,
    )
    expected = expected.permute(0, 2, 3, 1).reshape(16, 9).numpy()
    np.testing.assert_array_equal(actual, expected)
