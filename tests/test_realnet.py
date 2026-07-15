from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")


class _TinyFeatureExtractor(torch.nn.Module):
    out_channels = (8, 8, 8, 8)

    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        super().__init__()
        self.projection = torch.nn.Conv2d(3, 8, 1)
        for parameter in self.parameters():
            parameter.requires_grad = False

    def forward(self, x):  # noqa: ANN001
        x = self.projection(x)
        return tuple(torch.nn.functional.avg_pool2d(x, scale) for scale in (4, 8, 16, 32))


def _inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20)
    normal = rng.integers(0, 255, size=(4, 32, 32, 3), dtype=np.uint8)
    anomaly = normal.copy()
    anomaly[:, 8:24, 8:24] = 255 - anomaly[:, 8:24, 8:24]
    masks = np.zeros((4, 32, 32), dtype=np.uint8)
    masks[:, 8:24, 8:24] = 1
    return normal, anomaly, masks


def _detector(monkeypatch) -> object:  # noqa: ANN001
    import pyimgano.models.realnet as realnet_module
    from pyimgano.models import create_model

    monkeypatch.setattr(realnet_module, "RealNetFeatureExtractor", _TinyFeatureExtractor)
    return create_model(
        "vision_realnet",
        backbone="resnet18",
        pretrained=False,
        selected_channels=(4, 4, 4, 4),
        hidden_ratio=1.0,
        channel_mult=(1,),
        attention_mult=(),
        num_res_blocks=1,
        attention_head_channels=4,
        rrs_mode_numbers=(2, 2),
        rrs_num_residual_layers=1,
        epochs=1,
        afs_batches=1,
        batch_size=2,
        image_size=32,
        image_score_pool_size=4,
        device="cpu",
        random_state=0,
    )


def test_vision_realnet_contract_fit_score_and_map(monkeypatch) -> None:  # noqa: ANN001
    normal, anomaly, masks = _inputs()
    detector = _detector(monkeypatch)

    detector.fit(normal, synthetic_images=anomaly, synthetic_masks=masks)
    scores = np.asarray(detector.decision_function(normal[:2]), dtype=np.float64)
    maps = np.asarray(detector.predict_anomaly_map(normal[:2]), dtype=np.float64)

    assert scores.shape == (2,)
    assert maps.shape == (2, 32, 32)
    assert np.isfinite(scores).all()
    assert np.isfinite(maps).all()


def test_vision_realnet_fit_does_not_print_progress(monkeypatch, capsys) -> None:  # noqa: ANN001
    normal, anomaly, masks = _inputs()
    detector = _detector(monkeypatch)

    detector.fit(normal, synthetic_images=anomaly, synthetic_masks=masks)

    assert capsys.readouterr().out == ""
