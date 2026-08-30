from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("torch")
pytest.importorskip("torchvision")


def _write_rgb(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8), mode="RGB").save(path)


def test_vision_cflow_contract_fit_and_score(tmp_path: Path) -> None:
    from pyimgano.models import create_model

    rng = np.random.default_rng(30)
    train_paths: list[str] = []
    test_paths: list[str] = []
    for idx in range(4):
        path = tmp_path / f"train_{idx}.png"
        _write_rgb(path, rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8))
        train_paths.append(str(path))
    for idx in range(2):
        path = tmp_path / f"test_{idx}.png"
        _write_rgb(path, rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8))
        test_paths.append(str(path))

    det = create_model(
        "vision_cflow",
        backbone="resnet18",
        pretrained_backbone=False,
        pool_layers=3,
        n_flows=1,
        condition_dim=8,
        soft_permutation=False,
        image_size=64,
        epochs=1,
        sub_epochs=1,
        batch_size=2,
        fiber_batch_size=8,
        num_workers=0,
        device="cpu",
        verbose=0,
    )

    det.fit(train_paths)
    assert det.feature_extractor.output_channels == (128, 256, 512)
    assert len(det.decoders) == 3
    assert all(len(decoder.flows) == 1 for decoder in det.decoders)
    scores = np.asarray(det.decision_function(test_paths), dtype=np.float64).reshape(-1)
    maps = np.asarray(det.predict_anomaly_map(test_paths), dtype=np.float32)
    assert scores.shape == (2,)
    assert np.all(np.isfinite(scores))
    assert maps.shape == (2, 64, 64)
    assert np.all(np.isfinite(maps))
    np.testing.assert_allclose(scores, maps.reshape(2, -1).max(axis=1), rtol=1e-6, atol=1e-6)


def test_cflow_author_defaults_and_decoder_parameter_count() -> None:
    from pyimgano.models.cflow import ConditionalFlow, VisionCFlow

    parameters = inspect.signature(VisionCFlow).parameters
    assert parameters["backbone"].default == "wide_resnet50_2"
    assert parameters["pool_layers"].default == 3
    assert parameters["n_flows"].default == 8
    assert parameters["condition_dim"].default == 128
    assert parameters["clamp_alpha"].default == 1.9
    assert parameters["soft_permutation"].default is True
    assert parameters["image_size"].default == 256
    assert parameters["epochs"].default == 25
    assert parameters["sub_epochs"].default == 8
    assert parameters["batch_size"].default == 32
    assert parameters["fiber_batch_size"].default == 256
    assert parameters["lr"].default == 2e-4

    decoders = [
        ConditionalFlow(channels, 128, 8, soft_permutation=False) for channels in (128, 256, 512)
    ]
    assert sum(parameter.numel() for decoder in decoders for parameter in decoder.parameters()) == (
        8_620_032
    )


def test_cflow_positional_encoding_matches_author_layout() -> None:
    torch = pytest.importorskip("torch")

    from pyimgano.models.cflow import positional_encoding_2d

    encoding = positional_encoding_2d(8, 3, 4)
    assert encoding.shape == (8, 3, 4)
    assert encoding[0, 0, 0].item() == pytest.approx(0.0)
    assert encoding[1, 0, 0].item() == pytest.approx(1.0)
    assert encoding[0, 0, 1].item() == pytest.approx(torch.sin(torch.tensor(1.0)).item())
    assert encoding[4, 1, 0].item() == pytest.approx(torch.sin(torch.tensor(1.0)).item())
    assert encoding[5, 0, 0].item() == pytest.approx(1.0)


def test_cflow_fitted_normalizer_is_query_batch_invariant() -> None:
    torch = pytest.importorskip("torch")

    from pyimgano.models.cflow import _fixed_normalized_anomaly_maps

    query = torch.full((1, 2, 2), -1.0)
    contextual_query = torch.cat((query, torch.full((1, 2, 2), -10.0)))
    kwargs = {
        "log_probability_maxima": (0.0,),
        "probability_sum_maximum": 1.0,
        "output_size": 2,
    }

    alone = _fixed_normalized_anomaly_maps([query], **kwargs)
    with_context = _fixed_normalized_anomaly_maps([contextual_query], **kwargs)

    torch.testing.assert_close(alone[0], with_context[0])
    torch.testing.assert_close(alone, torch.full_like(alone, 1.0 - np.exp(-1.0)))
