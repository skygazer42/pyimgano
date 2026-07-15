from __future__ import annotations

import inspect

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")


def _make_rgb_batch(*, count: int = 4, size: int = 32) -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    out: list[np.ndarray] = []
    for _ in range(count):
        out.append(rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8))
    return out


def test_cutpaste_contract_accepts_numpy_image_list() -> None:
    from pyimgano.models import create_model

    train = _make_rgb_batch(count=4)
    test = _make_rgb_batch(count=2)

    det = create_model(
        "vision_cutpaste",
        contamination=0.25,
        epochs=1,
        batch_size=3,
        steps_per_epoch=1,
        device="cpu",
        pretrained=False,
    )

    det.fit(train)
    assert det.training_steps_ == 1
    scores = np.asarray(det.decision_function(test), dtype=np.float64).reshape(-1)
    assert scores.shape == (2,)
    assert np.all(np.isfinite(scores))


def test_cutpaste_fit_does_not_print_progress(capsys) -> None:
    from pyimgano.models import create_model

    train = _make_rgb_batch(count=4)

    det = create_model(
        "vision_cutpaste",
        contamination=0.25,
        epochs=10,
        batch_size=3,
        steps_per_epoch=1,
        device="cpu",
        pretrained=False,
    )

    det.fit(train)
    out = capsys.readouterr().out
    assert out == ""


def test_cutpaste_three_way_dataset_emits_all_classes() -> None:
    import torch

    from pyimgano.models.cutpaste import CutPasteDataset

    image = np.full((32, 32, 3), 127, dtype=np.uint8)
    dataset = CutPasteDataset(
        np.asarray([image]),
        transform=lambda item: torch.from_numpy(item.copy()).permute(2, 0, 1),
        augment_type="3way",
    )

    images, labels = dataset[0]
    assert images.shape == (3, 3, 32, 32)
    assert labels.tolist() == [0, 1, 2]


def test_cutpaste_paper_defaults_and_preprocessing() -> None:
    from pyimgano.models.cutpaste import CutPasteDetector

    parameters = inspect.signature(CutPasteDetector).parameters
    assert parameters["augment_type"].default == "3way"
    assert parameters["epochs"].default == 256
    assert parameters["batch_size"].default == 96
    assert parameters["steps_per_epoch"].default == 256
    assert parameters["image_size"].default == 256

    det = CutPasteDetector(epochs=1, steps_per_epoch=1, device="cpu")

    assert det.augment_type == "3way"
    assert det.image_size == 256
    assert det.batch_size == 96
    names = [type(item).__name__ for item in det._get_transform(training=False).transforms]
    assert names == ["ToPILImage", "Resize", "ToTensor", "Normalize"]
    output = det._get_transform(training=False)(
        np.zeros((40, 60, 3), dtype=np.uint8)
    )
    assert tuple(output.shape) == (3, 256, 256)


def test_cutpaste_augmentation_is_repeatable_with_local_seed() -> None:
    from pyimgano.models.cutpaste import CutPasteAugmentation

    image = np.arange(32 * 32 * 3, dtype=np.uint16).reshape(32, 32, 3)
    image = (image % 256).astype(np.uint8)
    first = CutPasteAugmentation(type="scar", rng=np.random.default_rng(7))(image)
    second = CutPasteAugmentation(type="scar", rng=np.random.default_rng(7))(image)

    np.testing.assert_array_equal(first, second)
    assert first.shape == image.shape
    assert first.dtype == image.dtype
