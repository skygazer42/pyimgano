from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")


def _image(path: Path, value: int) -> Path:
    Image.fromarray(np.full((9, 11, 3), value, dtype=np.uint8)).save(path)
    return path


def test_univad_paper_contract_and_metadata() -> None:
    from pyimgano.models import MODEL_REGISTRY
    from pyimgano.models.univad import (
        OFFICIAL_COMMIT,
        PAPER_CLIP_BACKBONE,
        PAPER_CLIP_LAYERS,
        PAPER_DINOV2_BACKBONE,
        PAPER_IMAGE_SIZE,
        VisionUniVAD,
    )

    metadata = MODEL_REGISTRY.info("vision_univad").metadata
    assert metadata["paper_fidelity"] == "external-backend"
    assert metadata["supports_pixel_map"] is True
    assert metadata["official_repository"] == "https://github.com/FantasticGNU/UniVAD"
    assert OFFICIAL_COMMIT == "64d32873dda44fad69786834ea5ee1394ef81975"
    assert PAPER_IMAGE_SIZE == 448
    assert PAPER_CLIP_BACKBONE == "ViT-L-14-336"
    assert PAPER_DINOV2_BACKBONE == "dinov2_vitg14"
    assert PAPER_CLIP_LAYERS == (6, 12, 18, 24)

    detector = VisionUniVAD(device="cpu")
    assert detector.repository_path is None
    assert detector.allow_download is False


def test_author_backend_runs_released_setup_preprocess_and_inference(tmp_path: Path) -> None:
    from pyimgano.models.univad import AuthorUniVADBackend

    class _AuthorModel(torch.nn.Module):
        image_size = 448
        out_layers = [6, 12, 18, 24]

        def __init__(self) -> None:
            super().__init__()
            self.setup_call = None
            self.forward_calls = []

        def setup(self, payload, *, re_seg):  # noqa: ANN001, ANN201
            self.setup_call = (payload, re_seg)

        def forward(self, image, image_path):  # noqa: ANN001, ANN201
            self.forward_calls.append((image, image_path))
            value = float(image.mean())
            return {
                "pred_score": torch.tensor(value),
                "pred_mask": torch.full((1, 1, 448, 448), value),
            }

    paths = [_image(tmp_path / "a.png", 0), _image(tmp_path / "b.png", 255)]
    model = _AuthorModel()
    backend = AuthorUniVADBackend(
        repository_path=None,
        device="cpu",
        model=model,
    )
    backend.setup_support(paths, class_name="bottle", resegment_components=False)
    scores, maps = backend.score_paths(paths)

    payload, re_seg = model.setup_call
    assert payload["few_shot_samples"].shape == (2, 3, 448, 448)
    assert payload["few_shot_samples"][0].min().item() == 0.0
    assert payload["few_shot_samples"][1].max().item() == 1.0
    assert payload["dataset_category"] == "bottle"
    assert payload["image_path"] == [str(path) for path in paths]
    assert re_seg is False
    assert scores == pytest.approx([0.0, 1.0])
    assert maps.shape == (2, 448, 448)


def test_univad_wrapper_sets_support_calibrates_and_returns_maps(tmp_path: Path) -> None:
    from pyimgano.models.univad import VisionUniVAD

    class _Backend:
        def __init__(self) -> None:
            self.setup_call = None

        def setup_support(self, paths, **kwargs):  # noqa: ANN001, ANN201
            self.setup_call = (list(paths), kwargs)

        def score_paths(self, paths):  # noqa: ANN001, ANN201
            scores = np.asarray([float(Path(path).stem) for path in paths], dtype=np.float32)
            maps = np.stack([np.full((4, 4), score, dtype=np.float32) for score in scores])
            return scores, maps

    paths = [_image(tmp_path / f"{index}.png", index) for index in range(1, 5)]
    backend = _Backend()
    detector = VisionUniVAD(
        backend=backend,
        class_name="bottle",
        device="cpu",
        resegment_components=False,
        contamination=0.25,
    ).fit(paths)

    assert detector.decision_scores_ == pytest.approx([1.0, 2.0, 3.0, 4.0])
    assert backend.setup_call[0] == [path.resolve() for path in paths]
    assert backend.setup_call[1] == {
        "class_name": "bottle",
        "resegment_components": False,
    }
    assert detector.decision_function(paths[:2]) == pytest.approx([1.0, 2.0])
    assert detector.predict(paths).tolist() == [0, 0, 0, 1]
    assert detector.predict_anomaly_map(paths[:2]).shape == (2, 4, 4)

    with pytest.raises(TypeError, match="requires image paths"):
        detector.decision_function([np.zeros((8, 8, 3), dtype=np.uint8)])


def test_univad_rejects_missing_official_runtime_at_first_use(tmp_path: Path) -> None:
    from pyimgano.models.univad import VisionUniVAD

    image = _image(tmp_path / "normal.png", 0)
    detector = VisionUniVAD(class_name="bottle", device="cpu")
    with pytest.raises(ValueError, match="requires repository_path"):
        detector.fit([image])
