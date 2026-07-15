from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")


def _image(path: Path, value: int) -> Path:
    Image.fromarray(np.full((9, 11, 3), value, dtype=np.uint8)).save(path)
    return path


def test_logsad_paper_contract_and_metadata() -> None:
    from pyimgano.models import MODEL_REGISTRY
    from pyimgano.models.logsad import (
        OFFICIAL_COMMIT,
        PAPER_CLIP_BACKBONE,
        PAPER_DINOV2_BACKBONE,
        PAPER_FEATURE_LAYERS,
        PAPER_FEATURE_SIZE,
        PAPER_IMAGE_SIZE,
        PAPER_SAM_BACKBONE,
        VisionLogSAD,
    )

    metadata = MODEL_REGISTRY.info("vision_logsad").metadata
    assert metadata["paper_fidelity"] == "external-backend"
    assert metadata["supports_pixel_map"] is True
    assert metadata["official_repository"] == "https://github.com/zhang0jhon/LogSAD"
    assert OFFICIAL_COMMIT == "06aed1a8d4181ce08ffa91f9e5f8733c27833b55"
    assert PAPER_IMAGE_SIZE == 448
    assert PAPER_FEATURE_SIZE == 64
    assert PAPER_CLIP_BACKBONE == "hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K"
    assert PAPER_DINOV2_BACKBONE == "dinov2_vitl14"
    assert PAPER_SAM_BACKBONE == "vit_h"
    assert PAPER_FEATURE_LAYERS == (6, 12, 18, 24)

    detector = VisionLogSAD(device="cpu")
    assert detector.repository_path is None
    assert detector.allow_download is False


def test_author_backend_runs_released_setup_preprocess_and_inference(tmp_path: Path) -> None:
    from pyimgano.models.logsad import AuthorLogSADBackend

    class _AuthorModel(torch.nn.Module):
        feature_list = [6, 12, 18, 24]
        feature_list_dinov2 = [6, 12, 18, 24]
        feat_size = 64
        ori_feat_size = 32
        memory_size = 2048
        n_neighbors = 2

        def __init__(self) -> None:
            super().__init__()
            self.setup_call = None
            self.forward_calls = []

        def setup(self, payload):  # noqa: ANN001, ANN201
            self.setup_call = payload

        def forward(self, image, image_paths):  # noqa: ANN001, ANN201
            self.forward_calls.append((image, image_paths))
            value = float(image.mean())
            return {
                "pred_score": torch.tensor(value),
                "anomaly_map": torch.full((64, 64), value),
            }

    paths = [_image(tmp_path / "a.png", 0), _image(tmp_path / "b.png", 255)]
    model = _AuthorModel()
    backend = AuthorLogSADBackend(repository_path=None, device="cpu", model=model)
    backend.setup_support(paths, class_name="pushpins")
    scores, maps = backend.score_paths(paths)

    payload = model.setup_call
    assert payload["few_shot_samples"].shape == (2, 3, 448, 448)
    assert payload["few_shot_samples"][0].min().item() == 0.0
    assert payload["few_shot_samples"][1].max().item() == 1.0
    assert payload["dataset_category"] == "pushpins"
    assert payload["few_shot_samples_path"] == [str(path) for path in paths]
    assert scores == pytest.approx([0.0, 1.0])
    assert maps.shape == (2, 64, 64)
    assert model.forward_calls[0][1] == [str(paths[0])]


def test_logsad_wrapper_sets_support_calibrates_and_returns_maps(tmp_path: Path) -> None:
    from pyimgano.models.logsad import VisionLogSAD

    class _Backend:
        def __init__(self) -> None:
            self.setup_call = None

        def setup_support(self, paths, **kwargs):  # noqa: ANN001, ANN201
            self.setup_call = (list(paths), kwargs)

        def score_paths(self, paths):  # noqa: ANN001, ANN201
            scores = np.asarray([float(Path(path).stem) for path in paths], dtype=np.float32)
            maps = np.stack([np.full((64, 64), score, dtype=np.float32) for score in scores])
            return scores, maps

    paths = [_image(tmp_path / f"{index}.png", index) for index in range(1, 5)]
    backend = _Backend()
    detector = VisionLogSAD(
        backend=backend,
        class_name="pushpins",
        device="cpu",
        contamination=0.25,
    ).fit(paths)

    assert detector.decision_scores_ == pytest.approx([1.0, 2.0, 3.0, 4.0])
    assert backend.setup_call == ([path.resolve() for path in paths], {"class_name": "pushpins"})
    assert detector.decision_function(paths[:2]) == pytest.approx([1.0, 2.0])
    assert detector.predict(paths).tolist() == [0, 0, 0, 1]
    assert detector.predict_anomaly_map(paths[:2]).shape == (2, 64, 64)

    with pytest.raises(TypeError, match="requires image paths"):
        detector.decision_function([np.zeros((8, 8, 3), dtype=np.uint8)])


def test_logsad_rejects_unsupported_category_and_missing_runtime(tmp_path: Path) -> None:
    from pyimgano.models.logsad import VisionLogSAD

    image = _image(tmp_path / "normal.png", 0)
    with pytest.raises(ValueError, match="supports categories"):
        VisionLogSAD(class_name="bottle", device="cpu").fit([image])

    with pytest.raises(ValueError, match="requires repository_path"):
        VisionLogSAD(class_name="pushpins", device="cpu").fit([image])
