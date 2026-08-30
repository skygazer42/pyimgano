from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")


def _image(path: Path, value: int) -> Path:
    Image.fromarray(np.full((11, 13, 3), value, dtype=np.uint8)).save(path)
    return path


def test_filo_paper_contract_metadata_and_offline_constructor() -> None:
    from pyimgano.models import MODEL_REGISTRY
    from pyimgano.models.filopp import (
        OFFICIAL_COMMIT,
        PAPER_CLIP_BACKBONE,
        PAPER_CONTEXT_TOKENS,
        PAPER_FEATURE_LAYERS,
        PAPER_IMAGE_SIZE,
        VisionFiLoPP,
    )

    metadata = MODEL_REGISTRY.info("vision_filopp").metadata
    assert metadata["paper_fidelity"] == "external-backend"
    assert metadata["backend"] == "official-filo"
    assert metadata["paper"].startswith("FiLo: Zero-Shot")
    assert "FiLo++ source is not present" in metadata["compatibility_note"]
    assert OFFICIAL_COMMIT == "36ff29ca09ba8ba3af24d7654582aea856031400"
    assert PAPER_CLIP_BACKBONE == "ViT-L-14-336"
    assert PAPER_IMAGE_SIZE == 518
    assert PAPER_FEATURE_LAYERS == (6, 12, 18, 24)
    assert PAPER_CONTEXT_TOKENS == 12

    detector = VisionFiLoPP(device="cpu")
    assert detector.repository_path is None
    assert detector.grounding_device is None
    assert detector.allow_download is False


def test_filo_validates_released_network_dimensions() -> None:
    from pyimgano.models.filopp import _validate_author_model

    def layer() -> SimpleNamespace:
        return SimpleNamespace(in_features=1024, out_features=768)

    def convolution() -> SimpleNamespace:
        return SimpleNamespace(in_channels=1024, out_channels=768)

    decoder = SimpleNamespace(
        **{
            name: [convolution(), convolution(), convolution()]
            for name in ("fc_11", "fc_33", "fc_55", "fc_77", "fc_15", "fc_51")
        }
    )
    model = SimpleNamespace(
        args=SimpleNamespace(
            clip_model="ViT-L-14-336",
            clip_pretrained="openai",
            image_size=518,
            features_list=[6, 12, 18, 24],
            n_ctx=12,
        ),
        normal_prompt_learner=SimpleNamespace(ctx=torch.zeros(12, 768)),
        abnormal_prompt_learner=SimpleNamespace(ctx=torch.zeros(12, 768)),
        decoder_linear=SimpleNamespace(fc=[layer(), layer(), layer(), layer()]),
        decoder_cov=decoder,
        adapter=SimpleNamespace(
            fc=torch.nn.Sequential(
                torch.nn.Linear(768, 384, bias=False),
                torch.nn.ReLU(),
                torch.nn.Linear(384, 768, bias=False),
                torch.nn.SiLU(),
            )
        ),
    )

    _validate_author_model(model)
    model.args.image_size = 224
    with pytest.raises(ValueError, match="image_size"):
        _validate_author_model(model)


def test_filo_checkpoint_loader_rejects_zero_key_coverage() -> None:
    from pyimgano.models.filopp import _load_filo_state_dict

    model = torch.nn.Linear(2, 1)
    with pytest.raises(ValueError, match="no parameters matching"):
        _load_filo_state_dict(model, {"totally_wrong": torch.ones(1)})


def test_filo_released_localization_and_score_postprocessing() -> None:
    from pyimgano.models.filopp import _paper_boxes_and_position, _paper_score_and_map

    boxes, positions = _paper_boxes_and_position(
        torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.1, 0.1, 0.05, 0.05]]),
        ["scratch(0.8)", "object(0.9)"],
        ["scratch"],
    )
    assert positions == ["center"]
    assert boxes[0].tolist() == pytest.approx([207.2, 207.2, 310.8, 310.8])
    assert boxes[1].tolist() == pytest.approx([0.1, 0.1, 0.05, 0.05])

    branch = torch.empty(1, 2, 518, 518)
    branch[:, 0] = 0.25
    branch[:, 1] = 0.75
    score, anomaly_map = _paper_score_and_map(
        torch.tensor([0.2, 0.8]),
        [branch, branch.clone()],
        torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
        blur=lambda value: value,
    )
    assert score == pytest.approx(0.775)
    assert anomaly_map.shape == (518, 518)
    assert anomaly_map[5, 5].item() == pytest.approx(0.75)
    assert anomaly_map[20, 20].item() == pytest.approx(0.525)


def test_filo_wrapper_calibrates_paths_and_returns_paper_maps(tmp_path: Path) -> None:
    from pyimgano.models.filopp import VisionFiLoPP

    class _Backend:
        def score_paths(self, paths):  # noqa: ANN001, ANN201
            scores = np.asarray([float(Path(path).stem) for path in paths], dtype=np.float32)
            maps = np.stack([np.full((518, 518), score, dtype=np.float32) for score in scores])
            return scores, maps

    paths = [_image(tmp_path / f"{index}.png", index) for index in range(1, 5)]
    detector = VisionFiLoPP(
        dataset="mvtec",
        class_name="bottle",
        backend=_Backend(),
        device="cpu",
        contamination=0.25,
    ).fit(paths)

    assert detector.decision_scores_ == pytest.approx([1.0, 2.0, 3.0, 4.0])
    assert detector.predict(paths).tolist() == [0, 0, 0, 1]
    assert detector.predict_anomaly_map(paths[:2]).shape == (2, 518, 518)
    with pytest.raises(TypeError, match="requires image paths"):
        detector.decision_function([np.zeros((8, 8, 3), dtype=np.uint8)])


def test_filo_rejects_unreleased_or_incomplete_runtime(tmp_path: Path) -> None:
    from pyimgano.models.filopp import VisionFiLoPP

    image = _image(tmp_path / "normal.png", 0)
    with pytest.raises(ValueError, match="supports categories"):
        VisionFiLoPP(class_name="unknown", device="cpu", backend=object()).fit([image])
    with pytest.raises(ValueError, match="requires repository_path"):
        VisionFiLoPP(class_name="bottle", device="cpu").fit([image])
