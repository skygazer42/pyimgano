from __future__ import annotations

import numpy as np
import pytest

import pyimgano.models as models


class FakePatchSearchBackend:
    def fit(self, train_patches):
        self.train_centroid = np.mean(np.concatenate(train_patches, axis=0), axis=0)
        return self

    def score(self, patch_grid):
        delta = patch_grid - self.train_centroid[None, :]
        patch_scores = np.linalg.norm(delta, axis=1)
        return float(np.max(patch_scores)), patch_scores


def test_visionad_scores_anomaly_higher_than_normal() -> None:
    detector = models.create_model(
        "vision_visionad",
        search_backend=FakePatchSearchBackend(),
        embedder=lambda image: image,
        contamination=0.25,
    )
    train = [np.array([[0.0, 0.0], [0.1, 0.0]], dtype=np.float32)]
    test = [
        np.array([[0.0, 0.0], [0.1, 0.1]], dtype=np.float32),
        np.array([[5.0, 5.0], [5.1, 5.1]], dtype=np.float32),
    ]
    detector.fit(train)
    scores = detector.decision_function(test)
    assert float(scores[1]) > float(scores[0])


def test_visionad_released_defaults_and_registry_metadata() -> None:
    from pyimgano.models.registry import MODEL_REGISTRY
    from pyimgano.models.visionad import (
        PAPER_BASE_LAYERS,
        PAPER_BASE_MODEL_NAME,
        PAPER_CROP_SIZE,
        PAPER_MAP_SIZE,
        PAPER_QUERY_VIEWS,
        PAPER_RESIZE_SIZE,
        PAPER_SUPPORT_VIEWS,
        RELEASE_LARGE_LAYERS,
        RELEASE_MODEL_NAME,
        TorchVisionADBackend,
    )

    released = TorchVisionADBackend()
    paper_baseline = TorchVisionADBackend(model_name=PAPER_BASE_MODEL_NAME)

    assert released.model_name == RELEASE_MODEL_NAME
    assert released.interested_layers == RELEASE_LARGE_LAYERS == tuple(range(4, 19))
    assert paper_baseline.interested_layers == PAPER_BASE_LAYERS == tuple(range(2, 10))
    assert (released.resize_size, released.crop_size, released.map_size) == (
        PAPER_RESIZE_SIZE,
        PAPER_CROP_SIZE,
        PAPER_MAP_SIZE,
    )
    assert PAPER_SUPPORT_VIEWS == (
        "identity",
        "rot90",
        "rot180",
        "rot270",
        "flip_y",
        "flip_x",
    )
    assert PAPER_QUERY_VIEWS == ("identity", "flip_y", "positive_clamp")
    assert MODEL_REGISTRY.info("vision_visionad").metadata["paper_fidelity"] == ("paper-adaptation")

    with pytest.raises(ValueError, match="pretrained=True"):
        models.create_model("vision_visionad")


def test_visionad_fuses_raw_intermediate_tokens_before_search() -> None:
    torch = pytest.importorskip("torch")
    from pyimgano.models.visionad import _forward_fused_tokens

    class AddBlock(torch.nn.Module):
        def __init__(self, value: float) -> None:
            super().__init__()
            self.value = value

        def forward(self, tokens):
            return tokens + self.value

    class FakeBackbone:
        blocks = torch.nn.ModuleList([AddBlock(1), AddBlock(2), AddBlock(3)])

        @staticmethod
        def prepare_tokens(images):
            return images

    fused = _forward_fused_tokens(FakeBackbone(), torch.zeros(1, 5, 2), (0, 2))

    # Block 0 produces 1; block 2 produces 1+2+3=6; VisionAD averages them.
    assert torch.equal(fused, torch.full((1, 5, 2), 3.5))


def test_visionad_category_indexed_cosine_memory() -> None:
    features = {
        "support_a": (np.array([[1.0, 0.0]], np.float32), [1.0, 0.0]),
        "support_b": (np.array([[0.0, 1.0]], np.float32), [0.0, 1.0]),
        "query_b": (np.array([[0.0, 1.0]], np.float32), [0.0, 1.0]),
        # Its patch matches category A, but its CLS/global feature retrieves B.
        "anomaly_b": (np.array([[1.0, 0.0]], np.float32), [0.0, 1.0]),
    }

    def embed(image):
        patches, global_feature = features[image]
        return patches, np.asarray(global_feature, np.float32), (1, 1), (8, 8)

    detector = models.create_model("vision_visionad", embedder=embed, contamination=0.25)
    detector.fit(["support_a", "support_b"], y=["a", "b"])

    scores = detector.decision_function(["query_b", "anomaly_b"])
    assert scores[0] == pytest.approx(0.0)
    assert scores[1] == pytest.approx(1.0)
    assert detector.get_anomaly_map("anomaly_b").shape == (1, 1)


def test_torch_visionad_backend_runs_all_released_views() -> None:
    torch = pytest.importorskip("torch")
    from pyimgano.models.visionad import PAPER_QUERY_VIEWS, TorchVisionADBackend

    class AddBlock(torch.nn.Module):
        def forward(self, tokens):
            return tokens + 0.01

    class TinyDino(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = torch.nn.ModuleList([AddBlock(), AddBlock()])
            self.num_register_tokens = 1

        def prepare_tokens(self, images):
            pooled = torch.nn.functional.avg_pool2d(images.mean(1, keepdim=True), 7, 7)
            values = pooled.flatten(2).transpose(1, 2)
            patches = torch.cat([values, torch.ones_like(values)], dim=-1)
            cls = patches.mean(dim=1, keepdim=True)
            register = torch.zeros_like(cls)
            return torch.cat([cls, register, patches], dim=1)

    def preprocess(image):
        array = np.array(image, dtype=np.float32, copy=True)
        return torch.from_numpy(array).permute(2, 0, 1) / 255.0

    backend = TorchVisionADBackend(
        model=TinyDino(),
        preprocess=preprocess,
        interested_layers=(0, 1),
        resize_size=14,
        crop_size=14,
        map_size=8,
        batch_size=3,
    )
    normal = np.zeros((14, 14, 3), dtype=np.uint8)
    anomaly = np.full((14, 14, 3), 255, dtype=np.uint8)

    backend.fit([normal], ["part"])
    normal_score, normal_map = backend.score(normal)
    anomaly_score, anomaly_map = backend.score(anomaly)

    assert set(backend._memories) == set(PAPER_QUERY_VIEWS)
    assert all(memory["part"].shape[0] == 24 for memory in backend._memories.values())
    assert backend.selected_category_ == "part"
    assert normal_map.shape == anomaly_map.shape == (8, 8)
    assert anomaly_score > normal_score
