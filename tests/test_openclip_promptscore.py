import numpy as np

from pyimgano import models


class _FakeEmbedder:
    def embed(self, image_path: str):
        # 4 patches (2x2), dim=2, deterministic by path.
        if "anomaly" in image_path:
            patches = np.tile(np.array([[0.0, 1.0]], dtype=np.float32), (4, 1))
        else:
            patches = np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (4, 1))
        return patches, (2, 2), (8, 8)


def test_openclip_promptscore_fit_predict_and_map():
    detector = models.create_model(
        "vision_openclip_promptscore",
        embedder=_FakeEmbedder(),
        text_features_normal=np.array([1.0, 0.0], dtype=np.float32),
        text_features_anomaly=np.array([0.0, 1.0], dtype=np.float32),
        contamination=0.1,
        aggregation_method="topk_mean",
        aggregation_topk=0.25,
    )
    detector.fit(["normal_1.png", "normal_2.png"])
    scores = detector.decision_function(["normal_x.png", "anomaly_x.png"])
    assert float(scores[1]) > float(scores[0])

    amap = detector.get_anomaly_map("anomaly_x.png")
    assert amap.shape == (8, 8)
    assert np.isfinite(amap).all()


def test_openclip_promptscore_accepts_numpy_inputs_with_custom_embedder():
    class _ArrayEmbedder:
        def embed(self, image):
            arr = np.asarray(image)
            is_anom = float(arr.max()) > 0.0
            patches = (
                np.tile(np.array([[0.0, 1.0]], dtype=np.float32), (4, 1))
                if is_anom
                else np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (4, 1))
            )
            return patches, (2, 2), (8, 8)

    detector = models.create_model(
        "vision_openclip_promptscore",
        embedder=_ArrayEmbedder(),
        text_features_normal=np.array([1.0, 0.0], dtype=np.float32),
        text_features_anomaly=np.array([0.0, 1.0], dtype=np.float32),
        contamination=0.1,
    )

    normal = np.zeros((8, 8, 3), dtype=np.uint8)
    anomaly = normal.copy()
    anomaly[2:4, 2:4, :] = 255

    detector.fit([normal, normal])
    scores = detector.decision_function([normal, anomaly])
    assert float(scores[1]) > float(scores[0])

    amap = detector.get_anomaly_map(anomaly)
    assert amap.shape == (8, 8)
    assert np.isfinite(amap).all()


def test_openclip_promptscore_caches_text_features_by_class_name():
    import torch

    from pyimgano.models.openclip_backend import VisionOpenCLIPPromptScore

    class _FakeModel:
        def __init__(self) -> None:
            self.encode_text_calls = 0

        def eval(self):
            return self

        def to(self, _device):
            return self

        def encode_text(self, tokens):
            self.encode_text_calls += 1
            return torch.ones((tokens.shape[0], 2), dtype=torch.float32)

    class _FakeOpenCLIP:
        def __init__(self) -> None:
            self.create_model_calls = 0
            self.tokenize_calls = 0
            self.model = _FakeModel()

        def create_model_and_transforms(self, _model_name: str, pretrained=None, **_kwargs):
            self.create_model_calls += 1

            def _preprocess(_image):
                raise AssertionError("preprocess should not be called by this test")

            return self.model, None, _preprocess

        def tokenize(self, prompts):
            self.tokenize_calls += 1
            return torch.zeros((len(prompts), 1), dtype=torch.int64)

    fake_open_clip = _FakeOpenCLIP()
    detector = VisionOpenCLIPPromptScore(
        open_clip_module=fake_open_clip,
        device="cpu",
        openclip_pretrained=None,
    )

    detector._ensure_text_features()
    assert fake_open_clip.model.encode_text_calls == 2  # normal + anomaly

    # Second call should be cached.
    detector._ensure_text_features()
    assert fake_open_clip.model.encode_text_calls == 2

    # Setting the same class name should not invalidate the cache.
    detector.set_class_name("object")
    detector._ensure_text_features()
    assert fake_open_clip.model.encode_text_calls == 2

    # Changing the class name should re-encode.
    detector.set_class_name("widget")
    detector._ensure_text_features()
    assert fake_open_clip.model.encode_text_calls == 4
