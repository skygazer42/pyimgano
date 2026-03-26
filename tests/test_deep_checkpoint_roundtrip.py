from __future__ import annotations

import numpy as np
import pytest


def _write_rgb(path, array: np.ndarray) -> None:  # noqa: ANN001 - test helper
    from PIL import Image

    Image.fromarray(array, mode="RGB").save(path)


class _DummyDinoModel:
    def __init__(self, torch_module, *, channels: int = 8) -> None:  # noqa: ANN001
        self._torch = torch_module
        self.channels = int(channels)
        self.scale = torch_module.nn.Parameter(torch_module.tensor(1.0))

    def eval(self):  # noqa: ANN201 - torch-like test stub
        return self

    def to(self, device):  # noqa: ANN001, ANN201 - torch-like test stub
        del device
        return self

    def state_dict(self):  # noqa: ANN201 - torch-like test stub
        return {"scale": self.scale.detach().clone()}

    def load_state_dict(self, state_dict, strict=False):  # noqa: ANN001, ANN201
        del strict
        self.scale.data.copy_(state_dict["scale"])
        return self

    def forward_features(self, x):  # noqa: ANN001, ANN201 - torch-like test stub
        pooled = self._torch.nn.functional.avg_pool2d(x, kernel_size=4, stride=4)
        tokens = pooled.mean(dim=1, keepdim=True)
        tokens = tokens.permute(0, 2, 3, 1).reshape(x.shape[0], -1, 1)
        tokens = tokens.repeat(1, 1, self.channels) * self.scale
        return {"x_norm_patchtokens": tokens}


def _install_dummy_torchhub_dino(monkeypatch) -> None:
    import torch
    from PIL import Image

    from pyimgano.models.anomalydino import TorchHubDinoV2Embedder

    def _ensure_loaded(self) -> None:  # noqa: ANN001
        if self._model is not None:
            return
        self._image_cls = Image
        self._torch = torch

        def _transform(image):  # noqa: ANN001
            arr = np.asarray(image, dtype=np.float32) / 255.0
            return torch.from_numpy(arr).permute(2, 0, 1)

        self._transform = _transform
        self._model = _DummyDinoModel(torch, channels=8)
        self._model.eval()
        self._model.to(self.device)
        self._patch_size = 4

    monkeypatch.setattr(TorchHubDinoV2Embedder, "_ensure_loaded", _ensure_loaded, raising=True)


def test_padim_checkpoint_roundtrip_on_image_paths(tmp_path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("cv2")

    import pyimgano.models  # noqa: F401
    from pyimgano.models.registry import create_model
    from pyimgano.training.checkpointing import save_checkpoint
    from pyimgano.workbench.checkpoint_restore import load_checkpoint_into_detector

    train_paths = []
    for idx, value in enumerate((80, 82, 84, 86), start=1):
        path = tmp_path / f"train_{idx}.png"
        img = np.full((48, 48, 3), value, dtype=np.uint8)
        _write_rgb(path, img)
        train_paths.append(str(path))

    normal_path = tmp_path / "normal.png"
    anomaly_path = tmp_path / "anomaly.png"
    _write_rgb(normal_path, np.full((48, 48, 3), 83, dtype=np.uint8))
    anomaly = np.full((48, 48, 3), 83, dtype=np.uint8)
    anomaly[12:30, 12:30, :] = 240
    _write_rgb(anomaly_path, anomaly)

    def _make_detector():
        return create_model(
            "vision_padim",
            contamination=0.2,
            backbone="resnet18",
            d_reduced=16,
            image_size=64,
            pretrained=False,
            device="cpu",
            projection_fit_samples=2,
            covariance_eps=0.01,
            random_state=42,
        )

    detector = _make_detector()
    detector.fit(train_paths)
    expected_scores = np.asarray(
        detector.decision_function([str(normal_path), str(anomaly_path)]),
        dtype=np.float64,
    )
    expected_maps = np.asarray(
        detector.predict_anomaly_map([str(normal_path), str(anomaly_path)]),
        dtype=np.float32,
    )
    expected_threshold = float(detector.threshold_)

    ckpt_path = tmp_path / "padim.ckpt"
    save_checkpoint(detector, ckpt_path)

    restored = _make_detector()
    load_checkpoint_into_detector(restored, ckpt_path)

    restored_scores = np.asarray(
        restored.decision_function([str(normal_path), str(anomaly_path)]),
        dtype=np.float64,
    )
    restored_maps = np.asarray(
        restored.predict_anomaly_map([str(normal_path), str(anomaly_path)]),
        dtype=np.float32,
    )

    np.testing.assert_allclose(restored_scores, expected_scores, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(restored_maps, expected_maps, rtol=1e-5, atol=1e-5)
    assert float(restored.threshold_) == pytest.approx(expected_threshold)


def test_draem_checkpoint_roundtrip_on_image_paths(tmp_path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("cv2")

    import pyimgano.models  # noqa: F401
    from pyimgano.models.registry import create_model
    from pyimgano.training.checkpointing import save_checkpoint
    from pyimgano.workbench.checkpoint_restore import load_checkpoint_into_detector

    train_paths = []
    for idx, value in enumerate((80, 82, 84, 86), start=1):
        path = tmp_path / f"train_{idx}.png"
        img = np.full((32, 32, 3), value, dtype=np.uint8)
        _write_rgb(path, img)
        train_paths.append(str(path))

    normal_path = tmp_path / "normal.png"
    anomaly_path = tmp_path / "anomaly.png"
    _write_rgb(normal_path, np.full((32, 32, 3), 83, dtype=np.uint8))
    anomaly = np.full((32, 32, 3), 83, dtype=np.uint8)
    anomaly[10:22, 10:22, :] = 240
    _write_rgb(anomaly_path, anomaly)

    def _make_detector():
        return create_model(
            "vision_draem",
            contamination=0.2,
            image_size=32,
            epochs=1,
            batch_size=2,
            lr=1e-4,
            num_workers=0,
            device="cpu",
        )

    detector = _make_detector()
    detector.fit(train_paths)
    expected_scores = np.asarray(
        detector.decision_function([str(normal_path), str(anomaly_path)]),
        dtype=np.float64,
    )
    expected_maps = np.asarray(
        detector.predict_anomaly_map([str(normal_path), str(anomaly_path)]),
        dtype=np.float32,
    )
    expected_threshold = float(detector.threshold_)

    ckpt_path = tmp_path / "draem.ckpt"
    save_checkpoint(detector, ckpt_path)

    restored = _make_detector()
    load_checkpoint_into_detector(restored, ckpt_path)

    restored_scores = np.asarray(
        restored.decision_function([str(normal_path), str(anomaly_path)]),
        dtype=np.float64,
    )
    restored_maps = np.asarray(
        restored.predict_anomaly_map([str(normal_path), str(anomaly_path)]),
        dtype=np.float32,
    )

    np.testing.assert_allclose(restored_scores, expected_scores, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(restored_maps, expected_maps, rtol=1e-5, atol=1e-5)
    assert float(restored.threshold_) == pytest.approx(expected_threshold)


def test_patchcore_checkpoint_roundtrip_on_image_paths(tmp_path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("cv2")

    import pyimgano.models  # noqa: F401
    from pyimgano.models.registry import create_model
    from pyimgano.training.checkpointing import save_checkpoint
    from pyimgano.workbench.checkpoint_restore import load_checkpoint_into_detector

    train_paths = []
    for idx, value in enumerate((80, 82, 84, 86), start=1):
        path = tmp_path / f"train_{idx}.png"
        img = np.full((32, 32, 3), value, dtype=np.uint8)
        _write_rgb(path, img)
        train_paths.append(str(path))

    normal_path = tmp_path / "normal.png"
    anomaly_path = tmp_path / "anomaly.png"
    _write_rgb(normal_path, np.full((32, 32, 3), 83, dtype=np.uint8))
    anomaly = np.full((32, 32, 3), 83, dtype=np.uint8)
    anomaly[10:22, 10:22, :] = 240
    _write_rgb(anomaly_path, anomaly)

    def _make_detector():
        return create_model(
            "vision_patchcore",
            contamination=0.2,
            backbone="resnet50",
            layers=["layer2", "layer3"],
            coreset_sampling_ratio=1.0,
            feature_projection_dim=32,
            projection_fit_samples=2,
            n_neighbors=1,
            knn_backend="sklearn",
            memory_bank_dtype="float32",
            random_seed=0,
            pretrained=False,
            device="cpu",
        )

    detector = _make_detector()
    detector.fit(train_paths)
    expected_scores = np.asarray(
        detector.decision_function([str(normal_path), str(anomaly_path)]),
        dtype=np.float64,
    )
    expected_map = np.asarray(detector.get_anomaly_map(str(anomaly_path)), dtype=np.float32)
    expected_threshold = float(detector.threshold_)

    ckpt_path = tmp_path / "patchcore.ckpt"
    save_checkpoint(detector, ckpt_path)

    restored = _make_detector()
    load_checkpoint_into_detector(restored, ckpt_path)

    restored_scores = np.asarray(
        restored.decision_function([str(normal_path), str(anomaly_path)]),
        dtype=np.float64,
    )
    restored_map = np.asarray(restored.get_anomaly_map(str(anomaly_path)), dtype=np.float32)

    np.testing.assert_allclose(restored_scores, expected_scores, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(restored_map, expected_map, rtol=1e-5, atol=1e-5)
    assert float(restored.threshold_) == pytest.approx(expected_threshold)


def test_fastflow_checkpoint_roundtrip_on_image_paths(tmp_path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("cv2")

    import pyimgano.models  # noqa: F401
    from pyimgano.models.registry import create_model
    from pyimgano.training.checkpointing import save_checkpoint
    from pyimgano.workbench.checkpoint_restore import load_checkpoint_into_detector

    train_paths = []
    for idx, value in enumerate((80, 82, 84, 86), start=1):
        path = tmp_path / f"train_fastflow_{idx}.png"
        img = np.full((32, 32, 3), value, dtype=np.uint8)
        _write_rgb(path, img)
        train_paths.append(str(path))

    eval_paths = train_paths[:2]

    kwargs = {
        "contamination": 0.2,
        "backbone": "resnet18",
        "pretrained_backbone": False,
        "selected_layers": ("layer3",),
        "embedding_dim": 32,
        "n_flow_steps": 2,
        "flow_hidden_ratio": 1.0,
        "lr": 1e-4,
        "epoch_num": 1,
        "batch_size": 2,
        "device": "cpu",
        "verbose": 0,
        "random_state": 42,
    }

    detector = create_model("vision_fastflow", **kwargs)
    detector.fit(train_paths)
    expected_scores = np.asarray(detector.decision_function(eval_paths), dtype=np.float64)
    expected_threshold = float(detector.threshold_)

    ckpt_path = tmp_path / "fastflow.ckpt"
    save_checkpoint(detector, ckpt_path)

    restored = create_model("vision_fastflow", **kwargs)
    load_checkpoint_into_detector(restored, ckpt_path)

    restored_scores = np.asarray(restored.decision_function(eval_paths), dtype=np.float64)
    np.testing.assert_allclose(restored_scores, expected_scores, rtol=1e-6, atol=1e-6)
    assert float(restored.threshold_) == pytest.approx(expected_threshold)


def test_softpatch_checkpoint_roundtrip_on_image_paths(tmp_path, monkeypatch) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("cv2")

    import pyimgano.models  # noqa: F401
    from pyimgano.models.anomalydino import TorchHubDinoV2Embedder
    from pyimgano.models.registry import create_model
    from pyimgano.training.checkpointing import save_checkpoint
    from pyimgano.workbench.checkpoint_restore import load_checkpoint_into_detector

    _install_dummy_torchhub_dino(monkeypatch)

    train_paths = []
    for idx, value in enumerate((80, 82, 84, 86), start=1):
        path = tmp_path / f"train_softpatch_{idx}.png"
        img = np.full((32, 32, 3), value, dtype=np.uint8)
        _write_rgb(path, img)
        train_paths.append(str(path))

    normal_path = tmp_path / "softpatch_normal.png"
    anomaly_path = tmp_path / "softpatch_anomaly.png"
    _write_rgb(normal_path, np.full((32, 32, 3), 83, dtype=np.uint8))
    anomaly = np.full((32, 32, 3), 83, dtype=np.uint8)
    anomaly[10:22, 10:22, :] = 240
    _write_rgb(anomaly_path, anomaly)

    def _make_detector():
        embedder = TorchHubDinoV2Embedder(model_name="dinov2_vits14", device="cpu", image_size=32)
        return create_model(
            "vision_softpatch",
            embedder=embedder,
            contamination=0.2,
            knn_backend="sklearn",
            n_neighbors=1,
            coreset_sampling_ratio=1.0,
            random_seed=0,
            train_patch_outlier_quantile=0.0,
            aggregation_method="topk_mean",
            aggregation_topk=0.05,
            pretrained=True,
            device="cpu",
            image_size=32,
            dino_model_name="dinov2_vits14",
        )

    detector = _make_detector()
    detector.fit(train_paths)
    expected_scores = np.asarray(
        detector.decision_function([str(normal_path), str(anomaly_path)]),
        dtype=np.float64,
    )
    expected_maps = np.asarray(
        detector.predict_anomaly_map([str(normal_path), str(anomaly_path)]),
        dtype=np.float32,
    )
    expected_threshold = float(detector.threshold_)

    ckpt_path = tmp_path / "softpatch.ckpt"
    save_checkpoint(detector, ckpt_path)

    restored = _make_detector()
    load_checkpoint_into_detector(restored, ckpt_path)

    restored_scores = np.asarray(
        restored.decision_function([str(normal_path), str(anomaly_path)]),
        dtype=np.float64,
    )
    restored_maps = np.asarray(
        restored.predict_anomaly_map([str(normal_path), str(anomaly_path)]),
        dtype=np.float32,
    )

    np.testing.assert_allclose(restored_scores, expected_scores, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(restored_maps, expected_maps, rtol=1e-5, atol=1e-5)
    assert float(restored.threshold_) == pytest.approx(expected_threshold)


def test_anomalydino_checkpoint_roundtrip_on_image_paths(tmp_path, monkeypatch) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("cv2")

    import pyimgano.models  # noqa: F401
    from pyimgano.models.anomalydino import TorchHubDinoV2Embedder
    from pyimgano.models.registry import create_model
    from pyimgano.training.checkpointing import save_checkpoint
    from pyimgano.workbench.checkpoint_restore import load_checkpoint_into_detector

    _install_dummy_torchhub_dino(monkeypatch)

    train_paths = []
    for idx, value in enumerate((80, 82, 84, 86), start=1):
        path = tmp_path / f"train_anomalydino_{idx}.png"
        img = np.full((32, 32, 3), value, dtype=np.uint8)
        _write_rgb(path, img)
        train_paths.append(str(path))

    normal_path = tmp_path / "anomalydino_normal.png"
    anomaly_path = tmp_path / "anomalydino_anomaly.png"
    _write_rgb(normal_path, np.full((32, 32, 3), 83, dtype=np.uint8))
    anomaly = np.full((32, 32, 3), 83, dtype=np.uint8)
    anomaly[10:22, 10:22, :] = 240
    _write_rgb(anomaly_path, anomaly)

    def _make_detector():
        embedder = TorchHubDinoV2Embedder(model_name="dinov2_vits14", device="cpu", image_size=32)
        return create_model(
            "vision_anomalydino",
            embedder=embedder,
            contamination=0.2,
            knn_backend="sklearn",
            n_neighbors=1,
            coreset_sampling_ratio=1.0,
            random_seed=0,
            aggregation_method="topk_mean",
            aggregation_topk=0.05,
            pretrained=True,
            device="cpu",
            image_size=32,
            dino_model_name="dinov2_vits14",
        )

    detector = _make_detector()
    detector.fit(train_paths)
    expected_scores = np.asarray(
        detector.decision_function([str(normal_path), str(anomaly_path)]),
        dtype=np.float64,
    )
    expected_maps = np.asarray(
        detector.predict_anomaly_map([str(normal_path), str(anomaly_path)]),
        dtype=np.float32,
    )
    expected_threshold = float(detector.threshold_)

    ckpt_path = tmp_path / "anomalydino.ckpt"
    save_checkpoint(detector, ckpt_path)

    restored = _make_detector()
    load_checkpoint_into_detector(restored, ckpt_path)

    restored_scores = np.asarray(
        restored.decision_function([str(normal_path), str(anomaly_path)]),
        dtype=np.float64,
    )
    restored_maps = np.asarray(
        restored.predict_anomaly_map([str(normal_path), str(anomaly_path)]),
        dtype=np.float32,
    )

    np.testing.assert_allclose(restored_scores, expected_scores, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(restored_maps, expected_maps, rtol=1e-5, atol=1e-5)
    assert float(restored.threshold_) == pytest.approx(expected_threshold)
