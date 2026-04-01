from __future__ import annotations

import numpy as np


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


def test_embedding_industrial_wrappers_are_registered() -> None:
    from pyimgano.models import list_models

    names = set(list_models())
    assert "vision_structural_ecod" in names
    assert "vision_structural_copod" in names
    assert "vision_structural_knn" in names
    assert "vision_structural_lof" in names
    assert "vision_structural_extra_trees_density" in names
    assert "vision_structural_mcd" in names
    assert "vision_structural_pca_md" in names
    assert "vision_structural_iforest" in names
    assert "vision_structural_lid" in names
    assert "vision_structural_mst_outlier" in names
    assert "vision_resnet18_ecod" in names
    assert "vision_resnet18_copod" in names
    assert "vision_resnet18_iforest" in names
    assert "vision_resnet18_knn" in names
    assert "vision_resnet18_knn_cosine" in names
    assert "vision_resnet18_knn_cosine_calibrated" in names
    assert "vision_resnet18_cosine_mahalanobis" in names
    assert "vision_resnet18_lid" in names
    assert "vision_resnet18_lof" in names
    assert "vision_resnet18_mcd" in names
    assert "vision_resnet18_mst_outlier" in names
    assert "vision_resnet18_pca_md" in names
    assert "vision_resnet18_extra_trees_density" in names
    assert "vision_resnet18_oddoneout" in names
    assert "vision_resnet18_torch_ae" in names
    assert "vision_torchscript_ecod" in names
    assert "vision_torchscript_copod" in names
    assert "vision_torchscript_iforest" in names
    assert "vision_torchscript_knn_cosine" in names
    assert "vision_torchscript_knn_cosine_calibrated" in names
    assert "vision_torchscript_cosine_mahalanobis" in names
    assert "vision_torchscript_lid" in names
    assert "vision_torchscript_lof" in names
    assert "vision_torchscript_mcd" in names
    assert "vision_torchscript_mst_outlier" in names
    assert "vision_torchscript_pca_md" in names
    assert "vision_torchscript_extra_trees_density" in names
    assert "vision_torchscript_oddoneout" in names
    assert "vision_onnx_ecod" in names
    assert "vision_onnx_copod" in names
    assert "vision_onnx_iforest" in names
    assert "vision_onnx_knn_cosine" in names
    assert "vision_onnx_knn_cosine_calibrated" in names
    assert "vision_onnx_cosine_mahalanobis" in names
    assert "vision_onnx_lid" in names
    assert "vision_onnx_lof" in names
    assert "vision_onnx_mcd" in names
    assert "vision_onnx_mst_outlier" in names
    assert "vision_onnx_pca_md" in names
    assert "vision_onnx_extra_trees_density" in names
    assert "vision_onnx_oddoneout" in names


def test_torchscript_industrial_wrappers_smoke(tmp_path) -> None:
    import pytest

    torch = pytest.importorskip("torch")
    nn = torch.nn
    from pyimgano.utils.torchscript_safe import trace_module

    class ToyEmbed(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))

        def forward(self, x):  # noqa: ANN001, ANN201 - torchscript signature
            y = self.pool(self.conv(x))
            return y.flatten(1)

    model = ToyEmbed().eval()
    example = torch.zeros((1, 3, 32, 32), dtype=torch.float32)
    scripted = trace_module(model, example)

    ckpt = tmp_path / "toy_embed.pt"
    scripted.save(str(ckpt))

    from PIL import Image

    rng = np.random.default_rng(0)
    train_paths = []
    for i in range(6):
        arr = rng.integers(0, 255, size=(40, 40, 3), dtype=np.uint8)
        p = tmp_path / f"train_{i}.png"
        Image.fromarray(arr, mode="RGB").save(str(p))
        train_paths.append(str(p))

    test_paths = train_paths[:2]

    from pyimgano.models import create_model

    for name in [
        "vision_torchscript_ecod",
        "vision_torchscript_copod",
        "vision_torchscript_iforest",
        "vision_torchscript_knn_cosine",
        "vision_torchscript_knn_cosine_calibrated",
        "vision_torchscript_cosine_mahalanobis",
    ]:
        det = create_model(
            name,
            contamination=0.2,
            checkpoint_path=str(ckpt),
            device="cpu",
            batch_size=4,
            image_size=32,
        )
        det.fit(train_paths)
        scores = np.asarray(det.decision_function(test_paths), dtype=np.float64).reshape(-1)
        assert scores.shape == (len(test_paths),)
        assert np.all(np.isfinite(scores))


def test_onnx_industrial_wrappers_smoke(tmp_path) -> None:
    import pytest

    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")

    from pyimgano.onnx_export_cli import main as export_main

    onnx_path = tmp_path / "resnet18_embed.onnx"
    rc = export_main(
        [
            "--backbone",
            "resnet18",
            "--no-pretrained",
            "--image-size",
            "32",
            "--opset",
            "17",
            "--dynamic-batch",
            "--no-verify",
            "--out",
            str(onnx_path),
        ]
    )
    assert rc == 0
    assert onnx_path.exists()
    assert onnx_path.stat().st_size > 0

    from PIL import Image

    rng = np.random.default_rng(0)
    train_paths = []
    for i in range(6):
        arr = rng.integers(0, 255, size=(40, 40, 3), dtype=np.uint8)
        p = tmp_path / f"train_onnx_{i}.png"
        Image.fromarray(arr, mode="RGB").save(str(p))
        train_paths.append(str(p))

    test_paths = train_paths[:2]

    from pyimgano.models import create_model

    for name in [
        "vision_onnx_ecod",
        "vision_onnx_copod",
        "vision_onnx_iforest",
        "vision_onnx_knn_cosine",
        "vision_onnx_knn_cosine_calibrated",
        "vision_onnx_cosine_mahalanobis",
    ]:
        det = create_model(
            name,
            contamination=0.2,
            checkpoint_path=str(onnx_path),
            device="cpu",
            batch_size=4,
            image_size=32,
        )
        det.fit(train_paths)
        scores = np.asarray(det.decision_function(test_paths), dtype=np.float64).reshape(-1)
        assert scores.shape == (len(test_paths),)
        assert np.all(np.isfinite(scores))


def test_embedding_industrial_wrappers_accept_identity_extractor_on_vectors() -> None:
    import pytest

    from pyimgano.models import create_model

    rng = np.random.default_rng(0)
    x = [rng.normal(size=(16,)).astype(np.float32) for _ in range(80)]

    for name in [
        "vision_structural_copod",
        "vision_structural_knn",
        "vision_structural_lof",
        "vision_structural_extra_trees_density",
        "vision_structural_mcd",
        "vision_structural_pca_md",
    ]:
        det = create_model(name, contamination=0.2, feature_extractor="identity")
        det.fit(x)
        scores = np.asarray(det.decision_function(x[:5]), dtype=np.float64).reshape(-1)
        assert scores.shape == (5,)
        assert np.all(np.isfinite(scores))

    for name in [
        "vision_resnet18_ecod",
        "vision_resnet18_copod",
        "vision_resnet18_iforest",
        "vision_resnet18_knn",
        "vision_resnet18_knn_cosine",
        "vision_resnet18_knn_cosine_calibrated",
        "vision_resnet18_cosine_mahalanobis",
        "vision_resnet18_lid",
        "vision_resnet18_lof",
        "vision_resnet18_mcd",
        "vision_resnet18_mst_outlier",
        "vision_resnet18_pca_md",
        "vision_resnet18_extra_trees_density",
        "vision_resnet18_oddoneout",
    ]:
        det = create_model(name, contamination=0.2, embedding_extractor="identity")
        det.fit(x)
        scores = np.asarray(det.decision_function(x[:5]), dtype=np.float64).reshape(-1)
        assert scores.shape == (5,)
        assert np.all(np.isfinite(scores))

    for name in [
        "vision_torchscript_ecod",
        "vision_torchscript_copod",
        "vision_torchscript_iforest",
        "vision_torchscript_knn_cosine",
        "vision_torchscript_knn_cosine_calibrated",
        "vision_torchscript_cosine_mahalanobis",
        "vision_torchscript_lid",
        "vision_torchscript_lof",
        "vision_torchscript_mcd",
        "vision_torchscript_mst_outlier",
        "vision_torchscript_pca_md",
        "vision_torchscript_extra_trees_density",
        "vision_torchscript_oddoneout",
        "vision_onnx_ecod",
        "vision_onnx_copod",
        "vision_onnx_iforest",
        "vision_onnx_knn_cosine",
        "vision_onnx_knn_cosine_calibrated",
        "vision_onnx_cosine_mahalanobis",
        "vision_onnx_lid",
        "vision_onnx_lof",
        "vision_onnx_mcd",
        "vision_onnx_mst_outlier",
        "vision_onnx_pca_md",
        "vision_onnx_extra_trees_density",
        "vision_onnx_oddoneout",
    ]:
        det = create_model(name, contamination=0.2, embedding_extractor="identity")
        det.fit(x)
        scores = np.asarray(det.decision_function(x[:5]), dtype=np.float64).reshape(-1)
        assert scores.shape == (5,)
        assert np.all(np.isfinite(scores))

    if not _torch_available():
        pytest.skip("torch is not installed (required for core_torch_autoencoder)")

    det = create_model(
        "vision_resnet18_torch_ae",
        contamination=0.25,
        embedding_extractor="identity",
        core_kwargs={
            "epochs": 2,
            "batch_size": 16,
            "lr": 1e-3,
            "device": "cpu",
            "random_state": 0,
            "hidden_dims": (12, 4),
        },
    )
    det.fit(x)
    scores = np.asarray(det.decision_function(x[:5]), dtype=np.float64).reshape(-1)
    assert scores.shape == (5,)
    assert np.all(np.isfinite(scores))
