from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")


def _tiny_rgb_batch() -> np.ndarray:
    return np.zeros((2, 16, 16, 3), dtype=np.uint8)


def test_fcdd_fit_passes_explicit_num_workers(monkeypatch) -> None:
    import pyimgano.models.fcdd as fcdd_module

    observed: dict[str, int | None] = {}
    original_dataloader = fcdd_module.DataLoader

    def _recording_dataloader(*args, **kwargs):
        observed["num_workers"] = kwargs.get("num_workers")
        return original_dataloader(*args, **kwargs)

    monkeypatch.setattr(fcdd_module, "DataLoader", _recording_dataloader)

    detector = fcdd_module.FCDD(objective="occ", epochs=0, batch_size=2, device="cpu")
    detector.fit(_tiny_rgb_batch())

    assert observed["num_workers"] == 0


def test_memae_fit_passes_explicit_weight_decay_and_num_workers(monkeypatch) -> None:
    import pyimgano.models.memae as memae_module

    observed: dict[str, float | int | None] = {}
    original_dataloader = memae_module.DataLoader
    original_adam = memae_module.torch.optim.Adam

    def _recording_dataloader(*args, **kwargs):
        observed["num_workers"] = kwargs.get("num_workers")
        return original_dataloader(*args, **kwargs)

    def _recording_adam(*args, **kwargs):
        observed["weight_decay"] = kwargs.get("weight_decay")
        return original_adam(*args, **kwargs)

    monkeypatch.setattr(memae_module, "DataLoader", _recording_dataloader)
    monkeypatch.setattr(memae_module.torch.optim, "Adam", _recording_adam)

    detector = memae_module.MemAE(mem_dim=16, epochs=0, batch_size=2, device="cpu")
    detector.fit(_tiny_rgb_batch())

    assert observed["weight_decay"] == pytest.approx(0.0)
    assert observed["num_workers"] == 0


def test_deep_svdd_fit_passes_explicit_num_workers(monkeypatch) -> None:
    import pyimgano.models.deep_svdd as deep_svdd_module

    observed: dict[str, int | None] = {}
    original_dataloader = deep_svdd_module.DataLoader

    def _recording_dataloader(*args, **kwargs):
        observed["num_workers"] = kwargs.get("num_workers")
        return original_dataloader(*args, **kwargs)

    monkeypatch.setattr(deep_svdd_module, "DataLoader", _recording_dataloader)

    detector = deep_svdd_module.CoreDeepSVDD(
        n_features=8,
        hidden_neurons=[16, 8],
        epochs=0,
        batch_size=2,
        verbose=0,
        random_state=0,
    )
    detector.fit(np.zeros((4, 8), dtype=np.float32))

    assert observed["num_workers"] == 0


def test_cflow_fit_passes_explicit_weight_decay(tmp_path, monkeypatch) -> None:
    import cv2

    import pyimgano.models.cflow as cflow_module

    observed: dict[str, float | None] = {}
    original_adam = cflow_module.Adam

    def _recording_adam(*args, **kwargs):
        observed["weight_decay"] = kwargs.get("weight_decay")
        return original_adam(*args, **kwargs)

    monkeypatch.setattr(cflow_module, "Adam", _recording_adam)

    image = np.full((32, 32, 3), 127, dtype=np.uint8)
    image_path = tmp_path / "normal.png"
    assert cv2.imwrite(str(image_path), image)

    detector = cflow_module.VisionCFlow(
        backbone="resnet18",
        pretrained_backbone=False,
        n_flows=1,
        epochs=0,
        batch_size=1,
        device="cpu",
    )
    monkeypatch.setattr(
        detector,
        "decision_function",
        lambda X, batch_size=None: np.zeros((len(list(X)),), dtype=np.float64),
    )

    detector.fit([str(image_path)])

    assert observed["weight_decay"] == pytest.approx(0.0)


def test_simplenet_fit_passes_explicit_weight_decay(tmp_path, monkeypatch) -> None:
    import cv2
    import torch

    import pyimgano.models.simplenet as simplenet_module

    observed: dict[str, float | None] = {}
    original_adam = simplenet_module.Adam

    def _recording_adam(*args, **kwargs):
        observed["weight_decay"] = kwargs.get("weight_decay")
        return original_adam(*args, **kwargs)

    monkeypatch.setattr(simplenet_module, "Adam", _recording_adam)

    image = np.full((32, 32, 3), 127, dtype=np.uint8)
    image_path = tmp_path / "normal.png"
    assert cv2.imwrite(str(image_path), image)

    detector = simplenet_module.VisionSimpleNet.__new__(simplenet_module.VisionSimpleNet)
    detector.batch_size = 1
    detector.device = "cpu"
    detector.lr = 1e-3
    detector.epochs = 0
    detector.transform = lambda img: torch.from_numpy(img).permute(2, 0, 1).float()
    detector.adapter = torch.nn.Linear(1, 1)
    detector._build_reference_features = lambda X: setattr(  # type: ignore[method-assign]
        detector, "reference_features", np.zeros((1, 1), dtype=np.float32)
    )
    detector.decision_function = lambda X: np.zeros((len(list(X)),), dtype=np.float64)  # type: ignore[method-assign]
    detector._process_decision_scores = lambda: None  # type: ignore[method-assign]

    simplenet_module.VisionSimpleNet.fit(detector, [str(image_path)])

    assert observed["weight_decay"] == pytest.approx(0.0)


def test_regad_fit_passes_explicit_weight_decay(monkeypatch) -> None:
    import torch

    import pyimgano.models.regad as regad_module

    observed: dict[str, float | None] = {}
    original_adam = regad_module.torch.optim.Adam

    def _recording_adam(*args, **kwargs):
        observed["weight_decay"] = kwargs.get("weight_decay")
        return original_adam(*args, **kwargs)

    class _FakeRegNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stn = torch.nn.Conv2d(1, 1, kernel_size=1)

        def forward(self, x: torch.Tensor):
            registered = x[:, :1, :, :]
            return registered, registered

    monkeypatch.setattr(regad_module.torch.optim, "Adam", _recording_adam)

    detector = regad_module.VisionRegAD(device="cpu", epochs=0, batch_size=1)
    detector.reg_network_ = _FakeRegNet()
    detector._preprocess = lambda X: torch.zeros((len(X), 3, 4, 4), dtype=torch.float32)  # type: ignore[method-assign]

    detector.fit(np.zeros((2, 16, 16, 3), dtype=np.uint8))

    assert observed["weight_decay"] == pytest.approx(0.0)


def test_oneformore_fit_passes_explicit_weight_decay(monkeypatch) -> None:
    import torch

    import pyimgano.models.oneformore as oneformore_module

    observed: dict[str, float | None] = {}
    original_adam = oneformore_module.torch.optim.Adam

    def _recording_adam(*args, **kwargs):
        observed["weight_decay"] = kwargs.get("weight_decay")
        return original_adam(*args, **kwargs)

    class _FakeFeatureExtractor(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros((x.size(0), 4, 4, 4), dtype=torch.float32, device=x.device)

    class _FakeDiffusion(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            del t
            return x + self.bias.view(1, 1, 1, 1)

    monkeypatch.setattr(oneformore_module.torch.optim, "Adam", _recording_adam)

    detector = oneformore_module.VisionOneForMore(
        device="cpu",
        epochs=0,
        batch_size=1,
        use_gradient_projection=False,
    )
    detector.feature_extractor_ = _FakeFeatureExtractor()
    detector.diffusion_model_ = _FakeDiffusion()
    detector._preprocess = lambda X: torch.zeros((len(X), 3, 8, 8), dtype=torch.float32)  # type: ignore[method-assign]

    detector.fit(np.zeros((2, 16, 16, 3), dtype=np.uint8))

    assert observed["weight_decay"] == pytest.approx(0.0)
