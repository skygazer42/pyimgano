from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
import torch

_GLOBAL_NUMPY_RNG = np.random.mtrand._rand


def _global_numpy_state_after(action):
    state = _GLOBAL_NUMPY_RNG.get_state()
    try:
        action()
        return _GLOBAL_NUMPY_RNG.get_state()
    finally:
        _GLOBAL_NUMPY_RNG.set_state(state)


def _run_after_advancing_global_numpy(draws: int, action):
    state = _GLOBAL_NUMPY_RNG.get_state()
    try:
        _GLOBAL_NUMPY_RNG.bytes(draws)
        return action()
    finally:
        _GLOBAL_NUMPY_RNG.set_state(state)


def _numpy_states_equal(left, right) -> bool:
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


@pytest.mark.parametrize(
    ("factory"),
    [
        lambda: __import__("pyimgano.models.deep_svdd", fromlist=["CoreDeepSVDD"]).CoreDeepSVDD(
            n_features=4,
            hidden_neurons=[8, 4],
            epochs=1,
            batch_size=2,
            verbose=0,
            random_state=123,
        ),
        lambda: __import__(
            "pyimgano.models.bayesianpf",
            fromlist=["VisionBayesianPF"],
        ).VisionBayesianPF(device="cpu", random_state=123),
        lambda: __import__("pyimgano.models.dst", fromlist=["VisionDST"]).VisionDST(
            device="cpu",
            random_state=123,
        ),
        lambda: __import__("pyimgano.models.favae", fromlist=["VisionFAVAE"]).VisionFAVAE(
            device="cpu",
            random_state=123,
        ),
        lambda: __import__("pyimgano.models.gcad", fromlist=["VisionGCAD"]).VisionGCAD(
            device="cpu",
            random_state=123,
        ),
        lambda: __import__("pyimgano.models.glad", fromlist=["VisionGLAD"]).VisionGLAD(
            device="cpu",
            random_state=123,
        ),
        lambda: __import__("pyimgano.models.inctrl", fromlist=["VisionInCTRL"]).VisionInCTRL(
            device="cpu",
            random_state=123,
        ),
        lambda: __import__(
            "pyimgano.models.oneformore",
            fromlist=["VisionOneForMore"],
        ).VisionOneForMore(device="cpu", random_state=123),
        lambda: __import__("pyimgano.models.panda", fromlist=["VisionPANDA"]).VisionPANDA(
            device="cpu",
            random_state=123,
        ),
        lambda: __import__(
            "pyimgano.models.promptad",
            fromlist=["VisionPromptAD"],
        ).VisionPromptAD(device="cpu", random_state=123),
        lambda: __import__("pyimgano.models.regad", fromlist=["VisionRegAD"]).VisionRegAD(
            device="cpu",
            random_state=123,
        ),
        lambda: __import__(
            "pyimgano.models.differnet",
            fromlist=["DifferNetDetector"],
        ).DifferNetDetector(
            backbone="resnet18",
            pretrained=False,
            train_difference=False,
            device="cpu",
            random_state=123,
        ),
        lambda: __import__("pyimgano.models.riad", fromlist=["RIADDetector"]).RIADDetector(
            image_size=(32, 32),
            epochs=1,
            batch_size=2,
            device="cpu",
            random_state=123,
        ),
        lambda: __import__(
            "pyimgano.models.template_matching",
            fromlist=["TemplateMatching"],
        ).TemplateMatching(
            max_templates=2,
            random_state=123,
        ),
    ],
)
def test_random_state_constructors_do_not_reset_global_numpy_rng(factory) -> None:
    expected = _global_numpy_state_after(lambda: None)
    observed = _global_numpy_state_after(factory)

    assert _numpy_states_equal(observed, expected)


def test_template_matching_random_state_makes_template_sampling_repeatable() -> None:
    from pyimgano.models.template_matching import TemplateMatching

    x = np.arange(6 * 4 * 4, dtype=np.float32).reshape(6, 4, 4) / 255.0

    model_a = TemplateMatching(max_templates=3, random_state=7)
    model_b = TemplateMatching(max_templates=3, random_state=7)

    _run_after_advancing_global_numpy(3, lambda: model_a.fit(x))
    _run_after_advancing_global_numpy(11, lambda: model_b.fit(x))

    assert np.array_equal(model_a.templates_, model_b.templates_)


def test_image_decomposer_random_state_makes_masks_repeatable() -> None:
    from pyimgano.models.riad import ImageDecomposer

    image = np.ones((8, 8, 3), dtype=np.float32)

    decomposer_a = ImageDecomposer(n_splits=4, mask_ratio=0.5, random_state=5)
    decomposer_b = ImageDecomposer(n_splits=4, mask_ratio=0.5, random_state=5)

    _, mask_a, _ = decomposer_a.decompose(image)
    _, mask_b, _ = decomposer_b.decompose(image)

    assert np.array_equal(mask_a, mask_b)


def test_add_gaussian_noise_random_state_makes_noise_repeatable() -> None:
    from pyimgano.utils.image_ops_cv import add_gaussian_noise

    image = np.full((8, 8, 3), 100, dtype=np.uint8)

    out_a = _run_after_advancing_global_numpy(
        5, lambda: add_gaussian_noise(image, sigma=5.0, random_state=7)
    )
    out_b = _run_after_advancing_global_numpy(
        17, lambda: add_gaussian_noise(image, sigma=5.0, random_state=7)
    )

    assert np.array_equal(out_a, out_b)


def test_data_loader_random_state_makes_shuffle_repeatable() -> None:
    from pyimgano.utils.data_pipeline import DataLoader, Dataset

    class RangeDataset(Dataset):
        def __len__(self) -> int:
            return 6

        def __getitem__(self, index: int) -> int:
            return index

    loader_a = DataLoader(RangeDataset(), batch_size=2, shuffle=True, random_state=7)
    loader_b = DataLoader(RangeDataset(), batch_size=2, shuffle=True, random_state=7)

    order_a = _run_after_advancing_global_numpy(
        7, lambda: [int(item) for batch in loader_a for item in np.asarray(batch).tolist()]
    )
    order_b = _run_after_advancing_global_numpy(
        23, lambda: [int(item) for batch in loader_b for item in np.asarray(batch).tolist()]
    )

    assert order_a == order_b


def test_random_horizontal_flip_random_state_makes_augmentation_repeatable() -> None:
    from pyimgano.utils.data_pipeline import RandomHorizontalFlip

    image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    flip_a = RandomHorizontalFlip(p=0.5, random_state=11)
    flip_b = RandomHorizontalFlip(p=0.5, random_state=11)

    out_a = _run_after_advancing_global_numpy(13, lambda: flip_a(image))
    out_b = _run_after_advancing_global_numpy(29, lambda: flip_b(image))

    assert np.array_equal(out_a, out_b)


def test_patchcore_online_bank_capping_avoids_legacy_random_state(monkeypatch) -> None:
    import pyimgano.models.patchcore_online as patchcore_online

    x_train = np.arange(20, dtype=np.float64).reshape(10, 2)
    called: list[int] = []
    original_default_rng = patchcore_online.np.random.default_rng

    def _tracking_default_rng(seed=None):
        called.append(0 if seed is None else int(seed))
        return original_default_rng(seed)

    monkeypatch.setattr(patchcore_online.np.random, "default_rng", _tracking_default_rng)

    detector = patchcore_online.CorePatchCoreOnline(max_bank_size=4, random_state=7)
    detector.fit(x_train)

    assert called == [7]
    assert detector.memory_bank_ is not None
    assert detector.memory_bank_.shape == (4, 2)


def test_simplenet_reference_sampling_uses_default_rng(tmp_path, monkeypatch) -> None:
    import cv2

    import pyimgano.models.simplenet as simplenet_module

    image = np.full((8, 8, 3), 127, dtype=np.uint8)
    image_path = tmp_path / "normal.png"
    assert cv2.imwrite(str(image_path), image)

    detector = simplenet_module.VisionSimpleNet.__new__(simplenet_module.VisionSimpleNet)
    detector.adapter = torch.nn.Identity()
    detector.device = "cpu"
    detector.random_state = 11
    detector.transform = lambda img: torch.from_numpy(img).permute(2, 0, 1).float()
    detector._extract_features = lambda img_tensor: torch.ones((1, 4, 33, 33), dtype=torch.float32)  # type: ignore[method-assign]

    called: list[int] = []
    original_default_rng = simplenet_module.np.random.default_rng

    def _tracking_default_rng(seed=None):
        called.append(0 if seed is None else int(seed))
        return original_default_rng(seed)

    monkeypatch.setattr(simplenet_module.np.random, "default_rng", _tracking_default_rng)

    detector._build_reference_features([str(image_path)])

    assert called == [11]


def test_patchcore_lite_coreset_sampling_uses_default_rng(monkeypatch) -> None:
    import pyimgano.models.patchcore_lite as patchcore_lite_module

    x_train = np.arange(40, dtype=np.float64).reshape(20, 2)
    called: list[int] = []
    original_default_rng = patchcore_lite_module.np.random.default_rng

    def _tracking_default_rng(seed=None):
        called.append(0 if seed is None else int(seed))
        return original_default_rng(seed)

    monkeypatch.setattr(patchcore_lite_module.np.random, "default_rng", _tracking_default_rng)

    detector = patchcore_lite_module.CorePatchCoreLite(coreset_ratio=0.25, random_state=7)
    detector.fit(x_train)

    assert called == [7]
    assert detector.memory_bank_ is not None
    assert detector.memory_bank_.shape == (5, 2)


def test_feature_bagging_subspace_sampling_uses_default_rng(monkeypatch) -> None:
    import pyimgano.models.feature_bagging as feature_bagging_module

    x_train = np.arange(96, dtype=np.float64).reshape(12, 8)
    called: list[int] = []
    original_default_rng = feature_bagging_module.np.random.default_rng

    def _tracking_default_rng(seed=None):
        called.append(0 if seed is None else int(seed))
        return original_default_rng(seed)

    monkeypatch.setattr(feature_bagging_module.np.random, "default_rng", _tracking_default_rng)

    detector = feature_bagging_module.CoreFeatureBagging(
        n_estimators=3,
        max_features=0.75,
        random_state=7,
        n_neighbors=2,
    )
    detector.fit(x_train)

    assert len(called) == 3
    assert detector.estimators_features_


def test_winclip_random_state_makes_few_shot_sampling_repeatable(monkeypatch) -> None:
    import pyimgano.models.winclip as winclip_module

    class FakeClipModel:
        def eval(self) -> None:
            return None

        def encode_image(self, tensor: torch.Tensor) -> torch.Tensor:
            flat = tensor.reshape(tensor.size(0), -1)
            return flat.mean(dim=1, keepdim=True)

        def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
            return tokens.float()

    class FakeClipBackend:
        def load(self, clip_model: str, device=None):
            del clip_model, device

            def _preprocess(pil_img):
                array = np.asarray(pil_img, dtype=np.float32)
                return torch.from_numpy(array).permute(2, 0, 1)

            return FakeClipModel(), _preprocess

        def tokenize(self, prompts):
            return torch.ones((len(prompts), 1), dtype=torch.float32)

    monkeypatch.setattr(winclip_module, "require", lambda *args, **kwargs: FakeClipBackend())

    x = np.arange(6 * 8 * 8 * 3, dtype=np.uint8).reshape(6, 8, 8, 3)

    model_a = winclip_module.WinCLIPDetector(k_shot=3, device="cpu", random_state=5)
    model_b = winclip_module.WinCLIPDetector(k_shot=3, device="cpu", random_state=5)

    _run_after_advancing_global_numpy(9, lambda: model_a.fit(x))
    _run_after_advancing_global_numpy(27, lambda: model_b.fit(x))

    assert torch.equal(model_a.few_shot_features, model_b.few_shot_features)


def test_core_torch_autoencoder_fit_does_not_reset_global_numpy_rng() -> None:
    from pyimgano.models.registry import create_model

    x = np.random.default_rng(0).normal(size=(24, 6)).astype(np.float64)
    det = create_model(
        "core_torch_autoencoder",
        contamination=0.1,
        hidden_dims=(8, 4),
        epochs=1,
        batch_size=8,
        lr=1e-3,
        device="cpu",
        preprocessing=True,
        random_state=123,
    )

    expected = _global_numpy_state_after(lambda: None)
    observed = _global_numpy_state_after(lambda: det.fit(x))

    assert _numpy_states_equal(observed, expected)


def test_mambaad_fit_does_not_reset_global_numpy_rng(monkeypatch) -> None:
    from pyimgano.models.mambaad import VisionMambaAD

    class DummyEmbedder:
        def embed(self, image):
            del image
            return np.zeros((4, 8), dtype=np.float32), (2, 2), (32, 32)

    class DummyReconstructor(torch.nn.Module):
        def __init__(self, d_model: int) -> None:
            super().__init__()
            self.proj = torch.nn.Linear(d_model, d_model)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.proj(x)

    det = VisionMambaAD(
        embedder=DummyEmbedder(),
        device="cpu",
        epochs=1,
        batch_size=1,
        lr=1e-3,
        noise_std=0.0,
        random_seed=123,
    )

    def fake_ensure_model(*, d_model: int) -> None:
        det._d_model = int(d_model)
        det._reconstructor = DummyReconstructor(int(d_model)).to(det.device)

    monkeypatch.setattr(det, "_ensure_model", fake_ensure_model)
    monkeypatch.setattr(
        det,
        "decision_function",
        lambda x, batch_size=None: np.zeros((len(x),), dtype=np.float32),
    )

    expected = _global_numpy_state_after(lambda: None)
    observed = _global_numpy_state_after(
        lambda: det.fit([np.zeros((32, 32, 3), dtype=np.uint8)])
    )

    assert _numpy_states_equal(observed, expected)
