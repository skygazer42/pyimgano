from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def test_winclip_contract_fit_and_decision_function_with_fake_backend(monkeypatch) -> None:
    import pyimgano.models.winclip as winclip_module

    class FakeClipModel:
        def eval(self):
            return self

        def encode_image(self, tensor: torch.Tensor) -> torch.Tensor:
            flat = tensor.reshape(tensor.size(0), -1).float()
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

    rng = np.random.default_rng(40)
    train = rng.integers(0, 255, size=(4, 32, 32, 3), dtype=np.uint8)
    test = rng.integers(0, 255, size=(2, 32, 32, 3), dtype=np.uint8)

    det = winclip_module.WinCLIPDetector(k_shot=2, device="cpu", random_state=0)
    det.fit(train)

    scores = np.asarray(det.decision_function(test), dtype=np.float64).reshape(-1)
    assert scores.shape == (2,)
    assert np.all(np.isfinite(scores))
