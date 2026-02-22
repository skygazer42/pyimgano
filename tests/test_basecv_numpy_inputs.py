import numpy as np

import pyimgano.models.baseCv as baseCv
from pyimgano.datasets.array import VisionArrayDataset


def test_basecv_accepts_numpy_list(monkeypatch):
    class DummyDeep(baseCv.BaseVisionDeepDetector):
        def build_model(self):
            import torch.nn as nn

            self.model = nn.Identity()
            return self.model

    det = DummyDeep(epoch_num=1, batch_size=1, verbose=0, device="cpu")

    seen: dict[str, object] = {}

    monkeypatch.setattr(det, "training_prepare", lambda: None)

    def fake_train(loader):
        seen["train_dataset_type"] = type(loader.dataset)

    def fake_evaluate(loader):
        seen["eval_dataset_type"] = type(loader.dataset)
        return np.zeros((len(loader.dataset),), dtype=np.float32)

    monkeypatch.setattr(det, "train", fake_train)
    monkeypatch.setattr(det, "evaluate", fake_evaluate)

    imgs = [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(2)]
    det.fit(imgs)
    scores = det.decision_function(imgs)

    assert scores.shape == (2,)
    assert seen["train_dataset_type"] is VisionArrayDataset
    assert seen["eval_dataset_type"] is VisionArrayDataset
