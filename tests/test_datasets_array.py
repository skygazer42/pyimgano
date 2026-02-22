import numpy as np

from pyimgano.datasets.array import VisionArrayDataset


def test_array_dataset_returns_tensor_pair():
    imgs = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(2)]
    ds = VisionArrayDataset(images=imgs)
    x, y = ds[0]
    assert x.shape == y.shape
