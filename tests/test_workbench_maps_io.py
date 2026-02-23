import numpy as np

from pyimgano.workbench.maps import save_anomaly_map_npy


def test_save_anomaly_map_npy_writes_file(tmp_path):
    out_dir = tmp_path / "run"
    anomaly_map = np.arange(6, dtype=np.float32).reshape(2, 3)

    out_path = save_anomaly_map_npy(
        out_dir,
        index=12,
        input_path="/a/b/c.png",
        anomaly_map=anomaly_map,
    )

    assert out_path.exists()
    assert out_path.name == "000012_c.npy"
    loaded = np.load(out_path)
    assert loaded.shape == (2, 3)
    assert np.allclose(loaded, anomaly_map)


def test_save_anomaly_map_npy_sanitizes_stem(tmp_path):
    out_dir = tmp_path / "run"
    anomaly_map = np.zeros((1, 1), dtype=np.float32)

    out_path = save_anomaly_map_npy(
        out_dir,
        index=1,
        input_path="weird name (x)/a b.png",
        anomaly_map=anomaly_map,
    )
    assert out_path.name == "000001_a_b.npy"

