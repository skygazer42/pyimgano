import numpy as np


def _write_png(path, *, value: int = 128) -> None:
    from PIL import Image

    img = np.ones((16, 16, 3), dtype=np.uint8) * int(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(str(path))


def test_run_benchmark_includes_timing_breakdown(tmp_path) -> None:
    import pyimgano.models  # noqa: F401 - registry population

    from pyimgano.pipelines.run_benchmark import run_benchmark

    root = tmp_path / "custom_ds"
    _write_png(root / "train" / "normal" / "train_0.png", value=120)
    _write_png(root / "test" / "normal" / "good_0.png", value=120)
    _write_png(root / "test" / "anomaly" / "bad_0.png", value=240)

    payload = run_benchmark(
        dataset="custom",
        root=str(root),
        category="custom",
        model="vision_ecod",
        device="cpu",
        pretrained=False,
        save_run=False,
        per_image_jsonl=False,
        limit_train=1,
        limit_test=2,
    )

    timing = payload.get("timing", None)
    assert isinstance(timing, dict)
    assert {"total_s", "fit_s", "score_test_s"}.issubset(set(timing))
    for value in timing.values():
        assert isinstance(value, float)
        assert value >= 0.0
