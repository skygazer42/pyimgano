from __future__ import annotations

import numpy as np


class _DummyDetector:
    def fit(self, X, y=None):  # noqa: ANN001, ANN201
        _ = (X, y)
        return self

    def decision_function(self, X):  # noqa: ANN001, ANN201
        # Constant scores are fine; AUROC should become defined once labels
        # contain both 0 and 1.
        return np.zeros((len(list(X)),), dtype=np.float64)


class _SynthesisLikeCorruption:
    name = "synthesis_like"

    def __init__(self) -> None:
        self._i = 0

    def __call__(
        self, image, mask, *, severity: int, rng: np.random.Generator
    ):  # noqa: ANN001, ANN201
        _ = (mask, severity)
        self._i += 1

        h, w = int(image.shape[0]), int(image.shape[1])
        if self._i % 2 == 0:
            # No anomaly for even indices.
            return np.asarray(image, dtype=np.uint8), np.zeros((h, w), dtype=np.uint8)

        from pyimgano.synthesis.synthesizer import AnomalySynthesizer, SynthSpec

        syn = AnomalySynthesizer(
            SynthSpec(preset="scratch", probability=1.0, blend="alpha", alpha=0.8)
        )
        res = syn(np.asarray(image, dtype=np.uint8), rng=rng)
        return res.image_u8, res.mask_u8


def test_robustness_benchmark_can_use_synthesis_masks_to_update_labels() -> None:
    from pyimgano.robustness.benchmark import run_robustness_benchmark

    det = _DummyDetector()
    train = [np.full((32, 32, 3), 20, dtype=np.uint8) for _ in range(4)]
    test = [np.full((32, 32, 3), 30, dtype=np.uint8) for _ in range(4)]

    # Start with all-normal labels; clean AUROC is undefined (nan).
    labels = np.zeros((len(test),), dtype=np.int64)

    report = run_robustness_benchmark(
        det,
        train_images=train,
        test_images=test,
        test_labels=labels,
        test_masks=None,
        corruptions=[_SynthesisLikeCorruption()],
        severities=(1,),
        seed=0,
        pixel_segf1=False,
    )

    clean_auroc = float(report["clean"]["results"]["auroc"])
    assert np.isnan(clean_auroc)

    corr = report["corruptions"]["synthesis_like"]["severity_1"]["results"]
    corr_auroc = float(corr["auroc"])
    # After mask-derived label update, AUROC becomes defined (not nan).
    assert not np.isnan(corr_auroc)
