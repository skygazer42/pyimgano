from __future__ import annotations

import numpy as np

from pyimgano.inference.composite_runtime import CompositeArtifactRuntime


class _FeatureRuntime:
    runtime_info = {"backend": "onnxruntime", "providers": ["CPUExecutionProvider"]}

    def extract(self, inputs):  # noqa: ANN001
        return np.asarray([[float(np.asarray(item).mean())] for item in inputs])


class _Core:
    def decision_function(self, features):  # noqa: ANN001
        return np.asarray(features)[:, 0] * 2.0


def test_composite_runtime_applies_fitted_core_without_refit() -> None:
    runtime = CompositeArtifactRuntime(
        component_runtime=_FeatureRuntime(),
        fitted_core=_Core(),
        adapter_id="test-core-v1",
    )
    images = [np.full((2, 2, 3), 3, dtype=np.uint8), np.full((2, 2, 3), 7, dtype=np.uint8)]

    scores = runtime.decision_function(images)

    np.testing.assert_allclose(scores, [6.0, 14.0])
    assert runtime.runtime_info["selected_provider"] is None
    assert runtime.runtime_info["adapter_id"] == "test-core-v1"


def test_composite_runtime_prefers_explicit_registered_adapter() -> None:
    calls = {"count": 0}

    def adapter(*, component_runtime, fitted_core, inputs, include_maps):  # noqa: ANN001
        del component_runtime, fitted_core
        calls["count"] += 1
        return np.arange(len(inputs), dtype=np.float32), (
            np.zeros((len(inputs), 2, 2), dtype=np.float32) if include_maps else None
        )

    runtime = CompositeArtifactRuntime(
        component_runtime=object(),
        fitted_core=object(),
        adapter=adapter,
        adapter_id="registered-adapter-v1",
    )

    scores, maps = runtime.score_and_maps([object(), object()])

    np.testing.assert_array_equal(scores, [0.0, 1.0])
    assert maps is not None and maps.shape == (2, 2, 2)
    assert calls["count"] == 1
