from __future__ import annotations

from pathlib import Path

import numpy as np


def test_gpu_e2e_workflow_is_scheduled_manual_and_pinned() -> None:
    workflow = Path(".github/workflows/gpu-e2e.yml").read_text(encoding="utf-8")

    assert "workflow_dispatch:" in workflow
    assert "schedule:" in workflow
    assert "runs-on: [self-hosted, linux, x64, gpu]" in workflow
    assert "constraints/optional-py310-current.txt" in workflow
    assert "tools/run_official_gpu_e2e.py" in workflow
    assert "--allow-download" in workflow


def test_gpu_e2e_map_adapter_supports_batch_and_single_image_apis() -> None:
    from tools.run_official_gpu_e2e import _predict_maps

    inputs = [np.zeros((2, 3), dtype=np.uint8), np.ones((2, 3), dtype=np.uint8)]

    class _BatchDetector:
        def predict_anomaly_map(self, images):  # noqa: ANN001 - protocol stub
            return np.stack(images, axis=0)

    class _SingleDetector:
        def get_anomaly_map(self, image):  # noqa: ANN001 - protocol stub
            return image

    expected = np.stack(inputs, axis=0).astype(np.float32)
    np.testing.assert_array_equal(_predict_maps(_BatchDetector(), inputs), expected)
    np.testing.assert_array_equal(_predict_maps(_SingleDetector(), inputs), expected)
