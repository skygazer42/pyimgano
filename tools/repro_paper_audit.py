#!/usr/bin/env python3
"""Deterministic, offline regression checks for the 2026-08-30 paper-code audit.

This is not a benchmark runner.  It exercises small numerical invariants that
were violated by the audited baseline and now guard the repaired formulas and
fitted-detector API contracts.  It performs no network access and does not load
pretrained weights.
"""

from __future__ import annotations

import json
import platform
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from pyimgano.models.extra_trees_density import CoreExtraTreesDensity
from pyimgano.models.hbos import CoreHBOS
from pyimgano.models.imdd import CoreIMDD
from pyimgano.models.lid import _lid_from_knn_distances
from pyimgano.models.loci import CoreLOCI
from pyimgano.models.oneformore import _paper_image_ap_scores
from pyimgano.models.openclip_backend import _run_openclip_transformer
from pyimgano.models.pca import CorePCA
from pyimgano.models.qmcd import CoreQMCD
from pyimgano.models.sos import CoreSOS


def _version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "not-installed"


def _as_float(value: Any) -> float:
    return float(np.asarray(value, dtype=np.float64))


def reproduce() -> dict[str, Any]:
    rng6 = np.random.default_rng(0)
    train6 = rng6.normal(size=(40, 6))
    rng2 = np.random.default_rng(0)
    train2 = rng2.normal(size=(39, 2))
    query2 = np.asarray([[7.0, 7.0]], dtype=np.float64)

    distances = np.asarray([[1.0, 2.0, 4.0]], dtype=np.float64)
    lid_local = _as_float(_lid_from_knn_distances(distances, eps=1e-12)[0])
    lid_mle = _as_float(-1.0 / np.mean(np.log(distances / distances[:, [-1]])))

    loci = CoreLOCI().fit(train2)
    sos = CoreSOS().fit(train2)
    imdd_single_model = CoreIMDD(random_state=0, n_iter=10).fit(train6[:20])
    imdd_context_model = CoreIMDD(random_state=0, n_iter=10).fit(train6[:20])
    context2 = np.vstack((query2, train2))
    imdd_context = np.vstack((np.full((1, 6), 7.0), train6[:39]))

    imdd_repeat = CoreIMDD(random_state=0, n_iter=10).fit(train6[:20])
    imdd_first = imdd_repeat.decision_function(train6[:20])
    imdd_second = imdd_repeat.decision_function(train6[:20])

    extra = CoreExtraTreesDensity(n_estimators=10, random_state=0).fit(train6)
    zero6 = np.zeros((1, 6), dtype=np.float64)
    extra_single = _as_float(extra.decision_function(zero6)[0])
    extra_repeated = _as_float(extra.decision_function(np.repeat(zero6, 10, axis=0))[0])

    train5 = np.random.default_rng(0).normal(size=(40, 5))
    pca_default = CorePCA().fit(train5)
    pca_one = CorePCA(n_selected_components=1).fit(train5)

    qmcd = CoreQMCD().fit(train6)
    qmcd_queries = np.vstack(
        (
            np.zeros((1, 6), dtype=np.float64),
            np.full((1, 6), 12.0, dtype=np.float64),
            np.full((1, 6), -12.0, dtype=np.float64),
        )
    )
    qmcd_scores = qmcd.decision_function(qmcd_queries)

    hbos_train = np.concatenate((np.zeros(10), np.ones(90))).reshape(-1, 1)
    hbos = CoreHBOS(n_bins=2).fit(hbos_train)
    hbos_scores = hbos.decision_function(np.asarray([[0.0], [1.0], [100.0]]))

    import torch
    import torch.nn.functional as torch_f

    torch.manual_seed(0)
    transformer = torch.nn.TransformerEncoderLayer(
        d_model=4,
        nhead=2,
        dim_feedforward=8,
        dropout=0.0,
        batch_first=False,
    ).eval()
    tokens = torch.randn(1, 5, 4)
    with torch.no_grad():
        openclip_local = _run_openclip_transformer(transformer, tokens)
        openclip_lnb = transformer(tokens.permute(1, 0, 2)).permute(1, 0, 2)
    openclip_delta = float(torch.max(torch.abs(openclip_local - openclip_lnb)).item())

    point_map = torch.zeros((1, 1, 256, 256), dtype=torch.float32)
    point_map[:, :, 128, 128] = 1.0
    block_map = torch.zeros_like(point_map)
    block_map[:, :, 96:160, 96:160] = 1.0

    def ap_smoothing_max(anomaly_map: Any) -> float:
        value = anomaly_map
        for _ in range(8):
            value = torch_f.avg_pool2d(value, kernel_size=8, stride=1)
        return float(value.max().item())

    local_ap_scores = _paper_image_ap_scores(
        np.concatenate((point_map.numpy(), block_map.numpy()), axis=0)[:, 0]
    )

    result = {
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": _version("scipy"),
            "scikit_learn": _version("scikit-learn"),
            "torch": _version("torch"),
        },
        "H01_openclip_layout": {
            "seed": 0,
            "input_shape": [1, 5, 4],
            "max_abs_diff_from_explicit_lnb": openclip_delta,
        },
        "H02_lid": {"local": lid_local, "paper_mle": lid_mle},
        "H03_query_context": {
            "loci_single": _as_float(loci.decision_function(query2)[0]),
            "loci_with_39": _as_float(loci.decision_function(context2)[0]),
            "sos_single": _as_float(sos.decision_function(query2)[0]),
            "sos_with_39": _as_float(sos.decision_function(context2)[0]),
            "imdd_single": _as_float(
                imdd_single_model.decision_function(np.full((1, 6), 7.0))[0]
            ),
            "imdd_with_39": _as_float(imdd_context_model.decision_function(imdd_context)[0]),
            "imdd_repeat_max_abs_diff": _as_float(
                np.max(np.abs(imdd_first - imdd_second))
            ),
        },
        "H04_extra_trees": {
            "single": extra_single,
            "repeated_10_first": extra_repeated,
            "delta": extra_repeated - extra_single,
            "log_10": float(np.log(10.0)),
        },
        "H05_pca": {
            "default_train_max": _as_float(np.max(pca_default.decision_scores_)),
            "one_component_train_max": _as_float(np.max(pca_one.decision_scores_)),
        },
        "H06_qmcd": {
            "center": _as_float(qmcd_scores[0]),
            "far_positive": _as_float(qmcd_scores[1]),
            "far_negative": _as_float(qmcd_scores[2]),
        },
        "H08_one_for_more_score_stage": {
            "raw_point_max": float(point_map.max().item()),
            "raw_block_max": float(block_map.max().item()),
            "local_ap_point_max": _as_float(local_ap_scores[0]),
            "local_ap_block_max": _as_float(local_ap_scores[1]),
            "reference_ap_point_max": ap_smoothing_max(point_map),
            "reference_ap_block_max": ap_smoothing_max(block_map),
        },
        "M16_hbos": {
            "score_0": _as_float(hbos_scores[0]),
            "score_1": _as_float(hbos_scores[1]),
            "score_100": _as_float(hbos_scores[2]),
        },
    }

    context = result["H03_query_context"]
    assert np.isclose(lid_local, lid_mle, rtol=1e-12, atol=1e-12)
    assert openclip_delta <= 1e-7
    assert np.isclose(context["loci_single"], context["loci_with_39"])
    assert np.isclose(context["sos_single"], context["sos_with_39"])
    assert np.isclose(context["imdd_single"], context["imdd_with_39"])
    assert context["imdd_repeat_max_abs_diff"] <= 1e-12
    assert np.isclose(extra_single, extra_repeated)
    assert result["H05_pca"]["default_train_max"] > 1e-6
    assert result["H06_qmcd"]["far_positive"] > result["H06_qmcd"]["center"]
    assert result["H06_qmcd"]["far_negative"] > result["H06_qmcd"]["center"]
    assert np.allclose(
        local_ap_scores,
        [ap_smoothing_max(point_map), ap_smoothing_max(block_map)],
    )
    assert result["M16_hbos"]["score_100"] > result["M16_hbos"]["score_1"]
    return result


def main() -> None:
    print(json.dumps(reproduce(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
