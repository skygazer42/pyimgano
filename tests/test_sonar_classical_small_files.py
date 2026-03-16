from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _module_tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _stored_names(path: Path) -> set[str]:
    tree = _module_tree(path)
    names: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)

    return names


def _method_arg_names(path: Path, class_name: str, method_name: str) -> list[str]:
    tree = _module_tree(path)

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    return [arg.arg for arg in child.args.args]

    raise AssertionError(f"Method {class_name}.{method_name} not found in {path}")


@pytest.mark.parametrize(
    ("relative_path", "class_name"),
    [
        pytest.param("pyimgano/models/kde.py", "CoreKDE", id="core-kde-fit"),
        pytest.param("pyimgano/models/qmcd.py", "CoreQMCD", id="core-qmcd-fit"),
        pytest.param("pyimgano/models/inne.py", "CoreINNE", id="core-inne-fit"),
        pytest.param("pyimgano/models/sampling.py", "CoreSampling", id="core-sampling-fit"),
        pytest.param(
            "pyimgano/models/core_cosine_mahalanobis.py",
            "_CosineMahalanobisBackend",
            id="cosine-mahalanobis-backend-fit",
        ),
        pytest.param(
            "pyimgano/models/core_mahalanobis_shrinkage.py",
            "_MahalanobisShrinkageBackend",
            id="mahalanobis-shrinkage-backend-fit",
        ),
        pytest.param("pyimgano/models/dbscan.py", "CoreDBSCAN", id="core-dbscan-fit"),
        pytest.param(
            "pyimgano/models/random_projection_knn.py",
            "_RPkNNBackend",
            id="random-projection-knn-backend-fit",
        ),
        pytest.param("pyimgano/models/ocsvm.py", "CoreOCSVM", id="core-ocsvm-fit"),
        pytest.param("pyimgano/models/aaclip.py", "VisionAAClip", id="vision-aaclip-fit"),
        pytest.param(
            "pyimgano/models/anomalib_backend.py",
            "VisionAnomalibCheckpoint",
            id="vision-anomalib-checkpoint-fit",
        ),
        pytest.param("pyimgano/models/filopp.py", "VisionFiLoPP", id="vision-filopp-fit"),
        pytest.param("pyimgano/models/hbos.py", "CoreHBOS", id="core-hbos-fit"),
        pytest.param("pyimgano/models/imdd.py", "CoreIMDD", id="core-imdd-fit"),
        pytest.param("pyimgano/models/k_means.py", "CoreKMeans", id="core-kmeans-fit"),
        pytest.param("pyimgano/models/loci.py", "CoreLOCI", id="core-loci-fit"),
        pytest.param("pyimgano/models/logsad.py", "VisionLogSAD", id="vision-logsad-fit"),
        pytest.param("pyimgano/models/mcd.py", "CoreMCD", id="core-mcd-fit"),
        pytest.param(
            "pyimgano/models/patch_embedding_core_map.py",
            "VisionPatchEmbeddingCoreMap",
            id="vision-patch-embedding-core-map-fit",
        ),
        pytest.param(
            "pyimgano/models/score_ensemble.py",
            "VisionScoreEnsemble",
            id="vision-score-ensemble-fit",
        ),
        pytest.param("pyimgano/models/suod.py", "CoreSUOD", id="core-suod-fit"),
        pytest.param(
            "tests/dummy_pyimgano_plugin.py",
            "PluginDummyModel",
            id="dummy-plugin-model-fit",
        ),
    ],
)
def test_fit_signatures_use_private_unused_target_param(
    relative_path: str,
    class_name: str,
) -> None:
    params = _method_arg_names(ROOT / relative_path, class_name, "fit")
    assert "_y" in params
    assert "y" not in params


@pytest.mark.parametrize(
    ("relative_path", "legacy_name"),
    [
        pytest.param("pyimgano/models/mad.py", "X_arr", id="mad-no-x-arr"),
        pytest.param("pyimgano/models/qmcd.py", "X_norm", id="qmcd-no-x-norm"),
        pytest.param("pyimgano/models/dtc.py", "X_arr", id="dtc-no-x-arr"),
        pytest.param(
            "pyimgano/models/elliptic_envelope.py",
            "X_arr",
            id="elliptic-envelope-no-x-arr",
        ),
        pytest.param("pyimgano/models/hst.py", "X_arr", id="hst-no-x-arr"),
        pytest.param("pyimgano/models/knn_degree.py", "X_arr", id="knn-degree-no-x-arr"),
        pytest.param("pyimgano/models/ldof.py", "X_arr", id="ldof-no-x-arr"),
        pytest.param("pyimgano/models/loop.py", "X_arr", id="loop-no-x-arr"),
        pytest.param("pyimgano/models/mahalanobis.py", "X_arr", id="mahalanobis-no-x-arr"),
        pytest.param("pyimgano/models/odin.py", "X_arr", id="odin-no-x-arr"),
        pytest.param(
            "pyimgano/models/core_cosine_mahalanobis.py",
            "X_arr",
            id="cosine-mahalanobis-no-x-arr",
        ),
        pytest.param(
            "pyimgano/models/core_mahalanobis_shrinkage.py",
            "X_arr",
            id="mahalanobis-shrinkage-no-x-arr",
        ),
        pytest.param("pyimgano/models/dbscan.py", "X_proc", id="dbscan-no-x-proc"),
        pytest.param(
            "pyimgano/models/random_projection_knn.py",
            "X_arr",
            id="random-projection-knn-no-x-arr",
        ),
        pytest.param("pyimgano/models/ocsvm.py", "X_eval", id="ocsvm-no-x-eval"),
    ],
)
def test_small_classical_modules_avoid_legacy_uppercase_local_names(
    relative_path: str,
    legacy_name: str,
) -> None:
    names = _stored_names(ROOT / relative_path)
    assert legacy_name not in names
