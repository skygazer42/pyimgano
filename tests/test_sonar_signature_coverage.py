import inspect

import numpy as np
import pytest


def _make_instance_without_init(cls):
    # Many vision detectors pull optional heavy deps (e.g., torch) during __init__.
    # For these interface-level tests we bypass __init__ and only exercise the
    # early-guard branches added for SonarCloud signature compatibility.
    return cls.__new__(cls)


def test_predict_return_confidence_raises_without_optional_deps():
    from pyimgano.models.mambaad import VisionMambaAD
    from pyimgano.models.padim import VisionPaDiM
    from pyimgano.models.patchcore import VisionPatchCore

    for cls in (VisionMambaAD, VisionPaDiM, VisionPatchCore):
        inst = _make_instance_without_init(cls)
        with pytest.raises(NotImplementedError):
            cls.predict(inst, X=[], return_confidence=True)


def test_decision_function_rejects_non_positive_batch_size():
    from pyimgano.models.mambaad import VisionMambaAD
    from pyimgano.models.padim import VisionPaDiM
    from pyimgano.models.patchcore import VisionPatchCore

    for cls in (VisionMambaAD, VisionPaDiM, VisionPatchCore):
        inst = _make_instance_without_init(cls)
        with pytest.raises(ValueError):
            cls.decision_function(inst, X=[], batch_size=0)


def test_decision_function_accepts_positive_batch_size_on_empty_input():
    import numpy as np

    from pyimgano.models.mambaad import VisionMambaAD
    from pyimgano.models.padim import VisionPaDiM
    from pyimgano.models.patchcore import VisionPatchCore

    # VisionMambaAD: empty input should short-circuit before touching heavy deps.
    mamba = _make_instance_without_init(VisionMambaAD)
    scores = VisionMambaAD.decision_function(mamba, X=[], batch_size=1)
    assert isinstance(scores, np.ndarray)
    assert scores.shape == (0,)

    # VisionPaDiM: avoid heavy deps by bypassing __init__ and setting fitted markers.
    padim = _make_instance_without_init(VisionPaDiM)
    padim.means = np.zeros((1, 1), dtype=np.float32)
    padim.inv_covs = np.zeros((1, 1, 1), dtype=np.float32)
    padim.patch_shape = (1, 1)
    scores = VisionPaDiM.decision_function(padim, X=[], batch_size=1)
    assert isinstance(scores, np.ndarray)
    assert scores.shape == (0,)

    # VisionPatchCore: cover the post-guard path without fitting by triggering the
    # "not fitted" error after setting the minimal attributes it expects.
    patchcore = _make_instance_without_init(VisionPatchCore)
    patchcore._np = np
    patchcore.memory_bank = None
    patchcore.nn_index = None
    with pytest.raises(RuntimeError):
        VisionPatchCore.decision_function(patchcore, X=[], batch_size=1)


def test_workbench_lazy_exports_are_importable_and_actionable():
    import pyimgano.workbench as wb

    # Exercise module-level code paths (__dir__/__getattr__) for coverage on new code.
    names = dir(wb)
    assert "WorkbenchConfig" in names

    assert wb.WorkbenchConfig is not None

    with pytest.raises(AttributeError):
        getattr(wb, "definitely_not_exported")


def test_services_lazy_exports_are_importable_and_actionable():
    import pyimgano.services as services

    names = dir(services)
    assert "check_module" in names

    assert callable(services.check_module)

    with pytest.raises(AttributeError):
        getattr(services, "definitely_not_exported")


def _assert_x_signature(method) -> None:
    params = list(inspect.signature(method).parameters)
    assert "x" in params
    assert "X" not in params


def _assert_constructor_param_signature(target, *, expected: str, legacy: str) -> None:
    params = list(inspect.signature(target).parameters)
    assert expected in params
    assert legacy not in params


def _assert_signature_contains(method, *expected: str) -> None:
    params = list(inspect.signature(method).parameters)
    for name in expected:
        assert name in params


@pytest.mark.parametrize(
    ("cls", "method_names"),
    [
        pytest.param(
            __import__("pyimgano.models.anomalydino", fromlist=["VisionAnomalyDINO"]).VisionAnomalyDINO,
            ("fit", "decision_function", "predict", "predict_anomaly_map"),
            id="anomalydino",
        ),
        pytest.param(
            __import__("pyimgano.models.softpatch", fromlist=["VisionSoftPatch"]).VisionSoftPatch,
            ("fit", "decision_function", "predict", "predict_anomaly_map"),
            id="softpatch",
        ),
        pytest.param(
            __import__(
                "pyimgano.models.anomalib_backend", fromlist=["VisionAnomalibCheckpoint"]
            ).VisionAnomalibCheckpoint,
            ("fit", "decision_function", "predict", "predict_anomaly_map"),
            id="anomalib-backend",
        ),
        pytest.param(
            __import__("pyimgano.models.visionad", fromlist=["VisionVisionAD"]).VisionVisionAD,
            ("fit", "decision_function", "predict", "predict_anomaly_map"),
            id="visionad",
        ),
        pytest.param(
            __import__("pyimgano.models.filopp", fromlist=["VisionFiLoPP"]).VisionFiLoPP,
            ("fit", "decision_function", "predict", "predict_anomaly_map"),
            id="filopp",
        ),
        pytest.param(
            __import__("pyimgano.models.logsad", fromlist=["VisionLogSAD"]).VisionLogSAD,
            ("fit", "decision_function", "predict", "predict_anomaly_map"),
            id="logsad",
        ),
        pytest.param(
            __import__(
                "pyimgano.models.one_to_normal", fromlist=["VisionOneToNormal"]
            ).VisionOneToNormal,
            ("fit", "decision_function", "predict", "predict_anomaly_map"),
            id="one-to-normal",
        ),
        pytest.param(
            __import__("pyimgano.models.superad", fromlist=["VisionSuperAD"]).VisionSuperAD,
            ("fit", "decision_function", "predict", "predict_anomaly_map"),
            id="superad",
        ),
        pytest.param(
            __import__(
                "pyimgano.models.pixel_stats_map",
                fromlist=[
                    "_BasePixelStatsMapDetector",
                    "VisionPixelMeanAbsDiffMapDetector",
                    "VisionPixelGaussianMapDetector",
                    "VisionPixelMADMapDetector",
                ],
            )._BasePixelStatsMapDetector,
            ("decision_function", "predict_anomaly_map"),
            id="pixel-stats-base",
        ),
        pytest.param(
            __import__(
                "pyimgano.models.pixel_stats_map", fromlist=["VisionPixelMeanAbsDiffMapDetector"]
            ).VisionPixelMeanAbsDiffMapDetector,
            ("fit",),
            id="pixel-mean",
        ),
        pytest.param(
            __import__(
                "pyimgano.models.pixel_stats_map", fromlist=["VisionPixelGaussianMapDetector"]
            ).VisionPixelGaussianMapDetector,
            ("fit",),
            id="pixel-gaussian",
        ),
        pytest.param(
            __import__(
                "pyimgano.models.pixel_stats_map", fromlist=["VisionPixelMADMapDetector"]
            ).VisionPixelMADMapDetector,
            ("fit",),
            id="pixel-mad",
        ),
        pytest.param(
            __import__("pyimgano.models.aaclip", fromlist=["VisionAAClip"]).VisionAAClip,
            ("fit", "decision_function", "predict", "predict_anomaly_map"),
            id="aaclip",
        ),
        pytest.param(
            __import__("pyimgano.models.adaclip", fromlist=["VisionAdaCLIP"]).VisionAdaCLIP,
            ("fit", "decision_function", "predict"),
            id="adaclip",
        ),
        pytest.param(
            __import__(
                "pyimgano.models.openclip_patch_map", fromlist=["VisionOpenCLIPPatchMap"]
            ).VisionOpenCLIPPatchMap,
            ("fit", "decision_function", "predict", "predict_anomaly_map"),
            id="openclip-patch-map",
        ),
        pytest.param(
            __import__(
                "pyimgano.models.patch_embedding_core_map", fromlist=["VisionPatchEmbeddingCoreMap"]
            ).VisionPatchEmbeddingCoreMap,
            ("fit", "decision_function", "predict", "predict_anomaly_map"),
            id="patch-embedding-core-map",
        ),
        pytest.param(
            __import__(
                "pyimgano.models.patchcore_inspection_backend",
                fromlist=["VisionPatchCoreInspectionCheckpoint"],
            ).VisionPatchCoreInspectionCheckpoint,
            ("fit", "decision_function", "predict", "predict_anomaly_map"),
            id="patchcore-inspection-backend",
        ),
        pytest.param(
            __import__("pyimgano.models.univad", fromlist=["VisionUniVAD"]).VisionUniVAD,
            ("fit", "decision_function", "predict"),
            id="univad",
        ),
        pytest.param(
            __import__(
                "pyimgano.models.patchcore_lite_map", fromlist=["VisionPatchCoreLiteMap"]
            ).VisionPatchCoreLiteMap,
            ("fit", "decision_function", "predict", "predict_anomaly_map"),
            id="patchcore-lite-map",
        ),
        pytest.param(
            __import__(
                "pyimgano.models.patchcore_lite", fromlist=["CorePatchCoreLite"]
            ).CorePatchCoreLite,
            ("fit", "decision_function"),
            id="patchcore-lite-core",
        ),
        pytest.param(
            __import__(
                "pyimgano.models.patchcore_online",
                fromlist=["CorePatchCoreOnline", "VisionPatchCoreOnline"],
            ).CorePatchCoreOnline,
            ("fit", "partial_fit", "decision_function"),
            id="patchcore-online-core",
        ),
        pytest.param(
            __import__(
                "pyimgano.models.patchcore_online",
                fromlist=["CorePatchCoreOnline", "VisionPatchCoreOnline"],
            ).VisionPatchCoreOnline,
            ("partial_fit",),
            id="patchcore-online-vision",
        ),
        pytest.param(
            __import__(
                "pyimgano.models.core_score_standardizer", fromlist=["CoreScoreStandardizer"]
            ).CoreScoreStandardizer,
            ("fit", "decision_function"),
            id="core-score-standardizer",
        ),
        pytest.param(
            __import__(
                "pyimgano.models.vision_score_standardizer", fromlist=["VisionScoreStandardizer"]
            ).VisionScoreStandardizer,
            ("fit", "decision_function"),
            id="vision-score-standardizer",
        ),
        pytest.param(
            __import__(
                "pyimgano.pipelines.reference_map_pipeline", fromlist=["ReferenceMapPipeline"]
            ).ReferenceMapPipeline,
            ("fit", "decision_function"),
            id="reference-map-pipeline",
        ),
        pytest.param(
            __import__(
                "pyimgano.preprocessing.mixin", fromlist=["ExampleDetectorWithPreprocessing"]
            ).ExampleDetectorWithPreprocessing,
            ("fit", "predict"),
            id="example-preprocessing-detector",
        ),
        pytest.param(
            __import__("pyimgano.models.ssim", fromlist=["SSIMTemplateDetector"]).SSIMTemplateDetector,
            ("fit", "decision_function"),
            id="ssim-template",
        ),
        pytest.param(
            __import__(
                "pyimgano.models.ssim_map", fromlist=["SSIMTemplateMapDetector"]
            ).SSIMTemplateMapDetector,
            ("fit", "predict_anomaly_map", "decision_function"),
            id="ssim-template-map",
        ),
        pytest.param(
            __import__("pyimgano.models.ssim_map", fromlist=["SSIMStructMapDetector"]).SSIMStructMapDetector,
            ("fit", "predict_anomaly_map", "decision_function"),
            id="ssim-struct-map",
        ),
        pytest.param(
            __import__(
                "pyimgano.models.ssim_struct", fromlist=["SSIMStructDetector"]
            ).SSIMStructDetector,
            ("fit", "decision_function"),
            id="ssim-struct",
        ),
        pytest.param(
            __import__(
                "pyimgano.models.template_ncc_map", fromlist=["VisionTemplateNCCMapDetector"]
            ).VisionTemplateNCCMapDetector,
            ("fit", "predict_anomaly_map", "decision_function"),
            id="template-ncc-map",
        ),
        pytest.param(
            __import__("pyimgano.models.hst", fromlist=["CoreHST"]).CoreHST,
            ("fit", "decision_function"),
            id="core-hst",
        ),
        pytest.param(
            __import__(
                "pyimgano.models.knn_degree", fromlist=["CoreKNNGraphDegree"]
            ).CoreKNNGraphDegree,
            ("fit", "decision_function"),
            id="core-knn-degree",
        ),
        pytest.param(
            __import__("pyimgano.models.loop", fromlist=["CoreLoOP"]).CoreLoOP,
            ("fit", "decision_function"),
            id="core-loop",
        ),
        pytest.param(
            __import__(
                "pyimgano.models.mahalanobis", fromlist=["CoreMahalanobis"]
            ).CoreMahalanobis,
            ("fit", "decision_function"),
            id="core-mahalanobis",
        ),
        pytest.param(
            __import__("pyimgano.models.odin", fromlist=["CoreODIN"]).CoreODIN,
            ("fit", "decision_function"),
            id="core-odin",
        ),
        pytest.param(
            __import__("pyimgano.models.sod", fromlist=["CoreSOD"]).CoreSOD,
            ("fit", "decision_function"),
            id="core-sod",
        ),
        pytest.param(
            __import__("pyimgano.models.sod", fromlist=["VisionSOD"]).VisionSOD,
            ("fit", "decision_function"),
            id="vision-sod",
        ),
        pytest.param(
            __import__("pyimgano.models.baseml", fromlist=["BaseVisionDetector"]).BaseVisionDetector,
            ("fit", "decision_function"),
            id="base-vision-detector",
        ),
        pytest.param(
            __import__("pyimgano.models.baseCv", fromlist=["BaseVisionDeepDetector"]).BaseVisionDeepDetector,
            ("fit", "decision_function"),
            id="base-vision-deep-detector",
        ),
        pytest.param(
            __import__("pyimgano.models.lof_native", fromlist=["VisionLOF"]).VisionLOF,
            ("fit", "decision_function"),
            id="vision-lof",
        ),
        pytest.param(
            __import__("pyimgano.models.fastflow", fromlist=["FastFlow"]).FastFlow,
            ("fit",),
            id="fastflow",
        ),
        pytest.param(
            __import__("pyimgano.models.bgad", fromlist=["BGADDetector"]).BGADDetector,
            ("fit", "predict_proba", "predict_anomaly_map", "predict"),
            id="bgad",
        ),
        pytest.param(
            __import__("pyimgano.models.dsr", fromlist=["DSRDetector"]).DSRDetector,
            ("fit", "predict_proba", "predict_anomaly_map", "predict"),
            id="dsr",
        ),
        pytest.param(
            __import__("pyimgano.models.pni", fromlist=["PNIDetector"]).PNIDetector,
            ("fit", "predict_proba", "predict_anomaly_map"),
            id="pni",
        ),
        pytest.param(
            __import__(
                "pyimgano.models.histogram_comparison", fromlist=["HistogramComparison"]
            ).HistogramComparison,
            ("fit", "predict", "predict_label"),
            id="histogram-comparison",
        ),
        pytest.param(
            __import__("pyimgano.models.hog_svm", fromlist=["HOG_SVM"]).HOG_SVM,
            ("fit", "predict", "predict_label"),
            id="hog-svm",
        ),
        pytest.param(
            __import__("pyimgano.models.lbp", fromlist=["LBP"]).LBP,
            ("fit", "predict", "predict_label"),
            id="lbp",
        ),
        pytest.param(
            __import__("pyimgano.models.mad", fromlist=["CoreMAD"]).CoreMAD,
            ("fit", "decision_function"),
            id="core-mad",
        ),
        pytest.param(
            __import__(
                "pyimgano.models.template_matching", fromlist=["TemplateMatching"]
            ).TemplateMatching,
            ("fit", "predict", "predict_label"),
            id="template-matching",
        ),
        pytest.param(
            __import__(
                "pyimgano.models.openclip_backend", fromlist=["VisionOpenCLIPPromptScore"]
            ).VisionOpenCLIPPromptScore,
            ("fit", "decision_function", "predict", "predict_anomaly_map"),
            id="openclip-promptscore",
        ),
        pytest.param(
            __import__(
                "pyimgano.models.openclip_backend", fromlist=["VisionOpenCLIPPatchKNN"]
            ).VisionOpenCLIPPatchKNN,
            ("fit", "decision_function", "predict", "predict_anomaly_map"),
            id="openclip-patchknn",
        ),
    ],
)
def test_non_torch_detectors_use_lowercase_x_signatures(cls, method_names):
    for method_name in method_names:
        _assert_x_signature(getattr(cls, method_name))


@pytest.mark.parametrize(
    ("target", "expected", "legacy"),
    [
        pytest.param(
            __import__("pyimgano.models.kpca", fromlist=["_PyODKernelPCA"])._PyODKernelPCA,
            "copy_x",
            "copy_X",
            id="pyod-kpca-copy-x",
        ),
        pytest.param(
            __import__("pyimgano.models.kpca", fromlist=["CoreKPCA"]).CoreKPCA,
            "copy_x",
            "copy_X",
            id="core-kpca-copy-x",
        ),
        pytest.param(
            __import__("pyimgano.models.kpca", fromlist=["VisionKPCA"]).VisionKPCA,
            "copy_x",
            "copy_X",
            id="vision-kpca-copy-x",
        ),
        pytest.param(
            __import__("pyimgano.models.rgraph", fromlist=["VisionRGraph"]).VisionRGraph,
            "fit_intercept_lr",
            "fit_intercept_LR",
            id="vision-rgraph-fit-intercept-lr",
        ),
        pytest.param(
            __import__("pyimgano.models.spc", fromlist=["SPC"]).SPC,
            "l_value",
            "L",
            id="spc-l-value",
        ),
    ],
)
def test_constructor_signatures_use_lowercase_names(target, expected, legacy):
    _assert_constructor_param_signature(target, expected=expected, legacy=legacy)


@pytest.mark.parametrize(
    ("factory", "legacy_name", "canonical_name"),
    [
        pytest.param(
            lambda: __import__(
                "pyimgano.models.core_knn_cosine", fromlist=["_CosineKNNBackend"]
            )._CosineKNNBackend(
                contamination=0.1,
                n_neighbors=3,
                method="largest",
                normalize=True,
                eps=1e-12,
                n_jobs=1,
            ),
            "_X_train",
            "_x_train",
            id="core-knn-cosine-train-x",
        ),
        pytest.param(
            lambda: __import__("pyimgano.models.abod", fromlist=["CoreABOD"]).CoreABOD(
                contamination=0.1,
                n_neighbors=3,
            ),
            "X_train_",
            "x_train_",
            id="abod-train-x",
        ),
        pytest.param(
            lambda: __import__("pyimgano.models.cof", fromlist=["CoreCOF"]).CoreCOF(
                contamination=0.1,
                n_neighbors=3,
            ),
            "X_train_",
            "x_train_",
            id="cof-train-x",
        ),
        pytest.param(
            lambda: __import__("pyimgano.models.copod", fromlist=["CoreCOPOD"]).CoreCOPOD(
                contamination=0.1,
            ),
            "_X_sorted",
            "_x_sorted",
            id="copod-sorted-x",
        ),
        pytest.param(
            lambda: __import__("pyimgano.models.ecod", fromlist=["CoreECOD"]).CoreECOD(
                contamination=0.1,
            ),
            "_X_sorted",
            "_x_sorted",
            id="ecod-sorted-x",
        ),
        pytest.param(
            lambda: __import__("pyimgano.models.rgraph", fromlist=["CoreRGraph"]).CoreRGraph(
                contamination=0.1,
                transition_steps=2,
                n_nonzero=2,
                gamma=1.0,
                preprocessing=False,
            ),
            "_train_X",
            "_train_x",
            id="rgraph-train-x",
        ),
        pytest.param(
            lambda: __import__("pyimgano.models.sod", fromlist=["CoreSOD"]).CoreSOD(
                contamination=0.1,
                n_neighbors=3,
                ref_set=1,
            ),
            "_X_train",
            "_x_train",
            id="sod-train-x",
        ),
        pytest.param(
            lambda: __import__("pyimgano.models.lscp", fromlist=["CoreLSCP", "_default_lscp_detectors"]).CoreLSCP(
                detector_list=__import__(
                    "pyimgano.models.lscp", fromlist=["_default_lscp_detectors"]
                )._default_lscp_detectors(contamination=0.1, random_state=0),
                contamination=0.1,
                local_region_size=5,
                local_region_iterations=2,
                local_min_features=0.5,
                local_max_features=1.0,
                n_bins=3,
                random_state=0,
            ),
            "X_train_",
            "x_train_",
            id="lscp-train-x",
        ),
    ],
)
def test_legacy_training_attribute_aliases_remain_available(
    factory, legacy_name: str, canonical_name: str
) -> None:
    det = factory()
    x = np.random.default_rng(0).normal(size=(24, 4)).astype(np.float64)

    det.fit(x)

    canonical = getattr(det, canonical_name)
    legacy = getattr(det, legacy_name)

    assert canonical is not None
    assert legacy is not None
    if isinstance(canonical, np.ndarray):
        assert np.array_equal(np.asarray(legacy), canonical)
    else:
        assert legacy == canonical


def test_dcorr_projection_state_legacy_field_aliases_remain_available() -> None:
    from pyimgano.models.dcorr import CoreDCorr

    x = np.random.default_rng(0).normal(size=(18, 4)).astype(np.float64)
    det = CoreDCorr(contamination=0.1, n_projections=2, random_state=0)

    det.fit(x)

    state = det._states[0]
    assert np.array_equal(state.row_mean_A, state.row_mean_a)
    assert np.array_equal(state.row_mean_B, state.row_mean_b)
    assert state.grand_mean_A == state.grand_mean_a
    assert state.grand_mean_B == state.grand_mean_b


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        pytest.param(
            __import__(
                "pyimgano.models.histogram_comparison",
                fromlist=["HistogramComparison"],
            ).HistogramComparison.get_params,
            ("self", "deep"),
            id="histogram-comparison-get-params",
        ),
        pytest.param(
            __import__("pyimgano.models.hog_svm", fromlist=["HOG_SVM"]).HOG_SVM.get_params,
            ("self", "deep"),
            id="hog-svm-get-params",
        ),
        pytest.param(
            __import__("pyimgano.models.lbp", fromlist=["LBP"]).LBP.get_params,
            ("self", "deep"),
            id="lbp-get-params",
        ),
        pytest.param(
            __import__("pyimgano.models.spc", fromlist=["SPC"]).SPC.predict,
            ("self", "x", "return_confidence"),
            id="spc-predict-return-confidence",
        ),
        pytest.param(
            __import__("pyimgano.models.spc", fromlist=["SPC"]).SPC.get_params,
            ("self", "deep"),
            id="spc-get-params",
        ),
        pytest.param(
            __import__(
                "pyimgano.models.template_matching",
                fromlist=["TemplateMatching"],
            ).TemplateMatching.get_params,
            ("self", "deep"),
            id="template-matching-get-params",
        ),
    ],
)
def test_sonar_signature_compat_methods_expose_expected_optional_params(
    method, expected: tuple[str, ...]
) -> None:
    _assert_signature_contains(method, *expected)


def test_knn_internal_train_matrix_legacy_alias_maps_to_lowercase_field() -> None:
    from pyimgano.models.knn import CoreKNN

    det = CoreKNN(contamination=0.1, n_neighbors=3)
    first = np.arange(6, dtype=np.float64).reshape(3, 2)
    second = np.arange(4, dtype=np.float64).reshape(2, 2)

    det._x_train = first
    assert det._X_train is first

    det._X_train = second
    assert det._x_train is second


def test_sklearn_knn_index_legacy_backend_class_alias_maps_to_lowercase_field() -> None:
    from pyimgano.models.knn_index import SklearnKNNIndex

    index = SklearnKNNIndex(n_neighbors=3)

    index._nearest_neighbors_cls = object
    assert index._NearestNeighbors is object

    index._NearestNeighbors = str
    assert index._nearest_neighbors_cls is str


def test_anomalydino_embedder_legacy_image_alias_maps_to_lowercase_field() -> None:
    from pyimgano.models.anomalydino import TorchHubDinoV2Embedder

    embedder = TorchHubDinoV2Embedder()

    embedder._image_cls = object
    assert embedder._Image is object

    embedder._Image = str
    assert embedder._image_cls is str
