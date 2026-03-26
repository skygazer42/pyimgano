import inspect

import numpy as np
import pytest

_OPTIONAL_IMPORT_ROOTS = {
    "anomalib",
    "faiss",
    "mamba_ssm",
    "open_clip",
    "open_clip_torch",
    "patchcore",
    "skimage",
    "torch",
    "torchvision",
}


def _is_optional_dependency_error(exc: ModuleNotFoundError) -> bool:
    name = getattr(exc, "name", "") or ""
    root = str(name).split(".", 1)[0]
    return root in _OPTIONAL_IMPORT_ROOTS


def _import_attr_or_skip(module_path: str, attr_name: str):
    try:
        module = __import__(module_path, fromlist=[attr_name])
    except ModuleNotFoundError as exc:
        if _is_optional_dependency_error(exc):
            pytest.skip(f"optional dependency missing for {module_path}: {exc.name}")
        raise
    return getattr(module, attr_name)


def _import_attr_param(module_path: str, attr_name: str, *values, id: str):
    try:
        module = __import__(module_path, fromlist=[attr_name])
    except ModuleNotFoundError as exc:
        if _is_optional_dependency_error(exc):
            return pytest.param(
                None,
                *values,
                id=id,
                marks=pytest.mark.skip(
                    reason=f"optional dependency missing for {module_path}: {exc.name}"
                ),
            )
        raise
    return pytest.param(getattr(module, attr_name), *values, id=id)


def _make_instance_without_init(cls):
    # Many vision detectors pull optional heavy deps (e.g., torch) during __init__.
    # For these interface-level tests we bypass __init__ and only exercise the
    # early-guard branches added for SonarCloud signature compatibility.
    return cls.__new__(cls)


def test_predict_return_confidence_raises_without_optional_deps():
    VisionPaDiM = _import_attr_or_skip("pyimgano.models.padim", "VisionPaDiM")
    VisionPatchCore = _import_attr_or_skip("pyimgano.models.patchcore", "VisionPatchCore")

    for cls in (VisionPaDiM, VisionPatchCore):
        inst = _make_instance_without_init(cls)
        with pytest.raises(NotImplementedError):
            cls.predict(inst, X=[], return_confidence=True)


def test_decision_function_rejects_non_positive_batch_size():
    VisionPaDiM = _import_attr_or_skip("pyimgano.models.padim", "VisionPaDiM")
    VisionPatchCore = _import_attr_or_skip("pyimgano.models.patchcore", "VisionPatchCore")

    for cls in (VisionPaDiM, VisionPatchCore):
        inst = _make_instance_without_init(cls)
        with pytest.raises(ValueError):
            cls.decision_function(inst, X=[], batch_size=0)


def test_decision_function_accepts_positive_batch_size_on_empty_input():
    import numpy as np

    VisionPaDiM = _import_attr_or_skip("pyimgano.models.padim", "VisionPaDiM")
    VisionPatchCore = _import_attr_or_skip("pyimgano.models.patchcore", "VisionPatchCore")

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
        _import_attr_param(
            "pyimgano.models.anomalydino",
            "VisionAnomalyDINO",
            ("fit", "decision_function", "predict", "predict_anomaly_map"),
            id="anomalydino",
        ),
        _import_attr_param(
            "pyimgano.models.softpatch",
            "VisionSoftPatch",
            ("fit", "decision_function", "predict", "predict_anomaly_map"),
            id="softpatch",
        ),
        _import_attr_param(
            "pyimgano.models.anomalib_backend",
            "VisionAnomalibCheckpoint",
            ("fit", "decision_function", "predict", "predict_anomaly_map"),
            id="anomalib-backend",
        ),
        _import_attr_param(
            "pyimgano.models.visionad",
            "VisionVisionAD",
            ("fit", "decision_function", "predict", "predict_anomaly_map"),
            id="visionad",
        ),
        _import_attr_param(
            "pyimgano.models.filopp",
            "VisionFiLoPP",
            ("fit", "decision_function", "predict", "predict_anomaly_map"),
            id="filopp",
        ),
        _import_attr_param(
            "pyimgano.models.logsad",
            "VisionLogSAD",
            ("fit", "decision_function", "predict", "predict_anomaly_map"),
            id="logsad",
        ),
        _import_attr_param(
            "pyimgano.models.one_to_normal",
            "VisionOneToNormal",
            ("fit", "decision_function", "predict", "predict_anomaly_map"),
            id="one-to-normal",
        ),
        _import_attr_param(
            "pyimgano.models.superad",
            "VisionSuperAD",
            ("fit", "decision_function", "predict", "predict_anomaly_map"),
            id="superad",
        ),
        _import_attr_param(
            "pyimgano.models.pixel_stats_map",
            "_BasePixelStatsMapDetector",
            ("decision_function", "predict_anomaly_map"),
            id="pixel-stats-base",
        ),
        _import_attr_param(
            "pyimgano.models.pixel_stats_map",
            "VisionPixelMeanAbsDiffMapDetector",
            ("fit",),
            id="pixel-mean",
        ),
        _import_attr_param(
            "pyimgano.models.pixel_stats_map",
            "VisionPixelGaussianMapDetector",
            ("fit",),
            id="pixel-gaussian",
        ),
        _import_attr_param(
            "pyimgano.models.pixel_stats_map",
            "VisionPixelMADMapDetector",
            ("fit",),
            id="pixel-mad",
        ),
        _import_attr_param(
            "pyimgano.models.aaclip",
            "VisionAAClip",
            ("fit", "decision_function", "predict", "predict_anomaly_map"),
            id="aaclip",
        ),
        _import_attr_param(
            "pyimgano.models.adaclip",
            "VisionAdaCLIP",
            ("fit", "decision_function", "predict"),
            id="adaclip",
        ),
        _import_attr_param(
            "pyimgano.models.openclip_patch_map",
            "VisionOpenCLIPPatchMap",
            ("fit", "decision_function", "predict", "predict_anomaly_map"),
            id="openclip-patch-map",
        ),
        _import_attr_param(
            "pyimgano.models.patch_embedding_core_map",
            "VisionPatchEmbeddingCoreMap",
            ("fit", "decision_function", "predict", "predict_anomaly_map"),
            id="patch-embedding-core-map",
        ),
        _import_attr_param(
            "pyimgano.models.patchcore_inspection_backend",
            "VisionPatchCoreInspectionCheckpoint",
            ("fit", "decision_function", "predict", "predict_anomaly_map"),
            id="patchcore-inspection-backend",
        ),
        _import_attr_param(
            "pyimgano.models.univad",
            "VisionUniVAD",
            ("fit", "decision_function", "predict"),
            id="univad",
        ),
        _import_attr_param(
            "pyimgano.models.patchcore_lite_map",
            "VisionPatchCoreLiteMap",
            ("fit", "decision_function", "predict", "predict_anomaly_map"),
            id="patchcore-lite-map",
        ),
        _import_attr_param(
            "pyimgano.models.patchcore_lite",
            "CorePatchCoreLite",
            ("fit", "decision_function"),
            id="patchcore-lite-core",
        ),
        _import_attr_param(
            "pyimgano.models.patchcore_online",
            "CorePatchCoreOnline",
            ("fit", "partial_fit", "decision_function"),
            id="patchcore-online-core",
        ),
        _import_attr_param(
            "pyimgano.models.patchcore_online",
            "VisionPatchCoreOnline",
            ("partial_fit",),
            id="patchcore-online-vision",
        ),
        _import_attr_param(
            "pyimgano.models.core_score_standardizer",
            "CoreScoreStandardizer",
            ("fit", "decision_function"),
            id="core-score-standardizer",
        ),
        _import_attr_param(
            "pyimgano.models.vision_score_standardizer",
            "VisionScoreStandardizer",
            ("fit", "decision_function"),
            id="vision-score-standardizer",
        ),
        _import_attr_param(
            "pyimgano.pipelines.reference_map_pipeline",
            "ReferenceMapPipeline",
            ("fit", "decision_function"),
            id="reference-map-pipeline",
        ),
        _import_attr_param(
            "pyimgano.preprocessing.mixin",
            "ExampleDetectorWithPreprocessing",
            ("fit", "predict"),
            id="example-preprocessing-detector",
        ),
        _import_attr_param(
            "pyimgano.models.ssim",
            "SSIMTemplateDetector",
            ("fit", "decision_function"),
            id="ssim-template",
        ),
        _import_attr_param(
            "pyimgano.models.ssim_map",
            "SSIMTemplateMapDetector",
            ("fit", "predict_anomaly_map", "decision_function"),
            id="ssim-template-map",
        ),
        _import_attr_param(
            "pyimgano.models.ssim_map",
            "SSIMStructMapDetector",
            ("fit", "predict_anomaly_map", "decision_function"),
            id="ssim-struct-map",
        ),
        _import_attr_param(
            "pyimgano.models.ssim_struct",
            "SSIMStructDetector",
            ("fit", "decision_function"),
            id="ssim-struct",
        ),
        _import_attr_param(
            "pyimgano.models.template_ncc_map",
            "VisionTemplateNCCMapDetector",
            ("fit", "predict_anomaly_map", "decision_function"),
            id="template-ncc-map",
        ),
        _import_attr_param(
            "pyimgano.models.hst",
            "CoreHST",
            ("fit", "decision_function"),
            id="core-hst",
        ),
        _import_attr_param(
            "pyimgano.models.knn_degree",
            "CoreKNNGraphDegree",
            ("fit", "decision_function"),
            id="core-knn-degree",
        ),
        _import_attr_param(
            "pyimgano.models.loop",
            "CoreLoOP",
            ("fit", "decision_function"),
            id="core-loop",
        ),
        _import_attr_param(
            "pyimgano.models.mahalanobis",
            "CoreMahalanobis",
            ("fit", "decision_function"),
            id="core-mahalanobis",
        ),
        _import_attr_param(
            "pyimgano.models.odin",
            "CoreODIN",
            ("fit", "decision_function"),
            id="core-odin",
        ),
        _import_attr_param(
            "pyimgano.models.sod",
            "CoreSOD",
            ("fit", "decision_function"),
            id="core-sod",
        ),
        _import_attr_param(
            "pyimgano.models.sod",
            "VisionSOD",
            ("fit", "decision_function"),
            id="vision-sod",
        ),
        _import_attr_param(
            "pyimgano.models.baseml",
            "BaseVisionDetector",
            ("fit", "decision_function"),
            id="base-vision-detector",
        ),
        _import_attr_param(
            "pyimgano.models.baseCv",
            "BaseVisionDeepDetector",
            ("fit", "decision_function"),
            id="base-vision-deep-detector",
        ),
        _import_attr_param(
            "pyimgano.models.lof_native",
            "VisionLOF",
            ("fit", "decision_function"),
            id="vision-lof",
        ),
        _import_attr_param(
            "pyimgano.models.fastflow",
            "FastFlow",
            ("fit",),
            id="fastflow",
        ),
        _import_attr_param(
            "pyimgano.models.bgad",
            "BGADDetector",
            ("fit", "predict_proba", "predict_anomaly_map", "predict"),
            id="bgad",
        ),
        _import_attr_param(
            "pyimgano.models.dsr",
            "DSRDetector",
            ("fit", "predict_proba", "predict_anomaly_map", "predict"),
            id="dsr",
        ),
        _import_attr_param(
            "pyimgano.models.pni",
            "PNIDetector",
            ("fit", "predict_proba", "predict_anomaly_map"),
            id="pni",
        ),
        _import_attr_param(
            "pyimgano.models.histogram_comparison",
            "HistogramComparison",
            ("fit", "predict", "predict_label"),
            id="histogram-comparison",
        ),
        _import_attr_param(
            "pyimgano.models.hog_svm",
            "HOG_SVM",
            ("fit", "predict", "predict_label"),
            id="hog-svm",
        ),
        _import_attr_param(
            "pyimgano.models.lbp",
            "LBP",
            ("fit", "predict", "predict_label"),
            id="lbp",
        ),
        _import_attr_param(
            "pyimgano.models.mad",
            "CoreMAD",
            ("fit", "decision_function"),
            id="core-mad",
        ),
        _import_attr_param(
            "pyimgano.models.template_matching",
            "TemplateMatching",
            ("fit", "predict", "predict_label"),
            id="template-matching",
        ),
        _import_attr_param(
            "pyimgano.models.openclip_backend",
            "VisionOpenCLIPPromptScore",
            ("fit", "decision_function", "predict", "predict_anomaly_map"),
            id="openclip-promptscore",
        ),
        _import_attr_param(
            "pyimgano.models.openclip_backend",
            "VisionOpenCLIPPatchKNN",
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
        _import_attr_param(
            "pyimgano.models.kpca",
            "_PyODKernelPCA",
            "copy_x",
            "copy_X",
            id="pyod-kpca-copy-x",
        ),
        _import_attr_param(
            "pyimgano.models.kpca",
            "CoreKPCA",
            "copy_x",
            "copy_X",
            id="core-kpca-copy-x",
        ),
        _import_attr_param(
            "pyimgano.models.kpca",
            "VisionKPCA",
            "copy_x",
            "copy_X",
            id="vision-kpca-copy-x",
        ),
        _import_attr_param(
            "pyimgano.models.rgraph",
            "VisionRGraph",
            "fit_intercept_lr",
            "fit_intercept_LR",
            id="vision-rgraph-fit-intercept-lr",
        ),
        _import_attr_param(
            "pyimgano.models.spc",
            "SPC",
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
            lambda: __import__(
                "pyimgano.models.lscp", fromlist=["CoreLSCP", "_default_lscp_detectors"]
            ).CoreLSCP(
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
        _import_attr_param(
            "pyimgano.models.histogram_comparison",
            "HistogramComparison",
            ("self", "deep"),
            id="histogram-comparison-get-params",
        ),
        _import_attr_param(
            "pyimgano.models.hog_svm",
            "HOG_SVM",
            ("self", "deep"),
            id="hog-svm-get-params",
        ),
        _import_attr_param(
            "pyimgano.models.lbp",
            "LBP",
            ("self", "deep"),
            id="lbp-get-params",
        ),
        _import_attr_param(
            "pyimgano.models.spc",
            "SPC",
            ("self", "x", "return_confidence"),
            id="spc-predict-return-confidence",
        ),
        _import_attr_param(
            "pyimgano.models.spc",
            "SPC",
            ("self", "deep"),
            id="spc-get-params",
        ),
        _import_attr_param(
            "pyimgano.models.template_matching",
            "TemplateMatching",
            ("self", "deep"),
            id="template-matching-get-params",
        ),
    ],
)
def test_sonar_signature_compat_methods_expose_expected_optional_params(
    method, expected: tuple[str, ...]
) -> None:
    target = method
    if inspect.isclass(method):
        target = method.get_params if expected == ("self", "deep") else method.predict
    _assert_signature_contains(target, *expected)


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
    TorchHubDinoV2Embedder = _import_attr_or_skip(
        "pyimgano.models.anomalydino", "TorchHubDinoV2Embedder"
    )

    embedder = TorchHubDinoV2Embedder()

    embedder._image_cls = object
    assert embedder._Image is object

    embedder._Image = str
    assert embedder._image_cls is str
