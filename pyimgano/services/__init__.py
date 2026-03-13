"""Lazy compatibility facade for the service layer."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_BENCHMARK_SERVICE = "pyimgano.services.benchmark_service"
_DATASET_SPLIT_SERVICE = "pyimgano.services.dataset_split_service"
_DISCOVERY_SERVICE = "pyimgano.services.discovery_service"
_DOCTOR_SERVICE = "pyimgano.services.doctor_service"
_INFERENCE_SERVICE = "pyimgano.services.inference_service"
_INFER_ARTIFACT_SERVICE = "pyimgano.services.infer_artifact_service"
_INFER_CONTEXT_SERVICE = "pyimgano.services.infer_context_service"
_INFER_CONTINUE_SERVICE = "pyimgano.services.infer_continue_service"
_INFER_LOAD_SERVICE = "pyimgano.services.infer_load_service"
_INFER_OPTIONS_SERVICE = "pyimgano.services.infer_options_service"
_INFER_OUTPUT_SERVICE = "pyimgano.services.infer_output_service"
_INFER_RUNTIME_SERVICE = "pyimgano.services.infer_runtime_service"
_INFER_WRAPPER_SERVICE = "pyimgano.services.infer_wrapper_service"
_MODEL_OPTIONS_SERVICE = "pyimgano.services.model_options"
_PYIM_SERVICE = "pyimgano.services.pyim_service"
_ROBUSTNESS_SERVICE = "pyimgano.services.robustness_service"
_TRAIN_SERVICE = "pyimgano.services.train_service"
_WORKBENCH_SERVICE = "pyimgano.services.workbench_service"


_SERVICE_EXPORT_GROUPS: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    (
        "contracts",
        (
            ("BenchmarkRunRequest", _BENCHMARK_SERVICE),
            ("ConfigBackedInferContext", _INFER_CONTEXT_SERVICE),
            ("ConfigBackedInferLoadRequest", _INFER_LOAD_SERVICE),
            ("ConfigBackedInferLoadResult", _INFER_LOAD_SERVICE),
            ("ContinueOnErrorInferRequest", _INFER_CONTINUE_SERVICE),
            ("ContinueOnErrorInferResult", _INFER_CONTINUE_SERVICE),
            ("DefectsArtifactConfig", _INFER_ARTIFACT_SERVICE),
            ("DefectsArtifactConfigBuildRequest", _INFER_ARTIFACT_SERVICE),
            ("DirectInferLoadRequest", _INFER_LOAD_SERVICE),
            ("DirectInferLoadResult", _INFER_LOAD_SERVICE),
            ("InferErrorRecordRequest", _INFER_OUTPUT_SERVICE),
            ("InferArtifactOptions", _INFER_ARTIFACT_SERVICE),
            ("InferDetectorWrapperRequest", _INFER_WRAPPER_SERVICE),
            ("InferDetectorWrapperResult", _INFER_WRAPPER_SERVICE),
            ("InferResultArtifactCliRequest", _INFER_ARTIFACT_SERVICE),
            ("InferOutputTargets", _INFER_OUTPUT_SERVICE),
            ("InferOutputTargetsRequest", _INFER_OUTPUT_SERVICE),
            ("InferOutputWriteRequest", _INFER_OUTPUT_SERVICE),
            ("InferOutputWriteResult", _INFER_OUTPUT_SERVICE),
            ("InferResultArtifactAssemblyRequest", _INFER_ARTIFACT_SERVICE),
            ("InferResultArtifactBuildRequest", _INFER_ARTIFACT_SERVICE),
            ("InferResultArtifactRequest", _INFER_ARTIFACT_SERVICE),
            ("InferResultArtifactResult", _INFER_ARTIFACT_SERVICE),
            ("InferRuntimePlanRequest", _INFER_RUNTIME_SERVICE),
            ("InferRuntimePlanResult", _INFER_RUNTIME_SERVICE),
            ("LoadedBenchmarkSplit", _DATASET_SPLIT_SERVICE),
            ("FromRunInferContextRequest", _INFER_CONTEXT_SERVICE),
            ("InferenceRunResult", _INFERENCE_SERVICE),
            ("InferConfigContextRequest", _INFER_CONTEXT_SERVICE),
            ("PixelPostprocessConfig", _BENCHMARK_SERVICE),
            ("PyimListPayload", _PYIM_SERVICE),
            ("PyimListRequest", _PYIM_SERVICE),
            ("RobustnessRunRequest", _ROBUSTNESS_SERVICE),
            ("SuiteRunRequest", _BENCHMARK_SERVICE),
            ("TrainRunRequest", _TRAIN_SERVICE),
            ("WorkbenchThresholdCalibration", _WORKBENCH_SERVICE),
        ),
    ),
    (
        "builders",
        (
            ("apply_workbench_overrides", _WORKBENCH_SERVICE),
            ("apply_train_overrides", _TRAIN_SERVICE),
            ("build_defects_artifact_config", _INFER_ARTIFACT_SERVICE),
            ("build_feature_info_payload", _DISCOVERY_SERVICE),
            (
                "build_infer_result_artifact_build_request_from_cli",
                _INFER_ARTIFACT_SERVICE,
            ),
            ("build_infer_result_artifact_request", _INFER_ARTIFACT_SERVICE),
            (
                "build_infer_result_artifact_request_from_cli",
                _INFER_ARTIFACT_SERVICE,
            ),
            (
                "build_infer_result_artifact_request_from_options",
                _INFER_ARTIFACT_SERVICE,
            ),
            ("build_infer_error_record", _INFER_OUTPUT_SERVICE),
            ("build_model_info_payload", _DISCOVERY_SERVICE),
            ("build_model_preset_info_payload", _DISCOVERY_SERVICE),
            ("build_pixel_postprocess", _BENCHMARK_SERVICE),
            ("build_accelerator_checks", _DOCTOR_SERVICE),
            ("build_infer_config_payload", _WORKBENCH_SERVICE),
            ("build_require_extras_check", _DOCTOR_SERVICE),
            ("build_suite_checks", _DOCTOR_SERVICE),
            ("build_suite_info_payload", _DISCOVERY_SERVICE),
            ("build_sweep_info_payload", _DISCOVERY_SERVICE),
            ("build_train_dry_run_payload", _TRAIN_SERVICE),
        ),
    ),
    (
        "collection_and_preparation",
        (
            ("calibrate_workbench_threshold", _WORKBENCH_SERVICE),
            ("check_module", _DOCTOR_SERVICE),
            ("collect_doctor_payload", _DOCTOR_SERVICE),
            ("collect_pyim_listing_payload", _PYIM_SERVICE),
            ("create_workbench_detector", _WORKBENCH_SERVICE),
            ("iter_inference_records", _INFERENCE_SERVICE),
            ("load_benchmark_style_split", _DATASET_SPLIT_SERVICE),
            ("materialize_infer_result_artifacts", _INFER_ARTIFACT_SERVICE),
            ("open_infer_output_targets", _INFER_OUTPUT_SERVICE),
            ("prepare_infer_runtime_plan", _INFER_RUNTIME_SERVICE),
            ("prepare_from_run_context", _INFER_CONTEXT_SERVICE),
            ("prepare_infer_config_context", _INFER_CONTEXT_SERVICE),
        ),
    ),
    (
        "execution_and_listing",
        (
            ("run_continue_on_error_inference", _INFER_CONTINUE_SERVICE),
            ("apply_infer_detector_wrappers", _INFER_WRAPPER_SERVICE),
            ("load_config_backed_infer_detector", _INFER_LOAD_SERVICE),
            ("load_direct_infer_detector", _INFER_LOAD_SERVICE),
            ("apply_defects_defaults", _INFER_OPTIONS_SERVICE),
            ("resolve_defects_preset_payload", _INFER_OPTIONS_SERVICE),
            ("resolve_preprocessing_preset_knobs", _INFER_OPTIONS_SERVICE),
            ("list_baseline_suites_payload", _DISCOVERY_SERVICE),
            ("list_dataset_categories_payload", _DISCOVERY_SERVICE),
            ("list_discovery_feature_names", _DISCOVERY_SERVICE),
            ("list_discovery_model_names", _DISCOVERY_SERVICE),
            ("list_model_preset_infos_payload", _DISCOVERY_SERVICE),
            ("list_model_preset_names", _DISCOVERY_SERVICE),
            ("list_sweeps_payload", _DISCOVERY_SERVICE),
            ("load_train_config", _TRAIN_SERVICE),
            ("resolve_preprocessing_preset_payload", _WORKBENCH_SERVICE),
            ("run_inference", _INFERENCE_SERVICE),
            ("run_benchmark_request", _BENCHMARK_SERVICE),
            ("run_robustness_request", _ROBUSTNESS_SERVICE),
            ("run_suite_request", _BENCHMARK_SERVICE),
            ("run_train_preflight_payload", _TRAIN_SERVICE),
            ("run_train_request", _TRAIN_SERVICE),
            ("run_workbench", _WORKBENCH_SERVICE),
            ("resolve_model_options", _MODEL_OPTIONS_SERVICE),
            ("resolve_preset_kwargs", _MODEL_OPTIONS_SERVICE),
            ("split_csv_args", _DOCTOR_SERVICE),
            ("write_infer_output_payloads", _INFER_OUTPUT_SERVICE),
        ),
    ),
)


def _iter_service_export_items() -> list[tuple[str, str]]:
    return [item for _group_name, items in _SERVICE_EXPORT_GROUPS for item in items]


def _build_service_export_sources() -> dict[str, str]:
    sources: dict[str, str] = {}
    for export_name, module_name in _iter_service_export_items():
        if export_name in sources:
            raise ValueError(f"Duplicate service root export declared: {export_name}")
        sources[export_name] = module_name
    return sources


_SERVICE_EXPORT_SOURCES = _build_service_export_sources()


__all__ = list(_SERVICE_EXPORT_SOURCES)


def __getattr__(name: str) -> Any:
    try:
        module_name = _SERVICE_EXPORT_SOURCES[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(module_name)
    try:
        value = getattr(module, name)
    except AttributeError as exc:
        raise AttributeError(f"module {__name__!r} could not resolve export {name!r}") from exc
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
