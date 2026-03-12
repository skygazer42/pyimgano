"""Lazy compatibility facade for the service layer."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_SERVICE_EXPORT_GROUPS: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    (
        "contracts",
        (
            ("BenchmarkRunRequest", "pyimgano.services.benchmark_service"),
            ("ConfigBackedInferContext", "pyimgano.services.infer_context_service"),
            ("ConfigBackedInferLoadRequest", "pyimgano.services.infer_load_service"),
            ("ConfigBackedInferLoadResult", "pyimgano.services.infer_load_service"),
            ("ContinueOnErrorInferRequest", "pyimgano.services.infer_continue_service"),
            ("ContinueOnErrorInferResult", "pyimgano.services.infer_continue_service"),
            ("DefectsArtifactConfig", "pyimgano.services.infer_artifact_service"),
            ("DefectsArtifactConfigBuildRequest", "pyimgano.services.infer_artifact_service"),
            ("DirectInferLoadRequest", "pyimgano.services.infer_load_service"),
            ("DirectInferLoadResult", "pyimgano.services.infer_load_service"),
            ("InferErrorRecordRequest", "pyimgano.services.infer_output_service"),
            ("InferArtifactOptions", "pyimgano.services.infer_artifact_service"),
            ("InferDetectorWrapperRequest", "pyimgano.services.infer_wrapper_service"),
            ("InferDetectorWrapperResult", "pyimgano.services.infer_wrapper_service"),
            ("InferResultArtifactCliRequest", "pyimgano.services.infer_artifact_service"),
            ("InferOutputTargets", "pyimgano.services.infer_output_service"),
            ("InferOutputTargetsRequest", "pyimgano.services.infer_output_service"),
            ("InferOutputWriteRequest", "pyimgano.services.infer_output_service"),
            ("InferOutputWriteResult", "pyimgano.services.infer_output_service"),
            ("InferResultArtifactAssemblyRequest", "pyimgano.services.infer_artifact_service"),
            ("InferResultArtifactBuildRequest", "pyimgano.services.infer_artifact_service"),
            ("InferResultArtifactRequest", "pyimgano.services.infer_artifact_service"),
            ("InferResultArtifactResult", "pyimgano.services.infer_artifact_service"),
            ("InferRuntimePlanRequest", "pyimgano.services.infer_runtime_service"),
            ("InferRuntimePlanResult", "pyimgano.services.infer_runtime_service"),
            ("FromRunInferContextRequest", "pyimgano.services.infer_context_service"),
            ("InferenceRunResult", "pyimgano.services.inference_service"),
            ("InferConfigContextRequest", "pyimgano.services.infer_context_service"),
            ("PixelPostprocessConfig", "pyimgano.services.benchmark_service"),
            ("PyimListPayload", "pyimgano.services.pyim_service"),
            ("PyimListRequest", "pyimgano.services.pyim_service"),
            ("RobustnessRunRequest", "pyimgano.services.robustness_service"),
            ("SuiteRunRequest", "pyimgano.services.benchmark_service"),
            ("TrainRunRequest", "pyimgano.services.train_service"),
            ("WorkbenchThresholdCalibration", "pyimgano.services.workbench_service"),
        ),
    ),
    (
        "builders",
        (
            ("apply_workbench_overrides", "pyimgano.services.workbench_service"),
            ("apply_train_overrides", "pyimgano.services.train_service"),
            ("build_defects_artifact_config", "pyimgano.services.infer_artifact_service"),
            ("build_feature_info_payload", "pyimgano.services.discovery_service"),
            (
                "build_infer_result_artifact_build_request_from_cli",
                "pyimgano.services.infer_artifact_service",
            ),
            ("build_infer_result_artifact_request", "pyimgano.services.infer_artifact_service"),
            (
                "build_infer_result_artifact_request_from_cli",
                "pyimgano.services.infer_artifact_service",
            ),
            (
                "build_infer_result_artifact_request_from_options",
                "pyimgano.services.infer_artifact_service",
            ),
            ("build_infer_error_record", "pyimgano.services.infer_output_service"),
            ("build_model_info_payload", "pyimgano.services.discovery_service"),
            ("build_model_preset_info_payload", "pyimgano.services.discovery_service"),
            ("build_pixel_postprocess", "pyimgano.services.benchmark_service"),
            ("build_accelerator_checks", "pyimgano.services.doctor_service"),
            ("build_infer_config_payload", "pyimgano.services.workbench_service"),
            ("build_require_extras_check", "pyimgano.services.doctor_service"),
            ("build_suite_checks", "pyimgano.services.doctor_service"),
            ("build_suite_info_payload", "pyimgano.services.discovery_service"),
            ("build_sweep_info_payload", "pyimgano.services.discovery_service"),
            ("build_train_dry_run_payload", "pyimgano.services.train_service"),
        ),
    ),
    (
        "collection_and_preparation",
        (
            ("calibrate_workbench_threshold", "pyimgano.services.workbench_service"),
            ("check_module", "pyimgano.services.doctor_service"),
            ("collect_doctor_payload", "pyimgano.services.doctor_service"),
            ("collect_pyim_listing_payload", "pyimgano.services.pyim_service"),
            ("create_workbench_detector", "pyimgano.services.workbench_service"),
            ("iter_inference_records", "pyimgano.services.inference_service"),
            ("materialize_infer_result_artifacts", "pyimgano.services.infer_artifact_service"),
            ("open_infer_output_targets", "pyimgano.services.infer_output_service"),
            ("prepare_infer_runtime_plan", "pyimgano.services.infer_runtime_service"),
            ("prepare_from_run_context", "pyimgano.services.infer_context_service"),
            ("prepare_infer_config_context", "pyimgano.services.infer_context_service"),
        ),
    ),
    (
        "execution_and_listing",
        (
            ("run_continue_on_error_inference", "pyimgano.services.infer_continue_service"),
            ("apply_infer_detector_wrappers", "pyimgano.services.infer_wrapper_service"),
            ("load_config_backed_infer_detector", "pyimgano.services.infer_load_service"),
            ("load_direct_infer_detector", "pyimgano.services.infer_load_service"),
            ("apply_defects_defaults", "pyimgano.services.infer_options_service"),
            ("resolve_defects_preset_payload", "pyimgano.services.infer_options_service"),
            ("resolve_preprocessing_preset_knobs", "pyimgano.services.infer_options_service"),
            ("list_baseline_suites_payload", "pyimgano.services.discovery_service"),
            ("list_dataset_categories_payload", "pyimgano.services.discovery_service"),
            ("list_discovery_feature_names", "pyimgano.services.discovery_service"),
            ("list_discovery_model_names", "pyimgano.services.discovery_service"),
            ("list_model_preset_infos_payload", "pyimgano.services.discovery_service"),
            ("list_model_preset_names", "pyimgano.services.discovery_service"),
            ("list_sweeps_payload", "pyimgano.services.discovery_service"),
            ("load_train_config", "pyimgano.services.train_service"),
            ("resolve_preprocessing_preset_payload", "pyimgano.services.workbench_service"),
            ("run_inference", "pyimgano.services.inference_service"),
            ("run_benchmark_request", "pyimgano.services.benchmark_service"),
            ("run_robustness_request", "pyimgano.services.robustness_service"),
            ("run_suite_request", "pyimgano.services.benchmark_service"),
            ("run_train_preflight_payload", "pyimgano.services.train_service"),
            ("run_train_request", "pyimgano.services.train_service"),
            ("run_workbench", "pyimgano.services.workbench_service"),
            ("resolve_model_options", "pyimgano.services.model_options"),
            ("resolve_preset_kwargs", "pyimgano.services.model_options"),
            ("split_csv_args", "pyimgano.services.doctor_service"),
            ("write_infer_output_payloads", "pyimgano.services.infer_output_service"),
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
