from __future__ import annotations

import ast
from pathlib import Path

SERVICE_DIR = Path(__file__).resolve().parents[1] / "pyimgano" / "services"
SRC_DIR = Path(__file__).resolve().parents[1] / "pyimgano"
FORBIDDEN_MODULES = {"pyimgano.cli_common", "pyimgano.cli_presets"}
FORBIDDEN_PYIMGANO_MEMBERS = {"cli_common", "cli_presets"}


def _iter_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "pyimgano":
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")
                continue
            imports.append(module)

    return imports


def _extract_dunder_all(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets
        ):
            continue
        value = ast.literal_eval(node.value)
        if not isinstance(value, (list, tuple)):
            raise AssertionError(f"{path.name}: __all__ must be a list or tuple literal")
        items = list(value)
        if not all(isinstance(item, str) for item in items):
            raise AssertionError(f"{path.name}: __all__ must contain only strings")
        return items

    raise AssertionError(f"{path.name}: missing __all__ declaration")


def test_service_modules_do_not_import_cli_adapter_modules() -> None:
    violations: list[str] = []

    for path in sorted(SERVICE_DIR.glob("*.py")):
        for imported in _iter_imports(path):
            if imported in FORBIDDEN_MODULES or imported in {
                f"pyimgano.{member}" for member in FORBIDDEN_PYIMGANO_MEMBERS
            }:
                violations.append(f"{path.name}: {imported}")

    assert violations == []


def test_infer_services_use_workbench_run_service_boundary() -> None:
    violations: list[str] = []

    for file_name in ("infer_context_service.py", "infer_setup_service.py"):
        path = SERVICE_DIR / file_name
        direct_workbench_imports = [
            imported
            for imported in _iter_imports(path)
            if imported == "pyimgano.workbench.load_run"
        ]
        if direct_workbench_imports:
            violations.append(f"{file_name}: {', '.join(direct_workbench_imports)}")

    assert violations == []


def test_infer_runtime_service_uses_workbench_adaptation_service_boundary() -> None:
    runtime_path = SERVICE_DIR / "infer_runtime_service.py"

    violations = [
        imported
        for imported in _iter_imports(runtime_path)
        if imported == "pyimgano.workbench.adaptation"
    ]

    assert violations == []


def test_infer_options_service_uses_infer_defects_defaults_boundary() -> None:
    options_path = SERVICE_DIR / "infer_options_service.py"
    source = options_path.read_text(encoding="utf-8")

    assert "pyimgano.services.infer_defects_defaults_service" in source
    assert "apply_defects_defaults" in source
    assert "def apply_defects_defaults(" not in source


def test_runner_uses_dataset_loader_boundary() -> None:
    helper_path = SRC_DIR / "workbench" / "multi_category_execution.py"
    source = helper_path.read_text(encoding="utf-8")

    assert "pyimgano.workbench.dataset_loader" in source
    assert "list_workbench_categories" in source
    assert "list_manifest_categories" not in source
    assert "list_dataset_categories" not in source


def test_runner_uses_category_output_boundary() -> None:
    execution_path = SRC_DIR / "workbench" / "category_execution.py"
    source = execution_path.read_text(encoding="utf-8")

    assert "pyimgano.workbench.category_outputs" in source
    assert "save_workbench_category_outputs" in source
    assert "save_jsonl_records(" not in source
    assert "save_anomaly_map_npy(" not in source


def test_runner_uses_category_execution_boundary() -> None:
    runner_path = SRC_DIR / "workbench" / "runner.py"
    source = runner_path.read_text(encoding="utf-8")

    assert "pyimgano.workbench.category_execution" in source
    assert "run_workbench_category" in source
    assert "def _run_category(" not in source


def test_runner_uses_training_runtime_boundary() -> None:
    execution_path = SRC_DIR / "workbench" / "category_execution.py"
    source = execution_path.read_text(encoding="utf-8")

    assert "pyimgano.workbench.training_runtime" in source
    assert "run_workbench_training" in source
    assert "micro_finetune" not in source
    assert "save_checkpoint(" not in source


def test_runner_uses_category_report_boundary() -> None:
    execution_path = SRC_DIR / "workbench" / "category_execution.py"
    source = execution_path.read_text(encoding="utf-8")

    assert "pyimgano.workbench.category_report" in source
    assert "build_workbench_category_report" in source
    assert '"threshold_provenance"' not in source
    assert '"pixel_metrics_status"' not in source
    assert '"test_anomaly_ratio"' not in source


def test_runner_uses_inference_runtime_boundary() -> None:
    execution_path = SRC_DIR / "workbench" / "category_execution.py"
    source = execution_path.read_text(encoding="utf-8")

    assert "pyimgano.workbench.inference_runtime" in source
    assert "run_workbench_inference" in source
    assert "def _maybe_resize_maps_to_masks(" not in source
    assert "evaluate_detector(" not in source
    assert "from pyimgano.inference.api import infer" not in source


def test_runner_uses_detector_setup_boundary() -> None:
    execution_path = SRC_DIR / "workbench" / "category_execution.py"
    source = execution_path.read_text(encoding="utf-8")

    assert "pyimgano.workbench.detector_setup" in source
    assert "build_workbench_runtime_detector" in source
    assert "def _create_detector(" not in source
    assert "apply_tiling(" not in source
    assert "PreprocessingDetector" not in source


def test_runner_uses_runtime_split_boundary() -> None:
    execution_path = SRC_DIR / "workbench" / "category_execution.py"
    source = execution_path.read_text(encoding="utf-8")

    assert "pyimgano.workbench.runtime_split" in source
    assert "prepare_workbench_runtime_split" in source
    assert "limit_train" not in source
    assert "limit_test" not in source


def test_runner_uses_aggregate_report_boundary() -> None:
    helper_path = SRC_DIR / "workbench" / "multi_category_execution.py"
    source = helper_path.read_text(encoding="utf-8")

    assert "pyimgano.workbench.aggregate_report" in source
    assert "build_workbench_aggregate_report" in source
    assert "def _safe_float(" not in source
    assert '"mean_metrics"' not in source
    assert '"std_metrics"' not in source


def test_runner_uses_multi_category_execution_boundary() -> None:
    runner_path = SRC_DIR / "workbench" / "runner.py"
    source = runner_path.read_text(encoding="utf-8")

    assert "pyimgano.workbench.multi_category_execution" in source
    assert "run_all_workbench_categories" in source
    assert "list_workbench_categories(" not in source
    assert "build_workbench_aggregate_report" not in source
    assert "for cat in categories" not in source


def test_runner_uses_run_context_boundary() -> None:
    runner_path = SRC_DIR / "workbench" / "runner.py"
    source = runner_path.read_text(encoding="utf-8")

    assert "pyimgano.workbench.run_context" in source
    assert "initialize_workbench_run_context" in source
    assert "build_workbench_run_dir_name(" not in source
    assert "ensure_run_dir(" not in source
    assert "build_workbench_run_paths(" not in source


def test_runner_uses_run_report_boundary() -> None:
    runner_path = SRC_DIR / "workbench" / "runner.py"
    source = runner_path.read_text(encoding="utf-8")

    assert "pyimgano.workbench.run_report" in source
    assert "persist_workbench_run_report" in source
    assert "save_run_report(" not in source
    assert 'payload["run_dir"] = ' not in source


def test_runner_uses_infer_config_payload_boundary() -> None:
    runner_path = SRC_DIR / "workbench" / "runner.py"
    source = runner_path.read_text(encoding="utf-8")

    assert "pyimgano.workbench.infer_config_payload" in source
    assert "build_workbench_infer_config_payload" in source
    assert "pyimgano.services.workbench_service" not in source


def test_runner_uses_runtime_guardrails_boundary() -> None:
    runner_path = SRC_DIR / "workbench" / "runner.py"
    source = runner_path.read_text(encoding="utf-8")

    assert "pyimgano.workbench.runtime_guardrails" in source
    assert "validate_workbench_runtime_guardrails" in source
    assert "adaptation.save_maps requires output.save_run=true." not in source
    assert "training.enabled requires output.save_run=true." not in source
    assert "compute_model_capabilities(" not in source
    assert "MODEL_REGISTRY.info(" not in source
    assert "def _require_pixel_map_model_for_workbench_features(" not in source


def test_runtime_guardrails_uses_model_compatibility_boundary() -> None:
    guardrails_path = SRC_DIR / "workbench" / "runtime_guardrails.py"
    source = guardrails_path.read_text(encoding="utf-8")

    assert "pyimgano.workbench.model_compatibility" in source
    assert "collect_workbench_pixel_map_requirements" in source
    assert "load_workbench_model_capabilities" in source
    assert "compute_model_capabilities(" not in source
    assert "MODEL_REGISTRY.info(" not in source


def test_preflight_uses_manifest_preflight_boundary() -> None:
    preflight_path = SRC_DIR / "workbench" / "preflight.py"
    source = preflight_path.read_text(encoding="utf-8")

    assert "pyimgano.workbench.preflight_summary" in source
    assert "resolve_workbench_preflight_summary" in source
    assert "run_manifest_preflight(" not in source
    assert "def _preflight_manifest(" not in source


def test_preflight_uses_non_manifest_preflight_boundary() -> None:
    preflight_path = SRC_DIR / "workbench" / "preflight.py"
    source = preflight_path.read_text(encoding="utf-8")

    assert "pyimgano.workbench.preflight_summary" in source
    assert "resolve_workbench_preflight_summary" in source
    assert "run_non_manifest_preflight(" not in source
    assert "def _preflight_non_manifest(" not in source


def test_preflight_uses_summary_boundary() -> None:
    preflight_path = SRC_DIR / "workbench" / "preflight.py"
    source = preflight_path.read_text(encoding="utf-8")

    assert "pyimgano.workbench.preflight_summary" in source
    assert "resolve_workbench_preflight_summary" in source
    assert 'if ds == "manifest":' not in source
    assert 'dataset.lower() == "manifest"' not in source


def test_preflight_summary_module_hosts_dataset_dispatch_logic() -> None:
    helper_path = SRC_DIR / "workbench" / "preflight_summary.py"
    source = helper_path.read_text(encoding="utf-8")

    assert "def resolve_workbench_preflight_summary(" in source
    assert "pyimgano.workbench.preflight_dispatch" in source
    assert "resolve_preflight_dataset_dispatch" in source
    assert "pyimgano.workbench.manifest_preflight" in source
    assert "pyimgano.workbench.non_manifest_preflight" in source
    assert "run_manifest_preflight" in source
    assert "run_non_manifest_preflight" in source
    assert 'dataset.lower() == "manifest"' not in source


def test_preflight_uses_types_boundary() -> None:
    preflight_path = SRC_DIR / "workbench" / "preflight.py"
    source = preflight_path.read_text(encoding="utf-8")

    assert "pyimgano.workbench.preflight_types" in source
    assert "IssueSeverity" in source
    assert "PreflightIssue" in source
    assert "PreflightReport" in source
    assert "class PreflightIssue:" not in source
    assert "class PreflightReport:" not in source
    assert 'IssueSeverity = Literal["error", "warning", "info"]' not in source
    assert "@dataclass(" not in source


def test_preflight_types_module_hosts_preflight_dataclasses() -> None:
    helper_path = SRC_DIR / "workbench" / "preflight_types.py"
    source = helper_path.read_text(encoding="utf-8")

    assert 'IssueSeverity = Literal["error", "warning", "info"]' in source
    assert "class PreflightIssue:" in source
    assert "class PreflightReport:" in source
    assert "@dataclass(frozen=True)" in source


def test_preflight_uses_issue_factory_boundary() -> None:
    preflight_path = SRC_DIR / "workbench" / "preflight.py"
    source = preflight_path.read_text(encoding="utf-8")

    assert "pyimgano.workbench.preflight_issue_factory" in source
    assert "build_preflight_issue" in source
    assert "def _issue(" not in source
    assert "PreflightIssue(" not in source


def test_preflight_issue_factory_module_hosts_issue_construction_logic() -> None:
    helper_path = SRC_DIR / "workbench" / "preflight_issue_factory.py"
    source = helper_path.read_text(encoding="utf-8")

    assert "def build_preflight_issue(" in source
    assert "PreflightIssue(" in source
    assert "code=str(code)" in source
    assert "message=str(message)" in source


def test_preflight_uses_report_builder_boundary() -> None:
    preflight_path = SRC_DIR / "workbench" / "preflight.py"
    source = preflight_path.read_text(encoding="utf-8")

    assert "pyimgano.workbench.preflight_report" in source
    assert "build_preflight_report" in source
    assert "PreflightReport(" not in source


def test_preflight_report_module_hosts_report_assembly_logic() -> None:
    helper_path = SRC_DIR / "workbench" / "preflight_report.py"
    source = helper_path.read_text(encoding="utf-8")

    assert "def build_preflight_report(" in source
    assert "PreflightReport(" in source
    assert "dataset=str(dataset)" in source
    assert "category=str(category)" in source


def test_non_manifest_category_listing_uses_workbench_dataset_loader_boundary() -> None:
    helper_path = SRC_DIR / "workbench" / "non_manifest_category_listing.py"
    source = helper_path.read_text(encoding="utf-8")

    assert "pyimgano.workbench.dataset_loader" in source
    assert "list_workbench_categories" in source
    assert "list_dataset_categories" not in source


def test_non_manifest_preflight_uses_source_validation_boundary() -> None:
    helper_path = SRC_DIR / "workbench" / "non_manifest_preflight.py"
    source = helper_path.read_text(encoding="utf-8")

    assert "pyimgano.workbench.non_manifest_source_validation" in source
    assert "resolve_non_manifest_preflight_source" in source
    assert "DATASET_ROOT_MISSING" not in source
    assert "CUSTOM_DATASET_INVALID_STRUCTURE" not in source


def test_non_manifest_preflight_uses_category_selection_boundary() -> None:
    helper_path = SRC_DIR / "workbench" / "non_manifest_preflight.py"
    source = helper_path.read_text(encoding="utf-8")

    assert "pyimgano.workbench.non_manifest_category_selection" in source
    assert "select_non_manifest_preflight_categories" in source
    assert "DATASET_CATEGORY_EMPTY" not in source
    assert 'category.lower() != "all"' not in source


def test_non_manifest_preflight_uses_category_listing_boundary() -> None:
    helper_path = SRC_DIR / "workbench" / "non_manifest_preflight.py"
    source = helper_path.read_text(encoding="utf-8")

    assert "pyimgano.workbench.non_manifest_category_listing" in source
    assert "load_non_manifest_preflight_categories" in source
    assert "DATASET_CATEGORY_LIST_FAILED" not in source
    assert "Unable to list dataset categories." not in source
    assert "list_workbench_categories(config=config)" not in source


def test_non_manifest_preflight_uses_report_boundary() -> None:
    helper_path = SRC_DIR / "workbench" / "non_manifest_preflight.py"
    source = helper_path.read_text(encoding="utf-8")

    assert "pyimgano.workbench.non_manifest_preflight_report" in source
    assert "build_non_manifest_preflight_report" in source
    assert '"dataset_root": str(root)' not in source
    assert '"categories": categories' not in source
    assert '"ok": True' not in source


def test_non_manifest_source_validation_module_hosts_source_checks() -> None:
    helper_path = SRC_DIR / "workbench" / "non_manifest_source_validation.py"
    source = helper_path.read_text(encoding="utf-8")

    assert "def resolve_non_manifest_preflight_source(" in source
    assert "DATASET_ROOT_MISSING" in source
    assert "CUSTOM_DATASET_INVALID_STRUCTURE" in source
    assert "CustomDataset" in source


def test_non_manifest_category_selection_module_hosts_requested_category_logic() -> None:
    helper_path = SRC_DIR / "workbench" / "non_manifest_category_selection.py"
    source = helper_path.read_text(encoding="utf-8")

    assert "def select_non_manifest_preflight_categories(" in source
    assert "DATASET_CATEGORY_EMPTY" in source
    assert 'requested_category.lower() == "all"' in source
    assert "sorted(" in source


def test_non_manifest_category_listing_module_hosts_category_load_failure_handling() -> None:
    helper_path = SRC_DIR / "workbench" / "non_manifest_category_listing.py"
    source = helper_path.read_text(encoding="utf-8")

    assert "def load_non_manifest_preflight_categories(" in source
    assert "list_workbench_categories" in source
    assert "DATASET_CATEGORY_LIST_FAILED" in source
    assert "Unable to list dataset categories." in source


def test_non_manifest_preflight_report_module_hosts_success_payload_logic() -> None:
    helper_path = SRC_DIR / "workbench" / "non_manifest_preflight_report.py"
    source = helper_path.read_text(encoding="utf-8")

    assert "def build_non_manifest_preflight_report(" in source
    assert '"dataset_root": str(root)' in source
    assert '"categories": categories' in source
    assert '"ok": True' in source


def test_dataset_loader_uses_manifest_split_policy_boundary() -> None:
    helper_path = SRC_DIR / "workbench" / "dataset_loader.py"
    source = helper_path.read_text(encoding="utf-8")

    assert "pyimgano.workbench.manifest_split_policy" in source
    assert "build_manifest_split_policy" in source
    assert "ManifestSplitPolicy(" not in source


def test_preflight_uses_model_compatibility_boundary() -> None:
    helper_path = SRC_DIR / "workbench" / "preflight_model_compat.py"
    source = helper_path.read_text(encoding="utf-8")

    assert "pyimgano.workbench.model_compatibility" in source
    assert "collect_workbench_pixel_map_requirements" in source
    assert "load_workbench_model_capabilities" in source
    assert "compute_model_capabilities(" not in source
    assert "MODEL_REGISTRY.info(" not in source


def test_preflight_uses_model_compat_preflight_boundary() -> None:
    preflight_path = SRC_DIR / "workbench" / "preflight.py"
    source = preflight_path.read_text(encoding="utf-8")

    assert "pyimgano.workbench.preflight_model_compat" in source
    assert "run_workbench_model_compat_preflight" in source
    assert "def _preflight_model_compat(" not in source


def test_workbench_package_uses_lazy_export_facade() -> None:
    package_path = SRC_DIR / "workbench" / "__init__.py"
    source = package_path.read_text(encoding="utf-8")

    assert "_WORKBENCH_EXPORT_GROUPS" in source
    assert "_WORKBENCH_EXPORT_SOURCES" in source
    assert "def __getattr__(" in source
    assert "def __dir__(" in source
    assert "from .runner import" not in source
    assert "from .preflight import" not in source


def test_workbench_package_routes_preflight_type_exports_through_type_boundary() -> None:
    package_path = SRC_DIR / "workbench" / "__init__.py"
    source = package_path.read_text(encoding="utf-8")

    assert (
        '("PreflightIssue", "pyimgano.workbench.preflight_types")' in source
        or '("PreflightIssue", _WORKBENCH_PREFLIGHT_TYPES_MODULE)' in source
    )
    assert (
        '("PreflightReport", "pyimgano.workbench.preflight_types")' in source
        or '("PreflightReport", _WORKBENCH_PREFLIGHT_TYPES_MODULE)' in source
    )
    assert (
        '("run_preflight", "pyimgano.workbench.preflight")' in source
        or '("run_preflight", _WORKBENCH_PREFLIGHT_MODULE)' in source
    )


def test_config_module_uses_config_boundaries() -> None:
    config_path = SRC_DIR / "workbench" / "config.py"
    source = config_path.read_text(encoding="utf-8")

    assert "pyimgano.workbench.config_types" in source
    assert "pyimgano.workbench.config_parser" in source
    assert "build_workbench_config_from_dict" in source
    assert "def _require_mapping(" not in source
    assert "def _optional_int(" not in source
    assert "def _optional_float(" not in source
    assert "def _parse_resize(" not in source
    assert "class WorkbenchConfig:" not in source
    assert "@dataclass(" not in source


def test_config_parser_module_hosts_parsing_helpers() -> None:
    parser_path = SRC_DIR / "workbench" / "config_parser.py"
    source = parser_path.read_text(encoding="utf-8")

    assert "def build_workbench_config_from_dict(" in source
    assert "pyimgano.workbench.config_parse_primitives" in source
    assert "pyimgano.workbench.config_section_parsers" in source
    assert "def _require_mapping(" not in source
    assert "def _optional_int(" not in source
    assert "def _optional_float(" not in source
    assert "def _parse_resize(" not in source
    assert "def _parse_dataset_config(" not in source
    assert "def _parse_defects_config(" not in source


def test_config_parse_primitives_module_hosts_scalar_parsing_helpers() -> None:
    helper_path = SRC_DIR / "workbench" / "config_parse_primitives.py"
    source = helper_path.read_text(encoding="utf-8")

    assert "def _require_mapping(" in source
    assert "def _optional_int(" in source
    assert "def _optional_float(" in source
    assert "def _parse_resize(" in source
    assert "def _parse_checkpoint_name(" in source
    assert "def _parse_roi_xyxy_norm(" in source


def test_config_section_parsers_module_hosts_section_parsing_logic() -> None:
    helper_path = SRC_DIR / "workbench" / "config_section_parsers.py"
    source = helper_path.read_text(encoding="utf-8")

    assert "pyimgano.workbench.config_dataset_section_parser" in source
    assert "def _parse_dataset_config(" not in source
    assert "pyimgano.workbench.config_model_output_section_parser" in source
    assert "def _parse_model_config(" not in source
    assert "def _parse_output_config(" not in source
    assert "pyimgano.workbench.config_adaptation_section_parser" in source
    assert "def _parse_adaptation_config(" not in source
    assert "pyimgano.workbench.config_preprocessing_section_parser" in source
    assert "def _parse_preprocessing_config(" not in source
    assert "pyimgano.workbench.config_training_section_parser" in source
    assert "def _parse_training_config(" not in source
    assert "pyimgano.workbench.config_defects_section_parser" in source
    assert "def _parse_defects_config(" not in source


def test_config_section_parsers_uses_dataset_section_boundary() -> None:
    helper_path = SRC_DIR / "workbench" / "config_section_parsers.py"
    source = helper_path.read_text(encoding="utf-8")

    assert "pyimgano.workbench.config_dataset_section_parser" in source
    assert "_parse_dataset_config" in source
    assert "_parse_split_policy_config" in source
    assert "def _parse_split_policy_config(" not in source
    assert "def _parse_dataset_config(" not in source


def test_config_dataset_section_parser_module_hosts_dataset_parsing_logic() -> None:
    helper_path = SRC_DIR / "workbench" / "config_dataset_section_parser.py"
    source = helper_path.read_text(encoding="utf-8")

    assert "def _parse_split_policy_config(" in source
    assert "def _parse_dataset_config(" in source
    assert "dataset.manifest_path is required when dataset.name='manifest'" in source
    assert "dataset.split_policy.test_normal_fraction" in source


def test_config_section_parsers_uses_model_output_section_boundary() -> None:
    helper_path = SRC_DIR / "workbench" / "config_section_parsers.py"
    source = helper_path.read_text(encoding="utf-8")

    assert "pyimgano.workbench.config_model_output_section_parser" in source
    assert "_parse_model_config" in source
    assert "_parse_output_config" in source
    assert "def _parse_model_config(" not in source
    assert "def _parse_output_config(" not in source


def test_config_model_output_section_parser_module_hosts_model_output_parsing_logic() -> None:
    helper_path = SRC_DIR / "workbench" / "config_model_output_section_parser.py"
    source = helper_path.read_text(encoding="utf-8")

    assert "def _parse_model_config(" in source
    assert "def _parse_output_config(" in source
    assert "model.name is required" in source
    assert "model.model_kwargs" in source
    assert "model.contamination" in source
    assert "save_run=bool" in source


def test_config_section_parsers_uses_preprocessing_section_boundary() -> None:
    helper_path = SRC_DIR / "workbench" / "config_section_parsers.py"
    source = helper_path.read_text(encoding="utf-8")

    assert "pyimgano.workbench.config_preprocessing_section_parser" in source
    assert "_parse_preprocessing_config" in source
    assert "def _parse_preprocessing_config(" not in source


def test_config_preprocessing_section_parser_module_hosts_preprocessing_parsing_logic() -> None:
    helper_path = SRC_DIR / "workbench" / "config_preprocessing_section_parser.py"
    source = helper_path.read_text(encoding="utf-8")

    assert "def _parse_preprocessing_config(" in source
    assert "IlluminationContrastKnobs(" in source
    assert "white_balance must be one of" in source
    assert "homomorphic_cutoff" in source
    assert "clahe_clip_limit" in source
    assert "contrast percentiles must satisfy" in source


def test_config_section_parsers_uses_adaptation_section_boundary() -> None:
    helper_path = SRC_DIR / "workbench" / "config_section_parsers.py"
    source = helper_path.read_text(encoding="utf-8")

    assert "pyimgano.workbench.config_adaptation_section_parser" in source
    assert "_parse_adaptation_config" in source
    assert "def _parse_adaptation_config(" not in source


def test_config_adaptation_section_parser_module_hosts_adaptation_parsing_logic() -> None:
    helper_path = SRC_DIR / "workbench" / "config_adaptation_section_parser.py"
    source = helper_path.read_text(encoding="utf-8")

    assert "def _parse_adaptation_config(" in source
    assert "adaptation.tiling.tile_size" in source
    assert "score_topk" in source
    assert "adaptation.postprocess.component_threshold" in source
    assert "save_maps=bool" in source


def test_config_section_parsers_uses_defects_section_boundary() -> None:
    helper_path = SRC_DIR / "workbench" / "config_section_parsers.py"
    source = helper_path.read_text(encoding="utf-8")

    assert "pyimgano.workbench.config_defects_section_parser" in source
    assert "_parse_defects_config" in source
    assert "def _parse_defects_config(" not in source


def test_config_defects_section_parser_module_hosts_defects_parsing_logic() -> None:
    helper_path = SRC_DIR / "workbench" / "config_defects_section_parser.py"
    source = helper_path.read_text(encoding="utf-8")

    assert "def _parse_defects_config(" in source
    assert "def _parse_map_smoothing_config(" in source
    assert "def _parse_hysteresis_config(" in source
    assert "def _parse_shape_filters_config(" in source
    assert "def _parse_merge_nearby_config(" in source
    assert "defects.map_smoothing.method must be one of" in source
    assert "defects.pixel_normal_quantile must be in (0,1]" in source
    assert "defects.mask_format must be 'png' or 'npy'" in source


def test_config_section_parsers_uses_training_section_boundary() -> None:
    helper_path = SRC_DIR / "workbench" / "config_section_parsers.py"
    source = helper_path.read_text(encoding="utf-8")

    assert "pyimgano.workbench.config_training_section_parser" in source
    assert "_parse_training_config" in source
    assert "def _parse_training_config(" not in source


def test_config_training_section_parser_module_hosts_training_parsing_logic() -> None:
    helper_path = SRC_DIR / "workbench" / "config_training_section_parser.py"
    source = helper_path.read_text(encoding="utf-8")

    assert "def _parse_training_config(" in source
    assert "training.epochs must be positive or null" in source
    assert "training.lr must be positive or null" in source
    assert "_parse_checkpoint_name" in source


def test_adaptation_module_uses_type_and_runtime_boundaries() -> None:
    adaptation_path = SRC_DIR / "workbench" / "adaptation.py"
    source = adaptation_path.read_text(encoding="utf-8")

    assert "pyimgano.workbench.adaptation_types" in source
    assert "pyimgano.workbench.adaptation_runtime" in source
    assert "class TilingConfig:" not in source
    assert "class MapPostprocessConfig:" not in source
    assert "class AdaptationConfig:" not in source
    assert "@dataclass(" not in source
    assert "TiledDetector" not in source
    assert "AnomalyMapPostprocess" not in source


def test_config_parser_uses_adaptation_type_boundary() -> None:
    parser_path = SRC_DIR / "workbench" / "config_adaptation_section_parser.py"
    source = parser_path.read_text(encoding="utf-8")

    assert "pyimgano.workbench.adaptation_types" in source
    assert "pyimgano.workbench.adaptation import AdaptationConfig" not in source
    assert "pyimgano.workbench.adaptation import MapPostprocessConfig" not in source
    assert "pyimgano.workbench.adaptation import TilingConfig" not in source


def test_detector_setup_uses_adaptation_runtime_boundary() -> None:
    detector_setup_path = SRC_DIR / "workbench" / "detector_setup.py"
    source = detector_setup_path.read_text(encoding="utf-8")

    assert "pyimgano.workbench.adaptation_runtime" in source
    assert "pyimgano.workbench.adaptation import apply_tiling" not in source


def test_category_execution_uses_adaptation_runtime_boundary() -> None:
    execution_path = SRC_DIR / "workbench" / "category_execution.py"
    source = execution_path.read_text(encoding="utf-8")

    assert "pyimgano.workbench.adaptation_runtime" in source
    assert "pyimgano.workbench.adaptation import build_postprocess" not in source


def test_load_run_module_uses_run_artifact_and_checkpoint_boundaries() -> None:
    load_run_path = SRC_DIR / "workbench" / "load_run.py"
    source = load_run_path.read_text(encoding="utf-8")

    assert "pyimgano.workbench.run_artifacts" in source
    assert "pyimgano.workbench.checkpoint_restore" in source
    assert "def _load_json_object(" not in source
    assert "json.loads(" not in source
    assert "torch.load(" not in source
    assert "load_state_dict(" not in source
    assert "unwrap_runtime_detector" not in source


def test_run_artifacts_module_hosts_run_loading_helpers() -> None:
    run_artifacts_path = SRC_DIR / "workbench" / "run_artifacts.py"
    source = run_artifacts_path.read_text(encoding="utf-8")

    assert "def load_workbench_config_from_run(" in source
    assert "def load_report_from_run(" in source
    assert "def select_category_report(" in source
    assert "def extract_threshold(" in source
    assert "def resolve_checkpoint_path(" in source


def test_checkpoint_restore_module_hosts_detector_restore_logic() -> None:
    checkpoint_restore_path = SRC_DIR / "workbench" / "checkpoint_restore.py"
    source = checkpoint_restore_path.read_text(encoding="utf-8")

    assert "def load_checkpoint_into_detector(" in source
    assert "load_state_dict(" in source
    assert "unwrap_runtime_detector" in source


def test_manifest_preflight_uses_component_boundaries() -> None:
    helper_path = SRC_DIR / "workbench" / "manifest_preflight.py"
    source = helper_path.read_text(encoding="utf-8")

    assert "pyimgano.workbench.manifest_source_validation" in source
    assert "pyimgano.workbench.manifest_record_preflight" in source
    assert "pyimgano.workbench.manifest_preflight_categories" in source
    assert "resolve_manifest_preflight_source" in source
    assert "resolve_manifest_preflight_records" in source
    assert "preflight_manifest_categories" in source
    assert "def _load_manifest_records_best_effort(" not in source
    assert "def _parse_manifest_json(" not in source
    assert "def _resolve_manifest_path_best_effort(" not in source
    assert "def _preflight_manifest_category(" not in source


def test_manifest_preflight_uses_source_validation_boundary() -> None:
    helper_path = SRC_DIR / "workbench" / "manifest_preflight.py"
    source = helper_path.read_text(encoding="utf-8")

    assert "pyimgano.workbench.manifest_source_validation" in source
    assert "resolve_manifest_preflight_source" in source
    assert "MANIFEST_UNSUPPORTED_INPUT_MODE" not in source
    assert "MANIFEST_PATH_MISSING" not in source
    assert "MANIFEST_NOT_FOUND" not in source
    assert "MANIFEST_NOT_A_FILE" not in source
    assert "MANIFEST_NOT_READABLE" not in source


def test_manifest_preflight_uses_record_preflight_boundary() -> None:
    helper_path = SRC_DIR / "workbench" / "manifest_preflight.py"
    source = helper_path.read_text(encoding="utf-8")

    assert "pyimgano.workbench.manifest_record_preflight" in source
    assert "resolve_manifest_preflight_records" in source
    assert "MANIFEST_EMPTY" not in source
    assert "Manifest contains no valid records." not in source
    assert "if not records:" not in source


def test_manifest_preflight_uses_split_policy_boundary() -> None:
    helper_path = SRC_DIR / "workbench" / "manifest_preflight.py"
    source = helper_path.read_text(encoding="utf-8")

    assert "pyimgano.workbench.manifest_split_policy" in source
    assert "build_manifest_split_policy" in source
    assert "def _manifest_split_policy_from_config(" not in source
    assert "ManifestSplitPolicy(" not in source


def test_manifest_preflight_uses_category_batch_boundary() -> None:
    helper_path = SRC_DIR / "workbench" / "manifest_preflight.py"
    source = helper_path.read_text(encoding="utf-8")

    assert "pyimgano.workbench.manifest_preflight_categories" in source
    assert "preflight_manifest_categories" in source
    assert "for cat in categories" not in source
    assert "cat_records = [r for r in records" not in source


def test_manifest_preflight_uses_category_selection_boundary() -> None:
    helper_path = SRC_DIR / "workbench" / "manifest_preflight.py"
    source = helper_path.read_text(encoding="utf-8")

    assert "pyimgano.workbench.manifest_category_selection" in source
    assert "select_manifest_preflight_categories" in source
    assert "MANIFEST_CATEGORY_EMPTY" not in source
    assert 'requested_category.lower() == "all"' not in source


def test_manifest_preflight_uses_report_assembly_boundary() -> None:
    helper_path = SRC_DIR / "workbench" / "manifest_preflight.py"
    source = helper_path.read_text(encoding="utf-8")

    assert "pyimgano.workbench.manifest_preflight_report" in source
    assert "build_manifest_preflight_report" in source
    assert "out: dict[str, Any] = {" not in source
    assert 'out["manifest"] = {"ok": True}' not in source
    assert "split_policy={" not in source
    assert '"mode": str(policy.mode)' not in source
    assert '"scope": str(policy.scope)' not in source
    assert '"seed": int(policy.seed)' not in source
    assert '"test_normal_fraction": float(policy.test_normal_fraction)' not in source


def test_manifest_category_selection_module_hosts_requested_category_logic() -> None:
    helper_path = SRC_DIR / "workbench" / "manifest_category_selection.py"
    source = helper_path.read_text(encoding="utf-8")

    assert "def select_manifest_preflight_categories(" in source
    assert "MANIFEST_CATEGORY_EMPTY" in source
    assert 'requested_category.lower() == "all"' in source
    assert "sorted(" in source


def test_manifest_split_policy_module_hosts_config_mapping_logic() -> None:
    helper_path = SRC_DIR / "workbench" / "manifest_split_policy.py"
    source = helper_path.read_text(encoding="utf-8")

    assert "def build_manifest_split_policy(" in source
    assert "ManifestSplitPolicy" in source
    assert "config.dataset.split_policy" in source
    assert "config.seed" in source


def test_manifest_preflight_categories_module_hosts_category_dispatch_logic() -> None:
    helper_path = SRC_DIR / "workbench" / "manifest_preflight_categories.py"
    source = helper_path.read_text(encoding="utf-8")

    assert "def preflight_manifest_categories(" in source
    assert "preflight_manifest_category(" in source
    assert "for cat in categories" in source
    assert "str(record.category)" in source


def test_manifest_preflight_report_module_hosts_payload_assembly_logic() -> None:
    helper_path = SRC_DIR / "workbench" / "manifest_preflight_report.py"
    source = helper_path.read_text(encoding="utf-8")

    assert "def build_manifest_preflight_report(" in source
    assert '"mode": str(policy.mode)' in source
    assert '"scope": str(policy.scope)' in source
    assert '"seed": int(policy.seed)' in source
    assert '"test_normal_fraction": float(policy.test_normal_fraction)' in source
    assert '"per_category": per_category if requested_all else None' in source
    assert 'report["manifest"] = {"ok": True}' in source


def test_manifest_source_validation_module_hosts_manifest_source_checks() -> None:
    helper_path = SRC_DIR / "workbench" / "manifest_source_validation.py"
    source = helper_path.read_text(encoding="utf-8")

    assert "def resolve_manifest_preflight_source(" in source
    assert "MANIFEST_UNSUPPORTED_INPUT_MODE" in source
    assert "MANIFEST_PATH_MISSING" in source
    assert "MANIFEST_NOT_FOUND" in source
    assert "MANIFEST_NOT_A_FILE" in source
    assert "MANIFEST_NOT_READABLE" in source
    assert "DATASET_ROOT_MISSING" in source


def test_manifest_record_preflight_module_hosts_record_loading_helpers() -> None:
    helper_path = SRC_DIR / "workbench" / "manifest_record_preflight.py"
    source = helper_path.read_text(encoding="utf-8")

    assert "def load_manifest_records_best_effort(" in source
    assert "def resolve_manifest_preflight_records(" in source
    assert "def resolve_manifest_path_best_effort(" in source
    assert "def _parse_manifest_json(" in source
    assert "MANIFEST_EMPTY" in source


def test_manifest_category_preflight_module_hosts_category_analysis() -> None:
    helper_path = SRC_DIR / "workbench" / "manifest_category_preflight.py"
    source = helper_path.read_text(encoding="utf-8")

    assert "def preflight_manifest_category(" in source
    assert "counts_by_split = {" not in source
    assert "explicit_test_labels = {" not in source
    assert "MANIFEST_MISSING_IMAGE" not in source
    assert "MANIFEST_MISSING_MASK" not in source
    assert "MANIFEST_DUPLICATE_IMAGE" not in source
    assert "MANIFEST_GROUP_SPLIT_CONFLICT" not in source
    assert "MANIFEST_GROUP_ANOMALY_IN_TRAIN" not in source
    assert "assigned_counts = {" not in source


def test_manifest_category_preflight_uses_path_helper_boundary() -> None:
    helper_path = SRC_DIR / "workbench" / "manifest_category_preflight.py"
    source = helper_path.read_text(encoding="utf-8")

    assert "pyimgano.workbench.manifest_category_paths" in source
    assert "inspect_manifest_category_paths" in source
    assert "resolve_manifest_path_best_effort" not in source


def test_manifest_category_paths_module_hosts_path_resolution_checks() -> None:
    helper_path = SRC_DIR / "workbench" / "manifest_category_paths.py"
    source = helper_path.read_text(encoding="utf-8")

    assert "def inspect_manifest_category_paths(" in source
    assert "resolve_manifest_path_best_effort" in source
    assert "MANIFEST_MISSING_IMAGE" in source
    assert "MANIFEST_MISSING_MASK" in source
    assert "MANIFEST_DUPLICATE_IMAGE" in source


def test_manifest_category_preflight_uses_summary_helper_boundary() -> None:
    helper_path = SRC_DIR / "workbench" / "manifest_category_preflight.py"
    source = helper_path.read_text(encoding="utf-8")

    assert "pyimgano.workbench.manifest_category_summary" in source
    assert "summarize_manifest_category_records" in source
    assert "ManifestRecord" not in source


def test_manifest_category_summary_module_hosts_explicit_count_logic() -> None:
    helper_path = SRC_DIR / "workbench" / "manifest_category_summary.py"
    source = helper_path.read_text(encoding="utf-8")

    assert "def summarize_manifest_category_records(" in source
    assert "ManifestRecord" in source
    assert 'counts_by_split = {"train": 0, "val": 0, "test": 0, "unspecified": 0}' in source
    assert 'explicit_test_labels = {"normal": 0, "anomaly": 0}' in source


def test_manifest_category_preflight_uses_assignment_helper_boundary() -> None:
    helper_path = SRC_DIR / "workbench" / "manifest_category_preflight.py"
    source = helper_path.read_text(encoding="utf-8")

    assert "pyimgano.workbench.manifest_category_assignment" in source
    assert "analyze_manifest_category_assignment" in source
    assert "np.random.default_rng" not in source


def test_manifest_category_assignment_module_hosts_group_assignment_logic() -> None:
    helper_path = SRC_DIR / "workbench" / "manifest_category_assignment.py"
    source = helper_path.read_text(encoding="utf-8")

    assert "def analyze_manifest_category_assignment(" in source
    assert "MANIFEST_GROUP_SPLIT_CONFLICT" in source
    assert "MANIFEST_GROUP_ANOMALY_IN_TRAIN" in source
    assert "np.random.default_rng" in source
    assert 'assigned_counts = {"train": 0, "val": 0, "test": 0, "calibration": 0}' in source


def test_benchmark_and_robustness_services_use_dataset_split_service_boundary() -> None:
    violations: list[str] = []

    for file_name in ("benchmark_service.py", "robustness_service.py"):
        path = SERVICE_DIR / file_name
        source = path.read_text(encoding="utf-8")
        if "dataset_split_service" not in source:
            violations.append(f"{file_name}: missing dataset_split_service boundary")
        if "load_manifest_benchmark_split" in source:
            violations.append(f"{file_name}: direct load_manifest_benchmark_split usage")
        if "load_benchmark_split(" in source:
            violations.append(f"{file_name}: direct load_benchmark_split usage")

    assert violations == []


def test_service_modules_only_import_allowed_internal_service_modules() -> None:
    allowed_service_imports = {
        "benchmark_service.py": {
            "pyimgano.services.dataset_split_service",
            "pyimgano.services.model_options",
        },
        "discovery_service.py": set(),
        "doctor_service.py": {
            "pyimgano.services.discovery_service",
            "pyimgano.services.doctor_service_helpers",
        },
        "doctor_service_helpers.py": set(),
        "evaluation_harness_service.py": {
            "pyimgano.services.benchmark_service",
            "pyimgano.services.discovery_service",
            "pyimgano.services.train_service",
        },
        "infer_artifact_service.py": set(),
        "infer_context_service.py": {
            "pyimgano.services.workbench_run_service",
        },
        "infer_continue_service.py": {
            "pyimgano.services.inference_service",
        },
        "infer_defects_defaults_service.py": set(),
        "infer_options_service.py": {
            "pyimgano.services.infer_defects_defaults_service",
        },
        "infer_output_service.py": set(),
        "infer_runtime_service.py": {
            "pyimgano.services.inference_service",
            "pyimgano.services.workbench_adaptation_service",
        },
        "infer_setup_service.py": {
            "pyimgano.services.infer_load_service",
        },
        "infer_load_service.py": {
            "pyimgano.services.infer_context_service",
            "pyimgano.services.model_options",
            "pyimgano.services.workbench_run_service",
        },
        "infer_wrapper_service.py": set(),
        "inference_service.py": set(),
        "model_options.py": set(),
        "pyim_audit_service.py": set(),
        "pyim_payload_collectors.py": {
            "pyimgano.services.discovery_service",
        },
        "pyim_service.py": {
            "pyimgano.services.pyim_payload_collectors",
        },
        "robustness_service.py": {
            "pyimgano.services.benchmark_service",
            "pyimgano.services.dataset_split_service",
            "pyimgano.services.model_options",
        },
        "dataset_split_service.py": set(),
        "train_service.py": {
            "pyimgano.services.train_export_helpers",
            "pyimgano.services.workbench_service",
        },
        "train_export_helpers.py": set(),
        "workbench_run_service.py": set(),
        "workbench_adaptation_service.py": set(),
        "workbench_service.py": {
            "pyimgano.services.model_options",
        },
    }
    violations: list[str] = []

    for path in sorted(SERVICE_DIR.glob("*.py")):
        if path.name == "__init__.py":
            continue
        if path.name not in allowed_service_imports:
            violations.append(f"{path.name}: missing service import whitelist coverage")
            continue

        actual = {item for item in _iter_imports(path) if item.startswith("pyimgano.services.")}
        unexpected = sorted(actual - allowed_service_imports[path.name])
        if unexpected:
            violations.append(f"{path.name}: {', '.join(unexpected)}")

    assert violations == []


def test_service_modules_define_explicit_public_exports() -> None:
    violations: list[str] = []

    for path in sorted(SERVICE_DIR.glob("*.py")):
        if path.name == "__init__.py":
            continue
        try:
            _extract_dunder_all(path)
        except AssertionError as exc:
            violations.append(str(exc))

    assert violations == []


def test_pyim_cli_rendering_uses_section_view_adapter_boundary() -> None:
    rendering_path = SRC_DIR / "pyim_cli_rendering.py"

    violations = [
        imported
        for imported in _iter_imports(rendering_path)
        if imported in {"pyimgano.pyim_contracts", "pyimgano.pyim_list_spec"}
    ]

    assert violations == []


def test_pyim_service_uses_payload_collector_boundary() -> None:
    service_path = SERVICE_DIR / "pyim_service.py"

    violations = [
        imported
        for imported in _iter_imports(service_path)
        if imported
        in {
            "pyimgano.discovery",
            "pyimgano.models.registry",
            "pyimgano.presets.catalog",
            "pyimgano.recipes",
            "pyimgano.recipes.registry",
            "pyimgano.datasets.converters",
            "pyimgano.services.discovery_service",
        }
    ]

    assert violations == []


def test_pyim_cli_uses_app_facade_boundary() -> None:
    cli_path = SRC_DIR / "pyim_cli.py"

    violations = [
        imported
        for imported in _iter_imports(cli_path)
        if imported
        in {
            "pyimgano.cli_output",
            "pyimgano.pyim_contracts",
            "pyimgano.pyim_cli_rendering",
            "pyimgano.services.pyim_service",
            "pyimgano.models.registry",
        }
    ]

    assert violations == []


def test_pyim_app_uses_audit_helpers_boundary() -> None:
    app_path = SRC_DIR / "pyim_app.py"

    violations = [
        imported
        for imported in _iter_imports(app_path)
        if imported
        in {"pyimgano.cli_output", "pyimgano.models.registry", "pyimgano.pyim_contracts"}
    ]

    assert violations == []


def test_pyim_modules_only_import_allowed_internal_modules() -> None:
    allowed_internal_imports = {
        "pyim_cli.py": {
            "pyimgano.pyim_app",
            "pyimgano.pyim_cli_options",
        },
        "pyim_app.py": {
            "pyimgano.pyim_audit_rendering",
            "pyimgano.pyim_cli_options",
            "pyimgano.pyim_cli_rendering",
            "pyimgano.services.pyim_audit_service",
            "pyimgano.services.pyim_service",
        },
        "pyim_cli_options.py": {
            "pyimgano.discovery",
            "pyimgano.pyim_contracts",
            "pyimgano.pyim_list_spec",
        },
        "pyim_cli_rendering.py": {
            "pyimgano.cli_output",
            "pyimgano.pyim_section_views",
        },
        "pyim_audit_rendering.py": {
            "pyimgano.cli_output",
        },
        "pyim_contracts.py": {
            "pyimgano.pyim_list_spec",
        },
        "pyim_list_spec.py": set(),
        "pyim_section_views.py": {
            "pyimgano.pyim_contracts",
            "pyimgano.pyim_list_spec",
        },
        "services/pyim_service.py": {
            "pyimgano.pyim_contracts",
            "pyimgano.services.pyim_payload_collectors",
            "pyimgano.utils.extras",
        },
        "services/pyim_payload_collectors.py": {
            "pyimgano.datasets.converters",
            "pyimgano.discovery",
            "pyimgano.models.registry",
            "pyimgano.presets.catalog",
            "pyimgano.pyim_contracts",
            "pyimgano.pyim_list_spec",
            "pyimgano.recipes",
            "pyimgano.recipes.registry",
            "pyimgano.services.discovery_service",
        },
        "services/pyim_audit_service.py": {
            "pyimgano.models.registry",
        },
    }

    violations: list[str] = []

    for rel_path, allowed in allowed_internal_imports.items():
        path = SRC_DIR / rel_path
        actual = {item for item in _iter_imports(path) if item.startswith("pyimgano.")}
        unexpected = sorted(actual - allowed)
        if unexpected:
            violations.append(f"{rel_path}: {', '.join(unexpected)}")

    assert violations == []


def test_pyim_boundary_modules_define_expected_public_exports() -> None:
    expected_public_exports: dict[str, list[str]] = {
        "pyim_cli.py": ["main"],
        "pyim_app.py": [
            "PyimCommand",
            "run_pyim_command",
        ],
        "pyim_cli_options.py": [
            "PYIM_LIST_KIND_CHOICES",
            "PyimListOptions",
            "resolve_pyim_list_options",
        ],
        "pyim_cli_rendering.py": [
            "emit_pyim_list_payload",
        ],
        "pyim_audit_rendering.py": [
            "emit_pyim_audit_payload",
        ],
        "pyim_contracts.py": [
            "PyimDatasetSummary",
            "PyimListPayload",
            "PyimListRequest",
            "PyimMetadataContractField",
            "PyimModelFacetSummary",
            "PyimPreprocessingSchemeSummary",
            "PyimYearSummary",
        ],
        "pyim_list_spec.py": [
            "ALL_PAYLOAD_FIELDS",
            "CORE_PAYLOAD_FIELDS",
            "PYIM_ALL_TEXT_LIST_KINDS",
            "PYIM_LIST_KIND_CHOICES",
            "PyimListKindSpec",
            "get_pyim_list_kind_spec",
        ],
        "pyim_section_views.py": [
            "PyimListPayloadLike",
            "PyimTextSectionView",
            "iter_pyim_all_text_section_views",
            "resolve_pyim_json_payload",
            "resolve_pyim_text_section_view",
        ],
        "services/pyim_service.py": [
            "PyimListPayload",
            "PyimListRequest",
            "collect_pyim_model_selection_payload",
            "collect_pyim_listing_payload",
        ],
        "services/pyim_payload_collectors.py": [
            "collect_pyim_payload_field",
            "empty_pyim_payload_kwargs",
        ],
        "services/pyim_audit_service.py": [
            "collect_pyim_audit_payload",
        ],
    }
    violations: list[str] = []

    for rel_path, expected_exports in expected_public_exports.items():
        path = SRC_DIR / rel_path
        try:
            actual_exports = _extract_dunder_all(path)
        except AssertionError as exc:
            violations.append(f"{rel_path}: {exc}")
            continue

        if actual_exports != expected_exports:
            violations.append(
                f"{rel_path}: expected __all__={expected_exports}, found {actual_exports}"
            )

    assert violations == []


def test_selected_service_boundary_modules_define_expected_public_exports() -> None:
    expected_public_exports: dict[str, list[str]] = {
        "inference_service.py": [
            "InferenceRunResult",
            "iter_inference_records",
            "run_inference",
        ],
        "model_options.py": [
            "apply_onnx_session_options_shorthand",
            "enforce_checkpoint_requirement",
            "resolve_model_options",
            "resolve_preset_kwargs",
            "resolve_requested_model",
        ],
        "infer_load_service.py": [
            "ConfigBackedInferLoadRequest",
            "ConfigBackedInferLoadResult",
            "DirectInferLoadRequest",
            "DirectInferLoadResult",
            "load_config_backed_infer_detector",
            "load_direct_infer_detector",
        ],
        "workbench_run_service.py": [
            "extract_threshold",
            "load_checkpoint_into_detector",
            "load_report_from_run",
            "load_workbench_config_from_run",
            "resolve_checkpoint_path",
            "select_category_report",
        ],
        "workbench_adaptation_service.py": [
            "build_postprocess_from_payload",
        ],
        "infer_wrapper_service.py": [
            "InferDetectorWrapperRequest",
            "InferDetectorWrapperResult",
            "apply_infer_detector_wrappers",
        ],
        "infer_defects_defaults_service.py": [
            "apply_defects_defaults",
        ],
    }
    violations: list[str] = []

    for rel_path, expected_exports in expected_public_exports.items():
        path = SERVICE_DIR / rel_path
        try:
            actual_exports = _extract_dunder_all(path)
        except AssertionError as exc:
            violations.append(f"{rel_path}: {exc}")
            continue

        if actual_exports != expected_exports:
            violations.append(
                f"{rel_path}: expected __all__={expected_exports}, found {actual_exports}"
            )

    assert violations == []


def test_infer_cli_helper_modules_only_import_allowed_internal_modules() -> None:
    allowed_internal_imports = {
        "infer_cli_inputs.py": set(),
        "infer_cli_discovery.py": {
            "pyimgano.cli_discovery_options",
            "pyimgano.cli_discovery_rendering",
            "pyimgano.cli_listing",
            "pyimgano.services.discovery_service",
        },
        "infer_cli_onnx.py": {
            "pyimgano.features.onnx_embed",
            "pyimgano.infer_cli_inputs",
            "pyimgano.services.model_options",
        },
        "infer_cli_profile.py": set(),
    }
    violations: list[str] = []

    for rel_path, allowed in allowed_internal_imports.items():
        path = SRC_DIR / rel_path
        actual = {item for item in _iter_imports(path) if item.startswith("pyimgano.")}
        unexpected = sorted(actual - allowed)
        if unexpected:
            violations.append(f"{rel_path}: {', '.join(unexpected)}")

    assert violations == []


def test_infer_cli_helper_modules_define_expected_public_exports() -> None:
    expected_public_exports: dict[str, list[str]] = {
        "infer_cli_inputs.py": [
            "IMAGE_SUFFIXES",
            "collect_image_paths",
            "parse_csv_ints_arg",
            "parse_csv_strs_arg",
            "parse_json_mapping_arg",
        ],
        "infer_cli_discovery.py": [
            "maybe_run_infer_discovery_command",
        ],
        "infer_cli_onnx.py": [
            "apply_onnx_session_options_shorthand",
            "default_onnx_sweep_intra_values",
            "extract_onnx_checkpoint_path_for_sweep",
            "extract_session_options_for_sweep",
            "maybe_apply_onnx_session_options_and_sweep",
            "run_onnx_session_options_sweep",
        ],
        "infer_cli_profile.py": [
            "build_infer_profile_payload",
            "format_infer_profile_summary",
            "write_infer_profile_payload",
        ],
    }
    violations: list[str] = []

    for rel_path, expected_exports in expected_public_exports.items():
        path = SRC_DIR / rel_path
        try:
            actual_exports = _extract_dunder_all(path)
        except AssertionError as exc:
            violations.append(f"{rel_path}: {exc}")
            continue

        if actual_exports != expected_exports:
            violations.append(
                f"{rel_path}: expected __all__={expected_exports}, found {actual_exports}"
            )

    assert violations == []


def test_inference_helper_modules_define_expected_public_exports() -> None:
    expected_public_exports: dict[str, list[str]] = {
        "inference/decision_summary.py": [
            "build_decision_summary",
            "maybe_build_decision_summary",
        ],
        "inference/runtime_support.py": [
            "ImageInput",
            "apply_rejection_policy",
            "best_effort_label_confidence",
            "normalize_inputs",
            "resolve_rejection_threshold",
        ],
    }
    violations: list[str] = []

    for rel_path, expected_exports in expected_public_exports.items():
        path = SRC_DIR / rel_path
        try:
            actual_exports = _extract_dunder_all(path)
        except AssertionError as exc:
            violations.append(f"{rel_path}: {exc}")
            continue

        if actual_exports != expected_exports:
            violations.append(
                f"{rel_path}: expected __all__={expected_exports}, found {actual_exports}"
            )

    assert violations == []


def test_runs_helper_modules_define_expected_public_exports() -> None:
    expected_public_exports: dict[str, list[str]] = {
        "reporting/run_index_helpers.py": [
            "build_trust_comparison",
            "bundle_operator_contract_status_from_trust_summary",
            "comparability_gate_status",
            "compare_blocking_flags",
            "comparison_trust_gate",
            "comparison_trust_reason",
            "format_candidate_incompatibility_digest",
            "format_metric_value",
            "operator_contract_status_from_trust_summary",
        ],
        "runs_cli_rendering.py": [
            "format_acceptance_run_summary_line",
            "format_compare_run_brief_line",
            "format_publication_summary_line",
            "format_quality_summary_line",
            "format_run_brief_line",
        ],
    }
    violations: list[str] = []

    for rel_path, expected_exports in expected_public_exports.items():
        path = SRC_DIR / rel_path
        try:
            actual_exports = _extract_dunder_all(path)
        except AssertionError as exc:
            violations.append(f"{rel_path}: {exc}")
            continue

        if actual_exports != expected_exports:
            violations.append(
                f"{rel_path}: expected __all__={expected_exports}, found {actual_exports}"
            )

    assert violations == []


def test_doctor_helper_modules_define_expected_public_exports() -> None:
    expected_public_exports: dict[str, list[str]] = {
        "doctor_rendering.py": [
            "format_extra_recommendation_lines",
            "format_readiness_lines",
            "format_require_extras_line",
            "format_suite_check_line",
        ],
        "services/doctor_service_helpers.py": [
            "build_accelerator_checks",
            "build_require_extras_check",
            "split_csv_args",
        ],
    }
    violations: list[str] = []

    for rel_path, expected_exports in expected_public_exports.items():
        path = SRC_DIR / rel_path
        try:
            actual_exports = _extract_dunder_all(path)
        except AssertionError as exc:
            violations.append(f"{rel_path}: {exc}")
            continue

        if actual_exports != expected_exports:
            violations.append(
                f"{rel_path}: expected __all__={expected_exports}, found {actual_exports}"
            )

    assert violations == []


def test_bundle_helper_modules_define_expected_public_exports() -> None:
    expected_public_exports: dict[str, list[str]] = {
        "bundle_rendering.py": [
            "format_bundle_run_lines",
            "format_bundle_validate_lines",
        ],
        "bundle_cli_helpers.py": [
            "build_batch_gate_summary",
            "build_input_source_summary",
            "build_reason_codes",
            "run_exit_code",
            "validate_exit_code",
        ],
    }
    violations: list[str] = []

    for rel_path, expected_exports in expected_public_exports.items():
        path = SRC_DIR / rel_path
        try:
            actual_exports = _extract_dunder_all(path)
        except AssertionError as exc:
            violations.append(f"{rel_path}: {exc}")
            continue

        if actual_exports != expected_exports:
            violations.append(
                f"{rel_path}: expected __all__={expected_exports}, found {actual_exports}"
            )

    assert violations == []


def test_train_export_helper_modules_define_expected_public_exports() -> None:
    expected_public_exports: dict[str, list[str]] = {
        "services/train_export_helpers.py": [
            "apply_bundle_manifest_metadata",
            "build_optional_calibration_card_payload",
            "require_run_dir",
            "rewrite_bundle_paths",
            "validate_export_request",
        ],
    }
    violations: list[str] = []

    for rel_path, expected_exports in expected_public_exports.items():
        path = SRC_DIR / rel_path
        try:
            actual_exports = _extract_dunder_all(path)
        except AssertionError as exc:
            violations.append(f"{rel_path}: {exc}")
            continue

        if actual_exports != expected_exports:
            violations.append(
                f"{rel_path}: expected __all__={expected_exports}, found {actual_exports}"
            )

    assert violations == []


def test_preflight_helper_modules_define_expected_public_exports() -> None:
    expected_public_exports: dict[str, list[str]] = {
        "workbench/preflight_dispatch.py": [
            "resolve_preflight_dataset_dispatch",
        ],
        "workbench/manifest_preflight_flow.py": [
            "resolve_manifest_preflight_source_or_summary",
            "resolve_manifest_record_preflight_summary",
        ],
        "workbench/non_manifest_preflight_flow.py": [
            "resolve_non_manifest_category_listing_summary",
            "resolve_non_manifest_preflight_source_or_summary",
        ],
    }
    violations: list[str] = []

    for rel_path, expected_exports in expected_public_exports.items():
        path = SRC_DIR / rel_path
        try:
            actual_exports = _extract_dunder_all(path)
        except AssertionError as exc:
            violations.append(f"{rel_path}: {exc}")
            continue

        if actual_exports != expected_exports:
            violations.append(
                f"{rel_path}: expected __all__={expected_exports}, found {actual_exports}"
            )

    assert violations == []


def test_deploy_bundle_validation_helper_module_define_expected_public_exports() -> None:
    expected_public_exports: dict[str, list[str]] = {
        "reporting/deploy_bundle_contract_helpers.py": [
            "build_artifact_digests",
            "build_artifact_roles",
            "collect_existing_artifact_refs",
            "required_artifacts_present",
        ],
        "reporting/deploy_bundle_validation_helpers.py": [
            "append_operator_contract_presence_errors",
            "operator_contract_audit_state",
            "source_run_context",
            "validate_artifact_refs",
            "validate_exact_mapping",
            "validate_operator_contract_consistency",
            "validate_operator_contract_digests_map",
            "validate_required_presence_flag",
            "validate_weight_audit_files",
        ],
    }
    violations: list[str] = []

    for rel_path, expected_exports in expected_public_exports.items():
        path = SRC_DIR / rel_path
        try:
            actual_exports = _extract_dunder_all(path)
        except AssertionError as exc:
            violations.append(f"{rel_path}: {exc}")
            continue

        if actual_exports != expected_exports:
            violations.append(
                f"{rel_path}: expected __all__={expected_exports}, found {actual_exports}"
            )

    assert violations == []
