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
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "__all__":
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


def test_service_modules_only_import_allowed_internal_service_modules() -> None:
    allowed_service_imports = {
        "benchmark_service.py": {
            "pyimgano.services.model_options",
        },
        "discovery_service.py": set(),
        "doctor_service.py": {
            "pyimgano.services.discovery_service",
        },
        "infer_artifact_service.py": set(),
        "infer_context_service.py": {
            "pyimgano.services.workbench_run_service",
        },
        "infer_continue_service.py": {
            "pyimgano.services.inference_service",
        },
        "infer_options_service.py": set(),
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
            "pyimgano.services.model_options",
        },
        "train_service.py": {
            "pyimgano.services.workbench_service",
        },
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

        actual = {
            item
            for item in _iter_imports(path)
            if item.startswith("pyimgano.services.")
        }
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
        if imported in {"pyimgano.cli_output", "pyimgano.models.registry", "pyimgano.pyim_contracts"}
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
