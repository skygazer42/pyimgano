from __future__ import annotations

"""Public contracts for trained artifact manifests, policy, and safe staging."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pyimgano.artifacts.compatibility import (
    ArtifactCompatibilityError,
    ParsedCompatibilityRequirements,
    RuntimeCompatibilityReport,
    current_platform_tag,
    normalize_platform_tag,
    onnxruntime_requirement_for_graph,
    parse_compatibility_requirements,
    preflight_artifact_compatibility,
)
from pyimgano.artifacts.export_index import (
    EXPORT_INDEX_FILENAME,
    EXPORT_INDEX_SCHEMA_FAMILY,
    EXPORT_INDEX_SCHEMA_VERSION,
    ExportIndexError,
    build_export_index,
    category_slug,
    load_export_index,
    validate_export_index,
    write_export_index,
)
from pyimgano.artifacts.importers import import_onnx
from pyimgano.artifacts.io_contract import (
    MAX_IMAGE_DIMENSION,
    ArtifactIOContractError,
    validate_artifact_input_contract,
    validate_artifact_output_contract,
)
from pyimgano.artifacts.manifest import (
    ARTIFACT_MANIFEST_FILENAME,
    ARTIFACT_POLICY_SCHEMA_FAMILY,
    ARTIFACT_POLICY_SCHEMA_VERSION,
    ARTIFACT_SCHEMA_FAMILY,
    ARTIFACT_SCHEMA_VERSION,
    ArtifactManifestError,
    build_artifact_manifest,
    canonical_json_bytes,
    compute_artifact_id,
    compute_policy_id,
    compute_runtime_id,
    load_artifact_manifest,
    validate_artifact_manifest,
    write_artifact_manifest,
)
from pyimgano.artifacts.onnx_graph import (
    ONNXGraphContractInfo,
    validate_onnx_graph_contract,
    validate_onnx_model_contract,
)
from pyimgano.artifacts.policy import (
    ArtifactPolicyError,
    bind_policy,
    validate_artifact_policy,
    write_artifact_policy,
)
from pyimgano.artifacts.security import (
    ArtifactSecurityError,
    VerifiedArtifactStaging,
    resolve_contained_path,
    stage_verified_artifact,
    verify_artifact_files,
    verify_file,
)


def export_run(
    run_dir: str | Path,
    *,
    formats: Sequence[str] = ("native",),
    out: str | Path | None = None,
    category: str | None = None,
    verification_level: str = "reference_parity",
    strict: bool = True,
    trust_checkpoint: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Export executable fitted-detector artifacts from one persisted run."""

    from pyimgano.services.export_service import export_from_run

    return export_from_run(
        run_dir=run_dir,
        formats=formats,
        out_dir=out,
        category=category,
        verification_level=verification_level,
        strict=strict,
        trust_checkpoint=trust_checkpoint,
        overwrite=overwrite,
    )


__all__ = [
    "ARTIFACT_MANIFEST_FILENAME",
    "ARTIFACT_POLICY_SCHEMA_FAMILY",
    "ARTIFACT_POLICY_SCHEMA_VERSION",
    "ARTIFACT_SCHEMA_FAMILY",
    "ARTIFACT_SCHEMA_VERSION",
    "EXPORT_INDEX_FILENAME",
    "EXPORT_INDEX_SCHEMA_FAMILY",
    "EXPORT_INDEX_SCHEMA_VERSION",
    "ArtifactManifestError",
    "ArtifactCompatibilityError",
    "ArtifactIOContractError",
    "ONNXGraphContractInfo",
    "MAX_IMAGE_DIMENSION",
    "ParsedCompatibilityRequirements",
    "RuntimeCompatibilityReport",
    "ArtifactPolicyError",
    "ArtifactSecurityError",
    "ExportIndexError",
    "VerifiedArtifactStaging",
    "bind_policy",
    "build_export_index",
    "build_artifact_manifest",
    "canonical_json_bytes",
    "compute_artifact_id",
    "compute_policy_id",
    "compute_runtime_id",
    "current_platform_tag",
    "category_slug",
    "export_run",
    "import_onnx",
    "load_artifact_manifest",
    "load_export_index",
    "normalize_platform_tag",
    "onnxruntime_requirement_for_graph",
    "parse_compatibility_requirements",
    "preflight_artifact_compatibility",
    "resolve_contained_path",
    "stage_verified_artifact",
    "validate_artifact_manifest",
    "validate_artifact_input_contract",
    "validate_artifact_output_contract",
    "validate_artifact_policy",
    "validate_export_index",
    "validate_onnx_graph_contract",
    "validate_onnx_model_contract",
    "verify_artifact_files",
    "verify_file",
    "write_artifact_manifest",
    "write_artifact_policy",
    "write_export_index",
]
