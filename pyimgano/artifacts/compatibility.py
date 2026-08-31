from __future__ import annotations

import platform as platform_module
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from types import MappingProxyType
from typing import Any

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version


class ArtifactCompatibilityError(ValueError):
    """Raised when an artifact cannot run in the current environment."""


@dataclass(frozen=True)
class ParsedCompatibilityRequirements:
    pyimgano: SpecifierSet
    python: SpecifierSet
    runtime_versions: Mapping[str, SpecifierSet]


@dataclass(frozen=True)
class RuntimeCompatibilityReport:
    pyimgano_version: str
    python_version: str
    platform_tag: str
    runtime_versions: Mapping[str, str]


RuntimeVersionResolver = Callable[[str], str]

_RUNTIME_DISTRIBUTIONS = {
    "onnxruntime": ("onnxruntime", "onnxruntime-gpu"),
    "openvino": ("openvino",),
    "torch": ("torch",),
}
_BACKEND_RUNTIME_KEYS = {
    "onnxruntime": frozenset({"onnxruntime"}),
    "openvino": frozenset({"openvino"}),
    "torchscript": frozenset({"torch"}),
}
_COMPONENT_RUNTIME_KEYS = {
    "onnx": "onnxruntime",
    "openvino-ir": "openvino",
    "torchscript": "torch",
}
_RUNTIME_EXTRAS = {
    "onnxruntime": "onnx-runtime",
    "openvino": "openvino-runtime",
    "torch": "torch",
}
_OPERATING_SYSTEM_ALIASES = {
    "cygwin": "windows",
    "darwin": "macos",
    "linux": "linux",
    "linux2": "linux",
    "mac": "macos",
    "macos": "macos",
    "macosx": "macos",
    "msys": "windows",
    "win32": "windows",
    "windows": "windows",
}
_ARCHITECTURE_ALIASES = {
    "aarch64": "arm64",
    "amd64": "x86_64",
    "arm64": "arm64",
    "i386": "x86",
    "i486": "x86",
    "i586": "x86",
    "i686": "x86",
    "x64": "x86_64",
    "x86": "x86",
    "x86-64": "x86_64",
    "x86_64": "x86_64",
}
_SAFE_PLATFORM_PART = re.compile(r"^[a-z0-9_]+$")


def onnxruntime_requirement_for_graph(*, ir_version: int, default_opset: int) -> str:
    """Return the conservative ORT line required by the supported ONNX graph."""

    for field, value in (("ir_version", ir_version), ("default_opset", default_opset)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ArtifactCompatibilityError(f"{field} must be a positive integer.")
    # ORT's compatibility table maps 1.17 to opset 20 / IR 9 and 1.18 to
    # opset 21 / IR 10. The graph validator independently enforces the full
    # accepted IR/opset range and standard domains.
    if ir_version >= 10 or default_opset >= 21:
        return ">=1.18,<2"
    return ">=1.17,<2"


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactCompatibilityError(f"{field} must be a mapping.")
    return value


def _required_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ArtifactCompatibilityError(f"{field} must be a non-empty string.")
    return value.strip()


def _parse_specifier(value: Any, *, field: str) -> SpecifierSet:
    requirement = _required_string(value, field=field)
    try:
        return SpecifierSet(requirement)
    except InvalidSpecifier as exc:
        raise ArtifactCompatibilityError(
            f"{field} has invalid version specifier {requirement!r}: {exc}"
        ) from exc


def parse_compatibility_requirements(
    compatibility: Mapping[str, Any],
) -> ParsedCompatibilityRequirements:
    """Strictly parse every semantic-version requirement in a manifest block."""

    value = _mapping(compatibility, field="compatibility")
    runtime_values = _mapping(
        value.get("runtime_versions", {}), field="compatibility.runtime_versions"
    )
    unknown_runtime_keys = sorted(
        str(key) for key in runtime_values if str(key) not in _RUNTIME_DISTRIBUTIONS
    )
    if unknown_runtime_keys:
        raise ArtifactCompatibilityError(
            "compatibility.runtime_versions contains unsupported runtime keys: "
            f"{unknown_runtime_keys!r}."
        )

    runtime_requirements: dict[str, SpecifierSet] = {}
    for raw_key, requirement in runtime_values.items():
        key = _required_string(raw_key, field="compatibility.runtime_versions.<key>")
        runtime_requirements[key] = _parse_specifier(
            requirement, field=f"compatibility.runtime_versions.{key}"
        )
    return ParsedCompatibilityRequirements(
        pyimgano=_parse_specifier(value.get("pyimgano"), field="compatibility.pyimgano"),
        python=_parse_specifier(value.get("python"), field="compatibility.python"),
        runtime_versions=MappingProxyType(runtime_requirements),
    )


def normalize_platform_tag(value: str) -> str:
    """Normalize supported OS/architecture aliases to a stable manifest tag."""

    raw = _required_string(value, field="platform tag").lower()
    operating_system, separator, architecture = raw.partition("-")
    if not separator or not architecture:
        raise ArtifactCompatibilityError(
            f"Platform tag {value!r} must use the '<os>-<architecture>' form."
        )
    normalized_os = _OPERATING_SYSTEM_ALIASES.get(operating_system)
    if normalized_os is None:
        raise ArtifactCompatibilityError(
            f"Platform tag {value!r} uses unsupported operating system {operating_system!r}."
        )
    normalized_architecture = _ARCHITECTURE_ALIASES.get(architecture, architecture)
    if not _SAFE_PLATFORM_PART.fullmatch(normalized_architecture):
        raise ArtifactCompatibilityError(
            f"Platform tag {value!r} has an invalid architecture {architecture!r}."
        )
    return f"{normalized_os}-{normalized_architecture}"


def current_platform_tag(*, system: str | None = None, machine: str | None = None) -> str:
    """Return the normalized platform tag for the running interpreter."""

    current_system = str(system if system is not None else sys.platform)
    current_machine = str(machine if machine is not None else platform_module.machine())
    return normalize_platform_tag(f"{current_system}-{current_machine}")


def _expected_runtime_keys(
    manifest: Mapping[str, Any],
    *,
    declared_runtime_keys: frozenset[str],
) -> frozenset[str]:
    runtime = _mapping(manifest.get("runtime"), field="runtime")
    backend = _required_string(runtime.get("backend"), field="runtime.backend").lower()
    if backend in _BACKEND_RUNTIME_KEYS:
        return _BACKEND_RUNTIME_KEYS[backend]
    if backend != "pyimgano":
        raise ArtifactCompatibilityError(
            f"Cannot evaluate compatibility for unsupported runtime backend {backend!r}."
        )

    layout = _required_string(manifest.get("layout"), field="layout").lower()
    if layout != "composite":
        # Native adapters own their optional runtime requirements. Pure-Python
        # adapters may declare none; adapters backed by Torch may declare Torch.
        return declared_runtime_keys
    components = manifest.get("components")
    if not isinstance(components, Sequence) or isinstance(components, (str, bytes)):
        raise ArtifactCompatibilityError("components must be a sequence.")
    required: set[str] = set()
    for component in components:
        if not isinstance(component, Mapping) or component.get("role") != "runtime_model":
            continue
        component_format = _required_string(
            component.get("format"), field="components.runtime_model.format"
        ).lower()
        try:
            required.add(_COMPONENT_RUNTIME_KEYS[component_format])
        except KeyError as exc:
            raise ArtifactCompatibilityError(
                "Cannot evaluate compatibility for composite runtime-model format "
                f"{component_format!r}."
            ) from exc
    if not required:
        raise ArtifactCompatibilityError(
            "Composite artifacts must declare at least one supported runtime-model component."
        )
    return frozenset(required)


def _checked_version(value: Any, *, field: str) -> tuple[str, Version]:
    current = _required_string(value, field=field)
    try:
        return current, Version(current)
    except InvalidVersion as exc:
        raise ArtifactCompatibilityError(
            f"{field} reports invalid installed version {current!r}: {exc}"
        ) from exc


def _require_version_match(
    *,
    field: str,
    current: Any,
    requirement: SpecifierSet,
    remediation: str,
) -> str:
    current_text, parsed = _checked_version(current, field=field)
    if not requirement.contains(parsed, prereleases=True):
        raise ArtifactCompatibilityError(
            f"Artifact compatibility check failed for {field}: current version "
            f"{current_text!r} does not satisfy {str(requirement)!r}. {remediation}"
        )
    return current_text


def _declared_minimum_version(requirement: SpecifierSet, *, field: str) -> Version:
    candidates: list[Version] = []
    for specifier in requirement:
        operator = str(specifier.operator)
        if operator not in {">=", ">", "~=", "=="}:
            continue
        version_text = str(specifier.version)
        if operator == "==" and version_text.endswith(".*"):
            version_text = version_text[:-2]
        try:
            candidates.append(Version(version_text))
        except InvalidVersion as exc:
            raise ArtifactCompatibilityError(
                f"{field} has an unsupported lower-bound form: {specifier!s}."
            ) from exc
    if not candidates:
        raise ArtifactCompatibilityError(f"{field} must declare a finite lower bound.")
    return max(candidates)


def _loaded_pyimgano_version() -> str:
    from pyimgano import __version__

    return str(__version__)


def preflight_artifact_compatibility(
    manifest: Mapping[str, Any],
    *,
    pyimgano_version: str | None = None,
    python_version: str | None = None,
    platform_tag: str | None = None,
    runtime_version_resolver: RuntimeVersionResolver | None = None,
) -> RuntimeCompatibilityReport:
    """Fail closed when a validated artifact is incompatible with this runtime.

    The function performs metadata-only checks and never imports or constructs a
    backend runtime. Callers should invoke it before staging executable sessions.
    """

    payload = _mapping(manifest, field="manifest")
    compatibility = _mapping(payload.get("compatibility"), field="compatibility")
    requirements = parse_compatibility_requirements(compatibility)

    onnx_ir = compatibility.get("onnx_ir")
    onnx_opset = compatibility.get("onnx_opset")
    if (onnx_ir is None) != (onnx_opset is None):
        raise ArtifactCompatibilityError(
            "compatibility.onnx_ir and compatibility.onnx_opset must be declared together."
        )

    current_pyimgano = _require_version_match(
        field="compatibility.pyimgano",
        current=(pyimgano_version if pyimgano_version is not None else _loaded_pyimgano_version()),
        requirement=requirements.pyimgano,
        remediation="Install a compatible pyimgano release.",
    )
    current_python = _require_version_match(
        field="compatibility.python",
        current=(
            python_version if python_version is not None else platform_module.python_version()
        ),
        requirement=requirements.python,
        remediation="Use a compatible Python interpreter.",
    )

    declared_platforms = compatibility.get("platforms")
    if not isinstance(declared_platforms, Sequence) or isinstance(declared_platforms, (str, bytes)):
        raise ArtifactCompatibilityError("compatibility.platforms must be a sequence.")
    if not declared_platforms:
        raise ArtifactCompatibilityError("compatibility.platforms must not be empty.")
    allowed_platforms = frozenset(
        normalize_platform_tag(_required_string(item, field="compatibility.platforms[]"))
        for item in declared_platforms
    )
    current_platform = normalize_platform_tag(
        platform_tag if platform_tag is not None else current_platform_tag()
    )
    if current_platform not in allowed_platforms:
        raise ArtifactCompatibilityError(
            f"Artifact does not support current platform {current_platform!r}; declared "
            f"platforms are {sorted(allowed_platforms)!r}. Use a supported platform or artifact."
        )

    declared_runtime_keys = frozenset(requirements.runtime_versions)
    expected_runtime_keys = _expected_runtime_keys(
        payload,
        declared_runtime_keys=declared_runtime_keys,
    )
    missing_runtime_keys = sorted(expected_runtime_keys - declared_runtime_keys)
    if missing_runtime_keys:
        raise ArtifactCompatibilityError(
            f"Artifact backend requires runtime version declarations {missing_runtime_keys!r}."
        )
    unexpected_runtime_keys = sorted(declared_runtime_keys - expected_runtime_keys)
    if unexpected_runtime_keys:
        raise ArtifactCompatibilityError(
            "Artifact declares runtime versions unrelated to its executable components: "
            f"{unexpected_runtime_keys!r}."
        )
    if onnx_ir is not None and "onnxruntime" in expected_runtime_keys:
        minimum = onnxruntime_requirement_for_graph(
            ir_version=onnx_ir,
            default_opset=onnx_opset,
        )
        minimum_version = Version("1.18" if ">=1.18" in minimum else "1.17")
        declared_ort = requirements.runtime_versions["onnxruntime"]
        declared_minimum = _declared_minimum_version(
            declared_ort,
            field="compatibility.runtime_versions.onnxruntime",
        )
        if declared_minimum < minimum_version:
            raise ArtifactCompatibilityError(
                "compatibility.runtime_versions.onnxruntime has a lower bound older than "
                f"the graph metadata requires ({minimum})."
            )

    resolve_runtime_version = runtime_version_resolver or metadata.version
    current_runtime_versions: dict[str, str] = {}
    for runtime_key in sorted(expected_runtime_keys):
        distributions = _RUNTIME_DISTRIBUTIONS[runtime_key]
        requirement = requirements.runtime_versions[runtime_key]
        installed_version: str | None = None
        package_not_found: metadata.PackageNotFoundError | None = None
        for distribution in distributions:
            try:
                installed_version = resolve_runtime_version(distribution)
            except metadata.PackageNotFoundError as exc:
                package_not_found = exc
                continue
            break
        if installed_version is None:
            extra = _RUNTIME_EXTRAS[runtime_key]
            raise ArtifactCompatibilityError(
                f"Required backend runtime {runtime_key!r} is not installed; artifact requires "
                f"{str(requirement)!r}. Checked distributions {list(distributions)!r}. "
                f"Install 'pyimgano[{extra}]'."
            ) from package_not_found
        current_runtime_versions[runtime_key] = _require_version_match(
            field=f"compatibility.runtime_versions.{runtime_key}",
            current=installed_version,
            requirement=requirement,
            remediation=f"Install 'pyimgano[{_RUNTIME_EXTRAS[runtime_key]}]'.",
        )

    return RuntimeCompatibilityReport(
        pyimgano_version=current_pyimgano,
        python_version=current_python,
        platform_tag=current_platform,
        runtime_versions=MappingProxyType(current_runtime_versions),
    )


__all__ = [
    "ArtifactCompatibilityError",
    "ParsedCompatibilityRequirements",
    "RuntimeCompatibilityReport",
    "RuntimeVersionResolver",
    "current_platform_tag",
    "normalize_platform_tag",
    "onnxruntime_requirement_for_graph",
    "parse_compatibility_requirements",
    "preflight_artifact_compatibility",
]
