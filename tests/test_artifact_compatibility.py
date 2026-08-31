from __future__ import annotations

from importlib import metadata
from unittest.mock import Mock

import pytest

from pyimgano.artifacts.compatibility import (
    ArtifactCompatibilityError,
    current_platform_tag,
    normalize_platform_tag,
    onnxruntime_requirement_for_graph,
    parse_compatibility_requirements,
    preflight_artifact_compatibility,
)


def _manifest(
    *,
    backend: str = "onnxruntime",
    runtime_versions: dict[str, str] | None = None,
    platforms: list[str] | None = None,
) -> dict[str, object]:
    return {
        "layout": "single_graph",
        "runtime": {"backend": backend},
        "components": [
            {
                "role": "runtime_model",
                "format": {
                    "onnxruntime": "onnx",
                    "openvino": "openvino-ir",
                    "torchscript": "torchscript",
                }.get(backend, "native"),
            }
        ],
        "compatibility": {
            "pyimgano": ">=0.10,<0.11",
            "python": ">=3.9,<3.13",
            "platforms": platforms or ["linux-x86_64"],
            "runtime_versions": (
                runtime_versions if runtime_versions is not None else {"onnxruntime": ">=1.17,<2"}
            ),
        },
    }


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("pyimgano", "not-a-specifier", "compatibility.pyimgano"),
        ("python", ">=3.9,wat", "compatibility.python"),
    ],
)
def test_parse_rejects_malformed_primary_specifiers(field, value, match) -> None:  # noqa: ANN001
    compatibility = _manifest()["compatibility"]
    assert isinstance(compatibility, dict)
    compatibility[field] = value

    with pytest.raises(ArtifactCompatibilityError, match=match):
        parse_compatibility_requirements(compatibility)


def test_parse_rejects_malformed_runtime_specifier() -> None:
    compatibility = _manifest(runtime_versions={"onnxruntime": "definitely-not-a-range"})[
        "compatibility"
    ]
    assert isinstance(compatibility, dict)

    with pytest.raises(ArtifactCompatibilityError, match="runtime_versions.onnxruntime"):
        parse_compatibility_requirements(compatibility)


def test_preflight_rejects_future_python_before_runtime_resolution() -> None:
    manifest = _manifest()
    compatibility = manifest["compatibility"]
    assert isinstance(compatibility, dict)
    compatibility["python"] = ">=99"
    resolver = Mock(return_value="1.18.0")

    with pytest.raises(ArtifactCompatibilityError, match="compatibility.python"):
        preflight_artifact_compatibility(
            manifest,
            pyimgano_version="0.10.0",
            python_version="3.11.9",
            platform_tag="linux-x86_64",
            runtime_version_resolver=resolver,
        )

    resolver.assert_not_called()


def test_preflight_rejects_incompatible_pyimgano_version() -> None:
    with pytest.raises(ArtifactCompatibilityError, match="compatible pyimgano"):
        preflight_artifact_compatibility(
            _manifest(),
            pyimgano_version="0.11.0",
            python_version="3.11.9",
            platform_tag="linux-x86_64",
            runtime_version_resolver=lambda _name: "1.18.0",
        )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("linux-amd64", "linux-x86_64"),
        ("darwin-aarch64", "macos-arm64"),
        ("win32-x64", "windows-x86_64"),
    ],
)
def test_platform_aliases_are_canonical(value: str, expected: str) -> None:
    assert normalize_platform_tag(value) == expected


def test_current_platform_tag_normalizes_system_and_machine_aliases() -> None:
    assert current_platform_tag(system="Darwin", machine="aarch64") == "macos-arm64"


@pytest.mark.parametrize(
    ("ir_version", "opset", "expected"),
    [
        (9, 20, ">=1.17,<2"),
        (10, 20, ">=1.18,<2"),
        (9, 21, ">=1.18,<2"),
    ],
)
def test_onnxruntime_requirement_tracks_graph_metadata(
    ir_version: int, opset: int, expected: str
) -> None:
    assert (
        onnxruntime_requirement_for_graph(
            ir_version=ir_version,
            default_opset=opset,
        )
        == expected
    )


def test_preflight_rejects_underdeclared_onnxruntime_for_graph_metadata() -> None:
    manifest = _manifest(runtime_versions={"onnxruntime": ">=1.17,<2"})
    compatibility = manifest["compatibility"]
    assert isinstance(compatibility, dict)
    compatibility.update({"onnx_ir": 10, "onnx_opset": 21})

    with pytest.raises(ArtifactCompatibilityError, match="lower bound older"):
        preflight_artifact_compatibility(
            manifest,
            pyimgano_version="0.10.0",
            python_version="3.11.9",
            platform_tag="linux-x86_64",
            runtime_version_resolver=lambda _name: "1.18.0",
        )


def test_preflight_accepts_equivalent_platform_alias() -> None:
    report = preflight_artifact_compatibility(
        _manifest(platforms=["linux-amd64"]),
        pyimgano_version="0.10.0",
        python_version="3.11.9",
        platform_tag="linux-x86_64",
        runtime_version_resolver=lambda _name: "1.18.0",
    )

    assert report.platform_tag == "linux-x86_64"
    assert dict(report.runtime_versions) == {"onnxruntime": "1.18.0"}


def test_preflight_rejects_platform_mismatch() -> None:
    with pytest.raises(ArtifactCompatibilityError, match="current platform 'linux-x86_64'"):
        preflight_artifact_compatibility(
            _manifest(platforms=["macos-arm64"]),
            pyimgano_version="0.10.0",
            python_version="3.11.9",
            platform_tag="linux-amd64",
            runtime_version_resolver=lambda _name: "1.18.0",
        )


def test_preflight_rejects_unknown_runtime_key() -> None:
    with pytest.raises(ArtifactCompatibilityError, match="unsupported runtime keys.*mystery"):
        preflight_artifact_compatibility(
            _manifest(runtime_versions={"mystery": ">=1"}),
            pyimgano_version="0.10.0",
            python_version="3.11.9",
            platform_tag="linux-x86_64",
        )


def test_preflight_requires_backend_runtime_declaration() -> None:
    with pytest.raises(ArtifactCompatibilityError, match="requires runtime version declarations"):
        preflight_artifact_compatibility(
            _manifest(runtime_versions={}),
            pyimgano_version="0.10.0",
            python_version="3.11.9",
            platform_tag="linux-x86_64",
        )


def test_preflight_rejects_uninstalled_backend_runtime() -> None:
    def missing(name: str) -> str:
        raise metadata.PackageNotFoundError(name)

    with pytest.raises(ArtifactCompatibilityError, match="not installed.*onnx-runtime"):
        preflight_artifact_compatibility(
            _manifest(),
            pyimgano_version="0.10.0",
            python_version="3.11.9",
            platform_tag="linux-x86_64",
            runtime_version_resolver=missing,
        )


def test_preflight_rejects_backend_runtime_version_mismatch() -> None:
    with pytest.raises(
        ArtifactCompatibilityError,
        match=r"runtime_versions\.onnxruntime.*1\.16\.0.*<2,>=1\.17",
    ):
        preflight_artifact_compatibility(
            _manifest(),
            pyimgano_version="0.10.0",
            python_version="3.11.9",
            platform_tag="linux-x86_64",
            runtime_version_resolver=lambda _name: "1.16.0",
        )


@pytest.mark.parametrize(
    ("backend", "runtime_key", "version", "requirement"),
    [
        ("torchscript", "torch", "2.6.0+cu124", ">=1.9"),
        ("openvino", "openvino", "2025.2.0", ">=2023,<2027"),
    ],
)
def test_preflight_checks_each_supported_graph_backend(
    backend: str, runtime_key: str, version: str, requirement: str
) -> None:
    report = preflight_artifact_compatibility(
        _manifest(backend=backend, runtime_versions={runtime_key: requirement}),
        pyimgano_version="0.10.0",
        python_version="3.11.9",
        platform_tag="linux-x86_64",
        runtime_version_resolver=lambda name: version if name == runtime_key else "0",
    )

    assert dict(report.runtime_versions) == {runtime_key: version}


def test_preflight_accepts_onnxruntime_gpu_distribution() -> None:
    resolver = Mock(side_effect=[metadata.PackageNotFoundError("onnxruntime"), "1.18.1"])

    report = preflight_artifact_compatibility(
        _manifest(),
        pyimgano_version="0.10.0",
        python_version="3.11.9",
        platform_tag="linux-x86_64",
        runtime_version_resolver=resolver,
    )

    assert dict(report.runtime_versions) == {"onnxruntime": "1.18.1"}
    assert [call.args[0] for call in resolver.call_args_list] == [
        "onnxruntime",
        "onnxruntime-gpu",
    ]


def test_native_artifact_requires_no_optional_backend_distribution() -> None:
    manifest = _manifest(backend="pyimgano", runtime_versions={})
    manifest["layout"] = "native_detector"
    resolver = Mock(side_effect=AssertionError("native preflight must not resolve a backend"))

    report = preflight_artifact_compatibility(
        manifest,
        pyimgano_version="0.10.0",
        python_version="3.11.9",
        platform_tag="linux-x86_64",
        runtime_version_resolver=resolver,
    )

    assert dict(report.runtime_versions) == {}
    resolver.assert_not_called()


def test_native_artifact_checks_adapter_declared_torch_runtime() -> None:
    manifest = _manifest(backend="pyimgano", runtime_versions={"torch": ">=1.9"})
    manifest["layout"] = "native_detector"
    resolver = Mock(return_value="2.4.0+cu121")

    report = preflight_artifact_compatibility(
        manifest,
        pyimgano_version="0.10.0",
        python_version="3.11.9",
        platform_tag="linux-x86_64",
        runtime_version_resolver=resolver,
    )

    assert dict(report.runtime_versions) == {"torch": "2.4.0+cu121"}
    resolver.assert_called_once_with("torch")


def test_composite_runtime_versions_derive_from_runtime_model_components() -> None:
    manifest = _manifest(runtime_versions={"onnxruntime": ">=1.17,<2"})
    manifest["layout"] = "composite"
    manifest["runtime"] = {"backend": "pyimgano"}

    report = preflight_artifact_compatibility(
        manifest,
        pyimgano_version="0.10.0",
        python_version="3.11.9",
        platform_tag="linux-x86_64",
        runtime_version_resolver=lambda _name: "1.18.0",
    )

    assert dict(report.runtime_versions) == {"onnxruntime": "1.18.0"}
