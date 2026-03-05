from __future__ import annotations

import re
from pathlib import Path


def _read_repo_file(relative_path: str) -> str:
    repo_root = Path(__file__).resolve().parents[1]
    return (repo_root / relative_path).read_text(encoding="utf-8")


def _parse_optional_dependency_extras(pyproject_text: str) -> set[str]:
    """Parse `[project.optional-dependencies]` keys from pyproject.toml.

    We intentionally avoid adding a TOML parser dependency in tests (py39+),
    and only need a minimal, stable parse for the extras table.
    """

    keys: set[str] = set()
    in_section = False
    for raw in pyproject_text.splitlines():
        line = raw.strip()
        if line == "[project.optional-dependencies]":
            in_section = True
            continue

        if not in_section:
            continue

        # Next section starts.
        if line.startswith("[") and line.endswith("]"):
            break

        m = re.match(r"^([A-Za-z0-9_]+)\s*=\s*\[", line)
        if m:
            keys.add(m.group(1))

    return keys


def test_extras_mapping_does_not_drift_from_pyproject_optional_deps() -> None:
    pyproject = _read_repo_file("pyproject.toml")
    extras = _parse_optional_dependency_extras(pyproject)
    assert extras, "Failed to parse [project.optional-dependencies] extras from pyproject.toml"

    # Dev/docs aggregation extras are intentionally not part of runtime import gating.
    ignored = {"dev", "docs", "all"}
    expected = extras - ignored

    from pyimgano.utils.extras import EXTRA_ROOT_MODULES

    mapped = set(EXTRA_ROOT_MODULES.keys())

    assert expected == mapped, (
        "EXTRA_ROOT_MODULES must stay in sync with pyproject.toml optional-dependencies "
        "(excluding dev/docs/all).\n"
        f"Missing in mapping: {sorted(expected - mapped)}\n"
        f"Unexpected in mapping: {sorted(mapped - expected)}"
    )


def test_extra_for_root_module_covers_common_roots() -> None:
    from pyimgano.utils.extras import extra_for_root_module

    assert extra_for_root_module("torch") == "torch"
    assert extra_for_root_module("torchvision") == "torch"
    assert extra_for_root_module("onnxruntime") == "onnx"
    assert extra_for_root_module("openvino") == "openvino"
    assert extra_for_root_module("skimage") == "skimage"
    assert extra_for_root_module("numba") == "numba"
    assert extra_for_root_module("faiss") == "faiss"
    assert extra_for_root_module("open_clip") == "clip"
    assert extra_for_root_module("diffusers") == "diffusion"
    assert extra_for_root_module("anomalib") == "anomalib"
    assert extra_for_root_module("mamba_ssm") == "mamba"
    assert extra_for_root_module("patchcore") == "patchcore_inspection"
