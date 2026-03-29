from __future__ import annotations


def test_format_suite_check_line_renders_missing_extras_suffix() -> None:
    from pyimgano.doctor_rendering import format_suite_check_line

    line = format_suite_check_line(
        suite_name="industrial-v4",
        info={
            "summary": {
                "total": 5,
                "runnable": 3,
                "missing_extras": ["torch", "faiss"],
            }
        },
    )

    assert line == "- industrial-v4: runnable 3/5 (missing extras: torch, faiss)"


def test_format_require_extras_line_handles_missing_and_ok() -> None:
    from pyimgano.doctor_rendering import format_require_extras_line

    assert format_require_extras_line(
        {
            "required": ["torch"],
            "missing": [],
            "ok": True,
            "install_hint": None,
        }
    ) == "require_extras: OK"

    assert format_require_extras_line(
        {
            "required": ["torch", "faiss"],
            "missing": ["faiss"],
            "ok": False,
            "install_hint": "pip install 'pyimgano[faiss]'",
        }
    ) == "require_extras: MISSING (faiss) -> pip install 'pyimgano[faiss]'"


def test_format_readiness_lines_renders_status_and_issues() -> None:
    from pyimgano.doctor_rendering import format_readiness_lines

    lines = format_readiness_lines(
        {
            "target_kind": "run",
            "path": "/tmp/run_a",
            "status": "warning",
            "issues": ["insufficient_quality_status", "missing_bundle_manifest"],
        }
    )

    assert lines == [
        "readiness:",
        "- target_kind: run",
        "- path: /tmp/run_a",
        "- status: warning",
        "- issues: insufficient_quality_status, missing_bundle_manifest",
    ]
