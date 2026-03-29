from __future__ import annotations


def test_resolve_manifest_preflight_source_or_summary_returns_summary_early() -> None:
    from pyimgano.workbench.manifest_preflight_flow import (
        resolve_manifest_preflight_source_or_summary,
    )

    source = {"summary": {"status": "error"}}

    resolved = resolve_manifest_preflight_source_or_summary(source)

    assert resolved == {"status": "error"}


def test_resolve_manifest_preflight_source_or_summary_returns_none_when_no_summary() -> None:
    from pyimgano.workbench.manifest_preflight_flow import (
        resolve_manifest_preflight_source_or_summary,
    )

    source = {"summary": None, "manifest_path": "/tmp/manifest.jsonl"}

    resolved = resolve_manifest_preflight_source_or_summary(source)

    assert resolved is None


def test_resolve_manifest_record_preflight_summary_returns_summary_early() -> None:
    from pyimgano.workbench.manifest_preflight_flow import (
        resolve_manifest_record_preflight_summary,
    )

    record_preflight = {"summary": {"status": "error"}}

    resolved = resolve_manifest_record_preflight_summary(record_preflight)

    assert resolved == {"status": "error"}
