from __future__ import annotations

from pathlib import Path

import tools.audit_third_party_notices as audit_module


def _write_repo_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_audit_third_party_notices_reports_missing_notice_entry(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    repo_root = tmp_path
    script_path = repo_root / "tools" / "audit_third_party_notices.py"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text("# test shim\n", encoding="utf-8")
    _write_repo_file(
        repo_root / "pyimgano" / "sample.py",
        "# UPSTREAM: openai/example-repo @ abcdef\n",
    )
    _write_repo_file(repo_root / "third_party" / "NOTICE.md", "")

    monkeypatch.setattr(audit_module, "__file__", str(script_path))

    rc = audit_module.main()

    captured = capsys.readouterr()
    assert rc == 1
    assert "missing third-party notice entries" in captured.err
    assert "openai/example-repo" in captured.err


def test_audit_third_party_notices_accepts_notice_url_entry(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    repo_root = tmp_path
    script_path = repo_root / "tools" / "audit_third_party_notices.py"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text("# test shim\n", encoding="utf-8")
    _write_repo_file(
        repo_root / "pyimgano" / "sample.py",
        "# UPSTREAM: https://github.com/openai/example-repo (commit abcdef)\n",
    )
    _write_repo_file(
        repo_root / "third_party" / "NOTICE.md",
        "Uses https://github.com/openai/example-repo for reference.\n",
    )

    monkeypatch.setattr(audit_module, "__file__", str(script_path))

    rc = audit_module.main()

    captured = capsys.readouterr()
    assert rc == 0
    assert "all are covered" in captured.out
