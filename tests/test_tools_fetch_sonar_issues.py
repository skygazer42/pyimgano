from __future__ import annotations

import json


def test_fetch_sonar_issues_text_output_formats_quality_gate_and_issue_summary(
    monkeypatch,
    capsys,
) -> None:
    import tools.fetch_sonar_issues as fetch_sonar_issues

    def _fake_fetch_json(api_path: str, token: str) -> dict:
        assert token == "token"
        if api_path.startswith("/api/qualitygates/project_status"):
            return {"projectStatus": {"status": "OK"}}
        if api_path.startswith("/api/issues/search"):
            return {
                "total": 2,
                "issues": [
                    {
                        "severity": "CRITICAL",
                        "rule": "python:S1192",
                        "component": "skygazer42_pyimgano:pyimgano/reporting/run_quality.py",
                        "line": 124,
                        "message": 'Define a constant instead of duplicating "infer_config.json".',
                    },
                    {
                        "severity": "MAJOR",
                        "rule": "python:S3358",
                        "component": "skygazer42_pyimgano:pyimgano/reporting/run_index.py",
                        "line": 2394,
                        "message": "Extract this nested conditional expression into an independent statement.",
                    },
                ],
            }
        raise AssertionError(api_path)

    monkeypatch.setenv("SONAR_TOKEN", "token")
    monkeypatch.setattr(fetch_sonar_issues, "fetch_json", _fake_fetch_json)

    assert fetch_sonar_issues.main(["--project-key", "skygazer42_pyimgano", "--limit", "2"]) == 0
    out = capsys.readouterr().out

    assert "Quality gate: OK" in out
    assert "Open issues: 2" in out
    assert "pyimgano/reporting/run_quality.py:124" in out
    assert "python:S1192" in out


def test_fetch_sonar_issues_json_output_includes_project_status_and_issues(
    monkeypatch,
    capsys,
) -> None:
    import tools.fetch_sonar_issues as fetch_sonar_issues

    def _fake_fetch_json(api_path: str, token: str) -> dict:
        assert token == "token"
        if api_path.startswith("/api/qualitygates/project_status"):
            return {"projectStatus": {"status": "OK"}}
        if api_path.startswith("/api/issues/search"):
            return {"total": 1, "issues": [{"rule": "python:S7519", "message": "Replace with dict fromkeys."}]}
        raise AssertionError(api_path)

    monkeypatch.setenv("SONAR_TOKEN", "token")
    monkeypatch.setattr(fetch_sonar_issues, "fetch_json", _fake_fetch_json)

    assert (
        fetch_sonar_issues.main(
            ["--project-key", "skygazer42_pyimgano", "--limit", "1", "--json"]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["project_key"] == "skygazer42_pyimgano"
    assert payload["project_status"]["status"] == "OK"
    assert payload["issues_total"] == 1
    assert payload["issues"][0]["rule"] == "python:S7519"
