from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request


SONAR_HOST_URL = "https://sonarcloud.io"


def fetch_json(api_path: str, token: str) -> dict:
    url = f"{SONAR_HOST_URL}{api_path}"
    request = urllib.request.Request(url)
    request.add_header("Authorization", "Basic " + _basic_auth_token(token))

    with urllib.request.urlopen(request) as response:
        return json.load(response)


def _basic_auth_token(token: str) -> str:
    import base64

    return base64.b64encode(f"{token}:".encode("utf-8")).decode("ascii")


def _issues_api_path(project_key: str, limit: int) -> str:
    query = urllib.parse.urlencode(
        {
            "componentKeys": project_key,
            "resolved": "false",
            "ps": str(limit),
        }
    )
    return f"/api/issues/search?{query}"


def _project_status_api_path(project_key: str) -> str:
    query = urllib.parse.urlencode({"projectKey": project_key})
    return f"/api/qualitygates/project_status?{query}"


def _format_component(issue: dict) -> str:
    component = str(issue.get("component", ""))
    return component.split(":", 1)[-1] if ":" in component else component


def _build_payload(project_key: str, token: str, limit: int) -> dict:
    project_status_payload = fetch_json(_project_status_api_path(project_key), token)
    issues_payload = fetch_json(_issues_api_path(project_key, limit), token)
    return {
        "project_key": project_key,
        "project_status": project_status_payload.get("projectStatus", {}),
        "issues_total": int(issues_payload.get("total", 0)),
        "issues": list(issues_payload.get("issues", [])),
    }


def _render_text(payload: dict) -> str:
    lines = [
        f"Quality gate: {payload['project_status'].get('status', 'UNKNOWN')}",
        f"Open issues: {payload['issues_total']}",
    ]

    for issue in payload["issues"]:
        location = _format_component(issue)
        line = issue.get("line")
        if line is not None:
            location = f"{location}:{line}"
        lines.append(
            f"- {issue.get('severity', 'UNKNOWN')} {issue.get('rule', '')} {location} {issue.get('message', '')}".rstrip()
        )

    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch SonarCloud quality gate and open issues.")
    parser.add_argument("--project-key", default="skygazer42_pyimgano")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--json", action="store_true", dest="json_output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    token = os.environ.get("SONAR_TOKEN")
    if not token:
        print("SONAR_TOKEN is required.", file=sys.stderr)
        return 1

    payload = _build_payload(args.project_key, token, args.limit)
    if args.json_output:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(_render_text(payload), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
