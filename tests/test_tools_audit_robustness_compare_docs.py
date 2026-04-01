from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_audit_robustness_compare_docs_reports_missing_requirements(tmp_path: Path) -> None:
    doc = tmp_path / "robustness.md"
    doc.write_text("pyimgano-runs compare\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "tools/audit_robustness_compare_docs.py",
            str(doc),
            "--require",
            "--same-robustness-protocol-as",
            "--require",
            "--require-same-robustness-protocol",
            "--require",
            "robustness_protocol_comparison",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 1
    assert "--same-robustness-protocol-as" in proc.stdout
    assert "--require-same-robustness-protocol" in proc.stdout
    assert "robustness_protocol_comparison" in proc.stdout
    assert str(doc) in proc.stdout


def test_audit_robustness_compare_docs_accepts_complete_doc(tmp_path: Path) -> None:
    doc = tmp_path / "robustness.md"
    doc.write_text(
        "\n".join(
            [
                "pyimgano-runs compare",
                "--same-robustness-protocol-as",
                "--require-same-robustness-protocol",
                "robustness_protocol_comparison",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            sys.executable,
            "tools/audit_robustness_compare_docs.py",
            str(doc),
            "--require",
            "--same-robustness-protocol-as",
            "--require",
            "--require-same-robustness-protocol",
            "--require",
            "robustness_protocol_comparison",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0
    assert "OK" in proc.stdout


def test_audit_robustness_compare_docs_default_rules_cover_discovery_filters() -> None:
    from tools.audit_robustness_compare_docs import DEFAULT_RULES

    rules = {rule.path: set(rule.required) for rule in DEFAULT_RULES}

    assert "--same-environment-as" in rules["docs/CLI_REFERENCE.md"]
    assert "--same-target-as" in rules["docs/CLI_REFERENCE.md"]
    assert "trust_comparison" in rules["docs/CLI_REFERENCE.md"]
    assert "candidate_blocking_reasons" in rules["docs/CLI_REFERENCE.md"]
    assert "candidate_comparability_gates" in rules["docs/CLI_REFERENCE.md"]
    assert "baseline_dataset_readiness" in rules["docs/CLI_REFERENCE.md"]
    assert "candidate_dataset_readiness" in rules["docs/CLI_REFERENCE.md"]
    assert "candidate_verdict." in rules["docs/CLI_REFERENCE.md"]
    assert "candidate_blocking_reasons." in rules["docs/CLI_REFERENCE.md"]
    assert "candidate_comparability_gates." in rules["docs/CLI_REFERENCE.md"]
    assert "baseline_dataset_readiness_status" in rules["docs/CLI_REFERENCE.md"]
    assert "candidate_dataset_readiness_status." in rules["docs/CLI_REFERENCE.md"]
    assert "comparison_trust_reason" in rules["docs/CLI_REFERENCE.md"]
    assert "comparison_trust_ref." in rules["docs/CLI_REFERENCE.md"]
    assert "--same-environment-as" in rules["docs/RUN_COMPARISON.md"]
    assert "--same-target-as" in rules["docs/RUN_COMPARISON.md"]
    assert "trust_comparison" in rules["docs/RUN_COMPARISON.md"]
    assert "candidate_blocking_reasons" in rules["docs/RUN_COMPARISON.md"]
    assert "candidate_comparability_gates" in rules["docs/RUN_COMPARISON.md"]
    assert "baseline_dataset_readiness" in rules["docs/RUN_COMPARISON.md"]
    assert "candidate_dataset_readiness" in rules["docs/RUN_COMPARISON.md"]
    assert "candidate_verdict." in rules["docs/RUN_COMPARISON.md"]
    assert "candidate_blocking_reasons." in rules["docs/RUN_COMPARISON.md"]
    assert "candidate_comparability_gates." in rules["docs/RUN_COMPARISON.md"]
    assert "baseline_dataset_readiness_status" in rules["docs/RUN_COMPARISON.md"]
    assert "candidate_dataset_readiness_status." in rules["docs/RUN_COMPARISON.md"]
    assert "comparison_trust_reason" in rules["docs/RUN_COMPARISON.md"]
    assert "comparison_trust_ref." in rules["docs/RUN_COMPARISON.md"]


def test_audit_robustness_compare_docs_current_repo_is_clean() -> None:
    proc = subprocess.run(
        [sys.executable, "tools/audit_robustness_compare_docs.py"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
