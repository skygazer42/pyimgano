import subprocess
import sys
from pathlib import Path


def test_audit_benchmark_configs_script_runs():
    subprocess.run([sys.executable, "tools/audit_benchmark_configs.py"], check=True)


def test_audit_benchmark_configs_reports_unknown_doc_reference(tmp_path: Path) -> None:
    doc = tmp_path / "bench_doc.md"
    doc.write_text(
        "pyimgano-benchmark --config official_not_a_real_config.json\n",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [sys.executable, "tools/audit_benchmark_configs.py", "--docs", str(doc)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 1
    assert "unknown benchmark config reference" in proc.stdout
    assert "official_not_a_real_config.json" in proc.stdout
