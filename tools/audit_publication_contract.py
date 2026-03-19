from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyimgano.reporting.publication_quality import evaluate_publication_quality
from pyimgano.reporting.suite_export import export_suite_tables


def main() -> int:
    payload = {
        "suite": "industrial-v4",
        "dataset": "mvtec",
        "category": "bottle",
        "rows": [{"name": "vision_patchcore", "auroc": 0.95, "run_dir": "runs/example"}],
        "split_fingerprint": {
            "schema_version": 1,
            "sha256": "b" * 64,
            "train_count": 10,
            "calibration_count": 0,
            "test_count": 5,
        },
        "benchmark_config": {
            "source": "benchmarks/configs/official_mvtec_industrial_v4_cpu_offline.json",
            "official": True,
            "sha256": "a" * 64,
        },
        "environment_fingerprint_sha256": "f" * 64,
    }

    with tempfile.TemporaryDirectory(prefix="pyimgano_publication_audit_") as tmp:
        out_dir = Path(tmp)
        (out_dir / "report.json").write_text('{"suite":"industrial-v4"}', encoding="utf-8")
        (out_dir / "config.json").write_text('{"config":{"seed":123}}', encoding="utf-8")
        (out_dir / "environment.json").write_text(
            '{"fingerprint_sha256":"' + ("f" * 64) + '"}',
            encoding="utf-8",
        )
        export_suite_tables(payload, out_dir, formats=["csv"])
        quality = evaluate_publication_quality(out_dir)
        trust_signals = dict(quality.get("trust_signals", {}))
        if str(quality.get("status")) != "ready":
            print(f"error: publication contract not ready: {quality}")
            return 1
        if not bool(trust_signals.get("has_official_benchmark_config")):
            print(f"error: publication contract missing official benchmark trust signal: {quality}")
            return 1
        if not bool(trust_signals.get("has_evaluation_contract")):
            print(f"error: publication contract missing evaluation contract trust signal: {quality}")
            return 1
        if not bool(trust_signals.get("has_benchmark_citation")):
            print(f"error: publication contract missing citation trust signal: {quality}")
            return 1
        if not bool(trust_signals.get("has_run_artifact_refs")):
            print(f"error: publication contract missing run artifact audit refs: {quality}")
            return 1
        if not bool(trust_signals.get("has_run_artifact_digests")):
            print(f"error: publication contract missing run artifact audit digests: {quality}")
            return 1
        if not bool(trust_signals.get("has_exported_file_digests")):
            print(f"error: publication contract missing exported leaderboard digests: {quality}")
            return 1

    print("OK: suite publication contract is ready, auditable, and trust-signaled.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
