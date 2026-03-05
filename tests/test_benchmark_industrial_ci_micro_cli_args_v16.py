from __future__ import annotations

import subprocess
import sys


def test_benchmark_industrial_ci_micro_cli_accepts_dataset_kind_and_models() -> None:
    subprocess.run(
        [
            sys.executable,
            "benchmarks/benchmark_industrial_ci_micro.py",
            "--dataset-kind",
            "template_patch",
            "--models",
            "vision_template_ncc_map",
            "--h",
            "32",
            "--w",
            "32",
            "--train",
            "4",
            "--test-normal",
            "2",
            "--test-anomaly",
            "2",
            "--noise-sigma",
            "2.0",
            "--seed",
            "0",
        ],
        check=True,
    )
