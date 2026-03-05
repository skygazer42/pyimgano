from __future__ import annotations

import argparse
from pathlib import Path

from pyimgano.synthesize_cli import synthesize_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="synthesis_generate_dataset_demo")
    parser.add_argument("--in-dir", required=True, help="Directory of normal images")
    parser.add_argument("--out-root", required=True, help="Output dataset root")
    parser.add_argument("--preset", default="scratch")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-train", type=int, default=8)
    parser.add_argument("--n-test-normal", type=int, default=4)
    parser.add_argument("--n-test-anomaly", type=int, default=4)
    args = parser.parse_args(argv)

    synthesize_dataset(
        in_dir=Path(args.in_dir),
        out_root=Path(args.out_root),
        category="synthetic_demo",
        preset=str(args.preset),
        seed=int(args.seed),
        n_train=int(args.n_train),
        n_test_normal=int(args.n_test_normal),
        n_test_anomaly=int(args.n_test_anomaly),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
