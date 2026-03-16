from __future__ import annotations

"""Best-effort audit: higher score should correspond to more anomalous samples.

This is not a formal proof. It uses a simple synthetic dataset:
- normal cluster around 0
- outlier cluster far away

If a detector consistently gives lower scores to the obvious outliers, we flag it.
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class AuditResult:
    model: str
    ok: bool
    mean_normal: float
    mean_outlier: float


def _synthetic_data(seed: int = 0, *, n_normal: int = 200, n_outlier: int = 10, d: int = 8):
    rng = np.random.default_rng(int(seed))
    x_normal = rng.normal(loc=0.0, scale=1.0, size=(int(n_normal), int(d)))
    x_outlier = rng.normal(loc=8.0, scale=1.0, size=(int(n_outlier), int(d)))
    x_all = np.concatenate([x_normal, x_outlier], axis=0)
    y = np.concatenate([np.zeros((x_normal.shape[0],)), np.ones((x_outlier.shape[0],))], axis=0)
    return x_all, y


def _ensure_repo_root_on_sys_path() -> None:
    # When invoked as `python tools/<script>.py`, Python sets sys.path[0] to
    # `tools/` rather than the repo root. Add the repo root so `import pyimgano`
    # works without requiring an editable install.
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))


def audit_score_direction(*, seed: int = 0) -> list[AuditResult]:
    _ensure_repo_root_on_sys_path()
    import pyimgano.models  # noqa: F401 - populate registry
    from pyimgano.models import create_model
    from pyimgano.models.registry import MODEL_REGISTRY, list_models

    X, y = _synthetic_data(seed=seed)

    results: list[AuditResult] = []
    for name in sorted(n for n in list_models() if n.startswith("core_")):
        entry = MODEL_REGISTRY.info(name)
        if "deep" in entry.tags:
            continue

        try:
            det = create_model(name, contamination=0.05)
            det.fit(X)
            scores = np.asarray(det.decision_function(X), dtype=np.float64).reshape(-1)
            mn = float(np.mean(scores[y == 0]))
            mo = float(np.mean(scores[y == 1]))
            ok = bool(mo > mn)
            results.append(AuditResult(model=name, ok=ok, mean_normal=mn, mean_outlier=mo))
        except Exception:
            # Not all models are robust on synthetic data; treat as warning.
            results.append(
                AuditResult(
                    model=name, ok=False, mean_normal=float("nan"), mean_outlier=float("nan")
                )
            )
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="audit_score_direction")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any model fails the heuristic (default: warn only)",
    )
    args = parser.parse_args(argv)

    res = audit_score_direction(seed=int(args.seed))
    bad = [r for r in res if not r.ok]

    for r in res:
        status = "OK" if r.ok else "WARN"
        print(
            f"{status:4s} {r.model:28s} mean_normal={r.mean_normal:10.4f} mean_outlier={r.mean_outlier:10.4f}"
        )

    if bad:
        print("")
        print(
            f"WARN: {len(bad)} model(s) did not score synthetic outliers higher on average.",
            file=sys.stderr,
        )
        print(
            "This is a heuristic; review manually before making any breaking changes.",
            file=sys.stderr,
        )
        return 1 if bool(args.strict) else 0

    print("")
    print("OK: all audited core models scored synthetic outliers higher on average.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
