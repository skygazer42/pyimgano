from __future__ import annotations

"""Best-effort audit: higher score should correspond to more anomalous samples.

This is not a formal proof. It uses a fitted-novelty dataset:
- a normal-only training set concentrated near a low-dimensional manifold
- held-out normal queries from the same manifold
- anomalous queries shifted along a low-variance, nearly orthogonal direction

If a detector consistently gives lower scores to the obvious outliers, we flag it.
"""

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class AuditResult:
    model: str
    ok: bool
    mean_normal: float
    mean_outlier: float
    duration_seconds: float
    error: str | None = None


def _normal_manifold(rng: np.random.Generator, *, n_samples: int, d: int) -> np.ndarray:
    if d < 3:
        raise ValueError("score-direction fixture requires at least three dimensions")
    latent = rng.normal(size=(int(n_samples), 2))
    values = rng.normal(scale=0.03, size=(int(n_samples), int(d)))
    values[:, 0] = 3.0 + latent[:, 0]
    values[:, 1] = 0.5 * latent[:, 1]
    return values


def _synthetic_data(
    seed: int = 0,
    *,
    n_train: int = 200,
    n_normal: int = 50,
    n_outlier: int = 10,
    d: int = 8,
):
    rng = np.random.default_rng(int(seed))
    x_train = _normal_manifold(rng, n_samples=int(n_train), d=int(d))
    x_normal = _normal_manifold(rng, n_samples=int(n_normal), d=int(d))
    x_outlier = _normal_manifold(rng, n_samples=int(n_outlier), d=int(d))
    x_outlier[:, -1] += 8.0
    x_query = np.concatenate([x_normal, x_outlier], axis=0)
    y = np.concatenate([np.zeros((x_normal.shape[0],)), np.ones((x_outlier.shape[0],))], axis=0)
    return x_train, x_query, y


def _ensure_repo_root_on_sys_path() -> None:
    # When invoked as `python tools/<script>.py`, Python sets sys.path[0] to
    # `tools/` rather than the repo root. Add the repo root so `import pyimgano`
    # works without requiring an editable install.
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))


def audit_score_direction(*, seed: int = 0, progress: bool = False) -> list[AuditResult]:
    _ensure_repo_root_on_sys_path()
    import pyimgano.models  # noqa: F401 - populate registry
    from pyimgano.models import create_model
    from pyimgano.models.registry import MODEL_REGISTRY, list_models

    x_train, x_query, y = _synthetic_data(seed=seed)

    results: list[AuditResult] = []
    for name in sorted(n for n in list_models() if n.startswith("core_")):
        entry = MODEL_REGISTRY.info(name)
        if "deep" in entry.tags:
            continue

        if progress:
            print(f"AUDIT {name}", file=sys.stderr, flush=True)
        started = time.perf_counter()
        try:
            det = create_model(name, contamination=0.05)
            det.fit(x_train)
            scores = np.asarray(det.decision_function(x_query), dtype=np.float64).reshape(-1)
            mn = float(np.mean(scores[y == 0]))
            mo = float(np.mean(scores[y == 1]))
            ok = bool(mo > mn)
            results.append(
                AuditResult(
                    model=name,
                    ok=ok,
                    mean_normal=mn,
                    mean_outlier=mo,
                    duration_seconds=time.perf_counter() - started,
                )
            )
        except Exception as exc:
            # Not all models are robust on synthetic data; treat as warning.
            results.append(
                AuditResult(
                    model=name,
                    ok=False,
                    mean_normal=float("nan"),
                    mean_outlier=float("nan"),
                    duration_seconds=time.perf_counter() - started,
                    error=f"{type(exc).__name__}: {exc}",
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
    parser.add_argument("--json", action="store_true", help="Emit structured JSON output")
    args = parser.parse_args(argv)

    res = audit_score_direction(seed=int(args.seed), progress=not bool(args.json))
    bad = [r for r in res if not r.ok]

    if bool(args.json):
        print(
            json.dumps(
                {
                    "seed": int(args.seed),
                    "results": [asdict(result) for result in res],
                    "warning_count": len(bad),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1 if bool(args.strict) and bad else 0

    for r in res:
        status = "OK" if r.ok else "WARN"
        detail = f" error={r.error}" if r.error is not None else ""
        print(
            f"{status:4s} {r.model:28s} mean_normal={r.mean_normal:10.4f} "
            f"mean_outlier={r.mean_outlier:10.4f} seconds={r.duration_seconds:8.3f}{detail}"
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
