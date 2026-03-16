"""Peak-Over-Threshold (POT) thresholding for anomaly scores.

This is a pragmatic implementation based on fitting a Generalized Pareto
Distribution (GPD) to score exceedances above a high threshold `u`.

If the tail is too small or fitting fails, it falls back to a simple quantile
threshold.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def fit_pot_threshold(
    scores,
    *,
    alpha: float,
    tail_fraction: float = 0.1,
    min_exceedances: int = 20,
    eps: float = 1e-12,
) -> tuple[float, dict[str, Any]]:
    """Fit a POT threshold.

    Parameters
    ----------
    scores:
        1D training scores (higher = more anomalous).
    alpha:
        Target tail probability `P(score > threshold)` (often use `contamination`).
    tail_fraction:
        Fraction of the largest scores used to fit the tail model (u = quantile(1-tail_fraction)).
    min_exceedances:
        Minimum number of exceedances required to attempt GPD fitting.
    """

    x = np.asarray(scores, dtype=np.float64).reshape(-1)
    if x.size == 0:
        raise ValueError("scores must be non-empty")

    a = float(alpha)
    if not (0.0 < a < 1.0):
        raise ValueError(f"alpha must be in (0,1), got {alpha}")

    tf = float(tail_fraction)
    if not (0.0 < tf < 1.0):
        raise ValueError(f"tail_fraction must be in (0,1), got {tail_fraction}")

    u = float(np.quantile(x, 1.0 - tf))
    exceed = x[x > u] - u
    n = int(x.size)
    nu = int(exceed.size)

    if nu < int(min_exceedances):
        thr = float(np.quantile(x, 1.0 - a))
        return thr, {
            "method": "quantile_fallback",
            "reason": "too_few_exceedances",
            "u": u,
            "n": n,
            "nu": nu,
            "alpha": a,
            "tail_fraction": tf,
        }

    try:
        from scipy.stats import genpareto  # type: ignore
    except Exception as exc:  # pragma: no cover
        thr = float(np.quantile(x, 1.0 - a))
        return thr, {
            "method": "quantile_fallback",
            "reason": f"scipy_missing: {exc}",
            "u": u,
            "n": n,
            "nu": nu,
            "alpha": a,
            "tail_fraction": tf,
        }

    try:
        # Fit exceedances with loc fixed at 0.
        c, _, scale = genpareto.fit(exceed, floc=0.0)
        c = float(c)
        scale = float(scale)
        if not np.isfinite(c) or not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("Invalid fitted GPD parameters")

        p_exceed_u = float(nu) / float(n)
        p_tail = a / max(p_exceed_u, float(eps))  # conditional tail prob beyond u
        p_tail = min(max(p_tail, float(eps)), 1.0 - float(eps))
        q = 1.0 - p_tail

        excess_thr = float(genpareto.ppf(q, c=c, loc=0.0, scale=scale))
        if not np.isfinite(excess_thr):
            raise ValueError("Invalid GPD ppf")

        thr = float(u + excess_thr)
        return thr, {
            "method": "pot",
            "u": u,
            "shape": c,
            "scale": scale,
            "p_exceed_u": p_exceed_u,
            "alpha": a,
            "tail_fraction": tf,
            "n": n,
            "nu": nu,
        }
    except Exception as exc:
        thr = float(np.quantile(x, 1.0 - a))
        return thr, {
            "method": "quantile_fallback",
            "reason": f"fit_failed: {exc}",
            "u": u,
            "n": n,
            "nu": nu,
            "alpha": a,
            "tail_fraction": tf,
        }
