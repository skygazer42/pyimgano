"""Core score ensemble model.

This module exists mainly to provide a stable import path:

    from pyimgano.models.core_score_ensemble import CoreScoreEnsemble

The implementation and registry entry live in `pyimgano.models.score_ensemble`
to keep related ensembles together.
"""

from __future__ import annotations

from .score_ensemble import CoreScoreEnsemble

__all__ = ["CoreScoreEnsemble"]

