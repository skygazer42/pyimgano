from __future__ import annotations

"""
Compatibility shim for older/alternate internal import paths.

Some model implementations import `BaseVisionDeepDetector` from
`pyimgano.models.base_dl`, while the canonical definition lives in
`pyimgano.models.baseCv`.
"""

from .baseCv import BaseVisionDeepDetector

__all__ = ["BaseVisionDeepDetector"]

