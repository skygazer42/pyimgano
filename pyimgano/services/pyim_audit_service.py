from __future__ import annotations

from typing import Any


def collect_pyim_audit_payload() -> dict[str, Any]:
    from pyimgano.models.registry import audit_model_metadata

    return audit_model_metadata()


__all__ = ["collect_pyim_audit_payload"]
