from __future__ import annotations

from typing import Any, Mapping

import pyimgano.services.workbench_service as workbench_service
from pyimgano.workbench.config import WorkbenchConfig


def build_workbench_infer_config_payload(
    *,
    config: WorkbenchConfig,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    return workbench_service.build_infer_config_payload(config=config, report=report)


__all__ = ["build_workbench_infer_config_payload"]
