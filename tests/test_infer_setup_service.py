from __future__ import annotations

import pyimgano.services.infer_load_service as infer_load_service
import pyimgano.services.infer_setup_service as infer_setup_service


def test_infer_setup_service_reexports_infer_load_boundary() -> None:
    assert infer_setup_service.__all__ == infer_load_service.__all__

    for export_name in infer_setup_service.__all__:
        assert getattr(infer_setup_service, export_name) is getattr(
            infer_load_service, export_name
        )
