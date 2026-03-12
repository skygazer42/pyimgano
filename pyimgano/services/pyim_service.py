from __future__ import annotations

from pyimgano.pyim_contracts import (
    PyimListPayload,
    PyimListRequest,
)
import pyimgano.services.pyim_payload_collectors as pyim_payload_collectors


def collect_pyim_listing_payload(request: PyimListRequest) -> PyimListPayload:
    payload_kwargs = pyim_payload_collectors.empty_pyim_payload_kwargs()

    for field_name in request.requested_payload_fields():
        payload_kwargs[field_name] = pyim_payload_collectors.collect_pyim_payload_field(
            field_name,
            request,
        )

    return PyimListPayload(**payload_kwargs)


__all__ = [
    "PyimListPayload",
    "PyimListRequest",
    "collect_pyim_listing_payload",
]
