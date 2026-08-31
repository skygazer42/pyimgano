from __future__ import annotations

"""Dependency discovery for ONNX tensors stored in external data files."""

from typing import Any

_MAX_PROTOBUF_MESSAGES = 1_000_000


def _iter_tensor_messages(model: Any):
    stack = [model]
    visited = 0
    while stack:
        message = stack.pop()
        visited += 1
        if visited > _MAX_PROTOBUF_MESSAGES:
            raise ValueError("ONNX protobuf contains too many nested messages.")

        descriptor = getattr(message, "DESCRIPTOR", None)
        if descriptor is None:
            continue
        if str(getattr(descriptor, "full_name", "")) == "onnx.TensorProto":
            yield message

        for field, value in message.ListFields():
            if getattr(field, "message_type", None) is None:
                continue
            is_repeated = getattr(field, "is_repeated", None)
            if is_repeated is None:
                is_repeated = int(getattr(field, "label", 0)) == int(
                    getattr(field, "LABEL_REPEATED", 3)
                )
            if bool(is_repeated):
                stack.extend(reversed(value))
            else:
                stack.append(value)


def external_data_locations(model: Any) -> list[str]:
    """Return every external-data location referenced by any ONNX TensorProto.

    TensorProto values can occur in graph initializers, sparse tensors, constant
    attributes, nested graphs, and functions.  Protobuf reflection keeps the
    dependency walk complete without relying on ONNX's private helper APIs.
    """

    locations: set[str] = set()
    for tensor in _iter_tensor_messages(model):
        entries = list(getattr(tensor, "external_data", ()))
        data_location = int(getattr(tensor, "data_location", 0))
        if not entries and data_location != 1:  # TensorProto.EXTERNAL == 1
            continue
        declared = [str(entry.value) for entry in entries if str(entry.key) == "location"]
        if len(declared) != 1:
            raise ValueError(
                "Every external ONNX tensor must declare exactly one external_data location."
            )
        locations.add(declared[0])
    return sorted(locations)


__all__ = ["external_data_locations"]
