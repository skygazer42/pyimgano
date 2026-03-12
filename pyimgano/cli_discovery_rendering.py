from __future__ import annotations

from typing import Any, Mapping

import pyimgano.cli_output as cli_output


def emit_signature_payload(payload: Mapping[str, Any], *, json_output: bool) -> int:
    if bool(json_output):
        return cli_output.emit_jsonable(payload)

    print(f"Name: {payload['name']}")
    tags = payload["tags"]
    print(f"Tags: {', '.join(tags) if tags else '<none>'}")
    print("Metadata:")
    metadata = payload["metadata"]
    if metadata:
        for key in sorted(metadata):
            print(f"  {key}: {metadata[key]}")
    else:
        print("  <none>")
    print("Signature:")
    print(f"  {payload['signature']}")
    print(f"Accepts **kwargs: {'yes' if payload['accepts_var_kwargs'] else 'no'}")
    print("Accepted kwargs:")
    for key in payload["accepted_kwargs"]:
        print(f"  - {key}")
    return 0


def emit_model_preset_payload(payload: Mapping[str, Any], *, json_output: bool) -> int:
    if bool(json_output):
        return cli_output.emit_jsonable(payload)

    print(f"Name: {payload['name']}")
    print(f"Model: {payload['model']}")
    print("Kwargs:")
    kwargs = payload["kwargs"]
    if kwargs:
        for key in sorted(kwargs):
            print(f"  {key}: {kwargs[key]}")
    else:
        print("  <none>")
    print(f"Optional: {'yes' if payload['optional'] else 'no'}")
    tags_out = payload.get("tags", [])
    print(f"Tags: {', '.join(str(t) for t in tags_out) if tags_out else '<none>'}")
    print(f"Description: {payload['description']}")
    return 0
