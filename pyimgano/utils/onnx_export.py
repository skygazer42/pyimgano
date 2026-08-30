from __future__ import annotations

import inspect
import warnings
from pathlib import Path
from typing import Any, Sequence, Union


def export_torch_model(
    *,
    torch: Any,
    model: Any,
    dummy_input: Any,
    output_path: Union[str, Path],
    input_names: Sequence[str],
    output_names: Sequence[str],
    opset_version: int,
    dynamic_batch: bool = True,
    do_constant_folding: bool = True,
) -> None:
    """Export a single-input Torch model without deprecated current-Torch APIs.

    Torch 2.9 made the dynamo exporter the default. It expects
    ``dynamic_shapes`` rather than ``dynamic_axes`` and currently implements
    ONNX opset 18 and newer. Older requested opsets deliberately use the legacy
    exporter so the requested schema is preserved.
    """

    export_parameters = inspect.signature(torch.onnx.export).parameters
    export_api = getattr(torch, "export", None)
    dim_type = getattr(export_api, "Dim", None)
    use_dynamo = (
        int(opset_version) >= 18
        and "dynamo" in export_parameters
        and "dynamic_shapes" in export_parameters
        and dim_type is not None
    )

    kwargs: dict[str, Any] = {
        "input_names": list(input_names),
        "output_names": list(output_names),
        "opset_version": int(opset_version),
        "do_constant_folding": bool(do_constant_folding),
    }
    if use_dynamo:
        kwargs["dynamo"] = True
        if dynamic_batch:
            kwargs["dynamic_shapes"] = ({0: dim_type("batch")},)
    else:
        if "dynamo" in export_parameters:
            kwargs["dynamo"] = False
        if dynamic_batch:
            kwargs["dynamic_axes"] = {
                str(input_names[0]): {0: "batch"},
                str(output_names[0]): {0: "batch"},
            }

    with warnings.catch_warnings():
        # Upstream pytree currently emits this while the current dynamo ONNX
        # exporter copies its graph specification. It is not caused by the
        # exported model or by pyimgano's API usage.
        warnings.filterwarnings(
            "ignore",
            message=r"`isinstance\(treespec, LeafSpec\)` is deprecated.*",
            category=FutureWarning,
        )
        if not use_dynamo:
            # The legacy route is intentional for opset < 18, which the current
            # exporter cannot reliably produce through version conversion.
            warnings.filterwarnings(
                "ignore",
                message=r"You are using the legacy TorchScript-based ONNX export.*",
                category=DeprecationWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message=r"The feature will be removed\. Please remove usage of this function",
                category=DeprecationWarning,
            )
        torch.onnx.export(model, dummy_input, str(output_path), **kwargs)
