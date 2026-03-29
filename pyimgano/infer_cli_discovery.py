from __future__ import annotations

from typing import Any

import pyimgano.cli_discovery_options as cli_discovery_options
import pyimgano.cli_discovery_rendering as cli_discovery_rendering
import pyimgano.cli_listing as cli_listing
import pyimgano.services.discovery_service as discovery_service


def maybe_run_infer_discovery_command(args: Any) -> int | None:
    list_models = bool(getattr(args, "list_models", False))
    list_model_presets = bool(getattr(args, "list_model_presets", False))

    cli_discovery_options.validate_mutually_exclusive_flags(
        [
            ("--list-models", list_models),
            ("--model-info", getattr(args, "model_info", None) is not None),
            ("--list-model-presets", list_model_presets),
            ("--model-preset-info", getattr(args, "model_preset_info", None) is not None),
        ]
    )
    model_list_options = cli_discovery_options.resolve_model_list_discovery_options(
        list_models=list_models,
        tags=getattr(args, "tags", None),
        family=getattr(args, "family", None),
        algorithm_type=getattr(args, "algorithm_type", None),
        year=getattr(args, "year", None),
        allow_family_without_list_models=list_model_presets,
    )

    if list_models:
        names = discovery_service.list_discovery_model_names(
            tags=model_list_options.tags,
            family=model_list_options.family,
            algorithm_type=model_list_options.algorithm_type,
            year=model_list_options.year,
        )
        return cli_listing.emit_listing(
            names,
            json_output=bool(getattr(args, "json", False)),
            sort_keys=False,
        )

    if getattr(args, "model_info", None) is not None:
        payload = discovery_service.build_model_info_payload(str(getattr(args, "model_info")))
        return cli_discovery_rendering.emit_signature_payload(
            payload,
            json_output=bool(getattr(args, "json", False)),
        )

    if list_model_presets:
        names = discovery_service.list_model_preset_names(
            tags=model_list_options.tags,
            family=model_list_options.family,
        )
        json_output = bool(getattr(args, "json", False))
        json_payload = None
        if json_output:
            json_payload = discovery_service.list_model_preset_infos_payload(
                tags=model_list_options.tags,
                family=model_list_options.family,
            )
        return cli_listing.emit_listing(
            names,
            json_output=json_output,
            json_payload=json_payload,
            sort_keys=False,
        )

    if getattr(args, "model_preset_info", None) is not None:
        payload = discovery_service.build_model_preset_info_payload(
            str(getattr(args, "model_preset_info"))
        )
        return cli_discovery_rendering.emit_model_preset_payload(
            payload,
            json_output=bool(getattr(args, "json", False)),
        )

    return None


__all__ = ["maybe_run_infer_discovery_command"]
