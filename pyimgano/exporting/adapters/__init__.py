from __future__ import annotations

"""Built-in fitted-detector export adapters.

Importing :mod:`pyimgano.exporting` registers these adapters explicitly.  The
registration helper is idempotent so application/plugin discovery can call it
again without replacing third-party registrations.
"""

from pyimgano.exporting.adapters.autoencoder import (
    AUTOENCODER_EXPORT_ADAPTER,
    AUTOENCODER_STATE_CODEC,
    AutoencoderExportAdapter,
    AutoencoderStateCodec,
)
from pyimgano.exporting.adapters.ecod_composite import (
    ECOD_COMPOSITE_EXPORT_ADAPTER,
    ECOD_CORE_STATE_CODEC,
    CoreECODStateCodec,
    ECODCompositeExportAdapter,
)


def register_builtin_export_adapters() -> None:
    from pyimgano.exporting.registry import EXPORT_ADAPTER_REGISTRY
    from pyimgano.exporting.state_codec import STATE_CODEC_REGISTRY

    if not EXPORT_ADAPTER_REGISTRY.contains("ae_resnet_unet"):
        EXPORT_ADAPTER_REGISTRY.register(AUTOENCODER_EXPORT_ADAPTER)
    else:
        registered = EXPORT_ADAPTER_REGISTRY.get("ae_resnet_unet")
        if registered is not AUTOENCODER_EXPORT_ADAPTER and (
            str(getattr(registered, "adapter_id", "")) != AUTOENCODER_EXPORT_ADAPTER.adapter_id
        ):
            raise RuntimeError(
                "The built-in ae_resnet_unet export adapter identity is already occupied."
            )

    try:
        registered_codec = STATE_CODEC_REGISTRY.get(
            AUTOENCODER_STATE_CODEC.codec_id,
            AUTOENCODER_STATE_CODEC.codec_version,
        )
    except KeyError:
        STATE_CODEC_REGISTRY.register(AUTOENCODER_STATE_CODEC)
    else:
        if registered_codec is not AUTOENCODER_STATE_CODEC and (
            int(getattr(registered_codec, "state_schema_version", 0))
            != AUTOENCODER_STATE_CODEC.state_schema_version
        ):
            raise RuntimeError(
                "The built-in ae_resnet_unet state codec identity is already occupied."
            )

    for model_name in ECOD_COMPOSITE_EXPORT_ADAPTER.model_names:
        if not EXPORT_ADAPTER_REGISTRY.contains(model_name):
            continue
        registered = EXPORT_ADAPTER_REGISTRY.get(model_name)
        if registered is not ECOD_COMPOSITE_EXPORT_ADAPTER and (
            str(getattr(registered, "adapter_id", "")) != ECOD_COMPOSITE_EXPORT_ADAPTER.adapter_id
        ):
            raise RuntimeError(
                f"The built-in {model_name} export adapter identity is already occupied."
            )
    missing_ecod_models = [
        name
        for name in ECOD_COMPOSITE_EXPORT_ADAPTER.model_names
        if not EXPORT_ADAPTER_REGISTRY.contains(name)
    ]
    if missing_ecod_models:
        if len(missing_ecod_models) != len(ECOD_COMPOSITE_EXPORT_ADAPTER.model_names):
            raise RuntimeError("The built-in ECOD composite adapter is only partially registered.")
        EXPORT_ADAPTER_REGISTRY.register(ECOD_COMPOSITE_EXPORT_ADAPTER)

    try:
        registered_ecod_codec = STATE_CODEC_REGISTRY.get(
            ECOD_CORE_STATE_CODEC.codec_id,
            ECOD_CORE_STATE_CODEC.codec_version,
        )
    except KeyError:
        STATE_CODEC_REGISTRY.register(ECOD_CORE_STATE_CODEC)
    else:
        if registered_ecod_codec is not ECOD_CORE_STATE_CODEC and (
            int(getattr(registered_ecod_codec, "state_schema_version", 0))
            != ECOD_CORE_STATE_CODEC.state_schema_version
        ):
            raise RuntimeError("The built-in CoreECOD state codec identity is already occupied.")


__all__ = [
    "AUTOENCODER_EXPORT_ADAPTER",
    "AUTOENCODER_STATE_CODEC",
    "ECOD_COMPOSITE_EXPORT_ADAPTER",
    "ECOD_CORE_STATE_CODEC",
    "AutoencoderExportAdapter",
    "AutoencoderStateCodec",
    "CoreECODStateCodec",
    "ECODCompositeExportAdapter",
    "register_builtin_export_adapters",
]
