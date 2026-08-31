from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable, Mapping, Sequence

from pyimgano.exporting.protocols import ExportAdapterProtocol
from pyimgano.exporting.types import ArtifactFormat, ExportCapability


class ExportAdapterRegistrationError(ValueError):
    pass


def _normalized_name(value: object, *, field: str) -> str:
    name = str(value).strip()
    if not name:
        raise ExportAdapterRegistrationError(f"{field} must not be empty.")
    return name


class ExportAdapterRegistry:
    """Explicit fitted-detector exporter registry.

    Model registry tags are deliberately not consulted here.  Missing adapters
    produce an unsupported cell instead of an optimistic inference.
    """

    def __init__(self) -> None:
        self._by_model: dict[str, ExportAdapterProtocol] = {}
        self._aliases: dict[str, str] = {}
        self._by_id: dict[str, ExportAdapterProtocol] = {}

    def register(
        self,
        adapter: ExportAdapterProtocol,
        *,
        model_names: Sequence[str] | None = None,
        aliases: Sequence[str] = (),
        overwrite: bool = False,
    ) -> ExportAdapterProtocol:
        adapter_id = _normalized_name(getattr(adapter, "adapter_id", ""), field="adapter_id")
        adapter_version = int(getattr(adapter, "adapter_version", 0))
        if adapter_version < 1:
            raise ExportAdapterRegistrationError("adapter_version must be a positive integer.")

        names_raw = model_names if model_names is not None else getattr(adapter, "model_names", ())
        names = tuple(_normalized_name(name, field="model name") for name in names_raw)
        if not names:
            raise ExportAdapterRegistrationError(
                "An export adapter must register at least one model."
            )
        if len(set(names)) != len(names):
            raise ExportAdapterRegistrationError("Export adapter model names must be unique.")

        alias_names = tuple(_normalized_name(alias, field="alias") for alias in aliases)
        if len(set(alias_names)) != len(alias_names):
            raise ExportAdapterRegistrationError("Export adapter aliases must be unique.")
        overlap = set(names) & set(alias_names)
        if overlap:
            raise ExportAdapterRegistrationError(
                f"Export adapter aliases overlap model names: {sorted(overlap)!r}"
            )

        occupied = [name for name in (*names, *alias_names) if self.contains(name)]
        if occupied and not overwrite:
            raise ExportAdapterRegistrationError(
                f"Export adapter model names already registered: {sorted(occupied)!r}"
            )
        existing_id = self._by_id.get(adapter_id)
        if existing_id is not None and existing_id is not adapter and not overwrite:
            raise ExportAdapterRegistrationError(
                f"Export adapter id already registered: {adapter_id!r}"
            )

        if overwrite:
            for name in (*names, *alias_names):
                self.unregister(name)
            if existing_id is not None and existing_id is not adapter:
                for name, registered in list(self._by_model.items()):
                    if registered is existing_id:
                        self.unregister(name)

        for name in names:
            self._by_model[name] = adapter
        canonical = names[0]
        for alias in alias_names:
            self._aliases[alias] = canonical
        self._by_id[adapter_id] = adapter
        return adapter

    def unregister(self, model_name: str) -> None:
        name = str(model_name).strip()
        canonical = self._aliases.pop(name, None)
        if canonical is not None:
            return
        adapter = self._by_model.pop(name, None)
        if adapter is None:
            return
        for alias, target in list(self._aliases.items()):
            if target == name:
                self._aliases.pop(alias, None)
        if adapter not in self._by_model.values():
            self._by_id.pop(str(getattr(adapter, "adapter_id", "")), None)

    def clear(self) -> None:
        self._by_model.clear()
        self._aliases.clear()
        self._by_id.clear()

    def contains(self, model_name: str) -> bool:
        name = str(model_name).strip()
        return name in self._by_model or name in self._aliases

    def canonical_model_name(self, model_name: str) -> str:
        name = str(model_name).strip()
        if name in self._aliases:
            return self._aliases[name]
        if name in self._by_model:
            return name
        raise KeyError(f"No trained-export adapter registered for model {name!r}.")

    def get(self, model_name: str) -> ExportAdapterProtocol:
        canonical = self.canonical_model_name(model_name)
        return self._by_model[canonical]

    def get_by_id(self, adapter_id: str) -> ExportAdapterProtocol:
        key = str(adapter_id).strip()
        try:
            return self._by_id[key]
        except KeyError as exc:
            raise KeyError(f"Export adapter id not registered: {key!r}.") from exc

    def available_models(self) -> list[str]:
        return sorted({*self._by_model, *self._aliases})

    @staticmethod
    def _bind_adapter_identity(
        capability: ExportCapability,
        adapter: ExportAdapterProtocol,
    ) -> ExportCapability:
        return replace(
            capability,
            adapter_id=(capability.adapter_id or str(adapter.adapter_id)),
            adapter_version=(capability.adapter_version or int(adapter.adapter_version)),
        )

    def capability(
        self,
        model_name: str,
        format: ArtifactFormat | str,
        *,
        context: Mapping[str, Any] | Any | None = None,
    ) -> ExportCapability:
        artifact_format = ArtifactFormat(str(format))
        if not self.contains(model_name):
            return ExportCapability.unsupported(
                artifact_format,
                reason_code="no_export_adapter",
                remediation=(
                    f"Register and certify a trained-export adapter for model {model_name!r}."
                ),
            )

        adapter = self.get(model_name)
        canonical_model = self.canonical_model_name(model_name)
        declared_for_model = getattr(adapter, "declared_capability_for_model", None)
        declared = (
            declared_for_model(canonical_model, artifact_format)
            if callable(declared_for_model)
            else adapter.declared_capability(artifact_format)
        )
        if declared.format is not artifact_format:
            raise ValueError(
                f"Adapter {adapter.adapter_id!r} returned capability for "
                f"{declared.format!s}, expected {artifact_format!s}."
            )
        declared = self._bind_adapter_identity(declared, adapter)
        if context is None:
            return declared

        effective_fn = getattr(adapter, "effective_capability", None)
        if not callable(effective_fn):
            return declared
        effective = effective_fn(artifact_format, context=context)
        if not isinstance(effective, ExportCapability):
            raise TypeError("effective_capability() must return ExportCapability.")
        if effective.format is not artifact_format:
            raise ValueError(
                f"Adapter {adapter.adapter_id!r} returned effective capability for "
                f"{effective.format!s}, expected {artifact_format!s}."
            )
        return self._bind_adapter_identity(effective, adapter)

    def support_table(
        self,
        *,
        model_names: Iterable[str],
        context_by_model: Mapping[str, Mapping[str, Any] | Any] | None = None,
    ) -> dict[str, dict[str, dict[str, Any]]]:
        contexts = dict(context_by_model or {})
        table: dict[str, dict[str, dict[str, Any]]] = {}
        for name in sorted({str(item) for item in model_names}):
            table[name] = {
                str(format): self.capability(
                    name,
                    format,
                    context=contexts.get(name),
                ).to_dict()
                for format in ArtifactFormat
            }
        return table


EXPORT_ADAPTER_REGISTRY = ExportAdapterRegistry()


def register_export_adapter(
    adapter: ExportAdapterProtocol | None = None,
    *,
    model_names: Sequence[str] | None = None,
    aliases: Sequence[str] = (),
    overwrite: bool = False,
):
    def _register(value: ExportAdapterProtocol) -> ExportAdapterProtocol:
        return EXPORT_ADAPTER_REGISTRY.register(
            value,
            model_names=model_names,
            aliases=aliases,
            overwrite=overwrite,
        )

    if adapter is None:
        return _register
    return _register(adapter)


def get_export_adapter(model_name: str) -> ExportAdapterProtocol:
    return EXPORT_ADAPTER_REGISTRY.get(model_name)


def get_export_capability(
    model_name: str,
    format: ArtifactFormat | str,
    *,
    context: Mapping[str, Any] | Any | None = None,
) -> ExportCapability:
    return EXPORT_ADAPTER_REGISTRY.capability(model_name, format, context=context)


def build_export_support_table(
    model_names: Iterable[str] | None = None,
    *,
    context_by_model: Mapping[str, Mapping[str, Any] | Any] | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    if model_names is None:
        try:
            from pyimgano.models.registry import list_models

            model_names = list_models()
        except Exception:
            model_names = EXPORT_ADAPTER_REGISTRY.available_models()
    return EXPORT_ADAPTER_REGISTRY.support_table(
        model_names=model_names,
        context_by_model=context_by_model,
    )


__all__ = [
    "EXPORT_ADAPTER_REGISTRY",
    "ExportAdapterRegistrationError",
    "ExportAdapterRegistry",
    "build_export_support_table",
    "get_export_adapter",
    "get_export_capability",
    "register_export_adapter",
]
