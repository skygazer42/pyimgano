from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class OpenVINOGraphExportError(RuntimeError):
    pass


@dataclass(frozen=True)
class OpenVINOGraphInfo:
    xml_path: Path
    bin_path: Path


def convert_verified_onnx_to_openvino(
    onnx_path: str | Path,
    xml_path: str | Path,
) -> OpenVINOGraphInfo:
    """Convert a previously checked ONNX full graph to a self-contained IR pair."""

    source = Path(onnx_path)
    if not source.is_file():
        raise FileNotFoundError(f"Verified ONNX source not found: {source}")
    target = Path(xml_path)
    if target.suffix.lower() != ".xml":
        raise OpenVINOGraphExportError("OpenVINO runtime entrypoint must use a .xml suffix.")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        import openvino as ov
    except ModuleNotFoundError as exc:  # pragma: no cover - capability gate
        raise OpenVINOGraphExportError(
            "OpenVINO conversion requires pyimgano[openvino-export]."
        ) from exc

    try:
        model = ov.convert_model(str(source))
        ov.save_model(model, str(target), compress_to_fp16=False)
    except Exception as exc:
        raise OpenVINOGraphExportError(
            f"Failed to convert verified ONNX graph to OpenVINO IR: {exc}"
        ) from exc
    weights = target.with_suffix(".bin")
    if not target.is_file() or not weights.is_file():
        raise OpenVINOGraphExportError(
            "OpenVINO conversion did not produce the required .xml/.bin pair."
        )
    if target.stat().st_size <= 0 or weights.stat().st_size <= 0:
        raise OpenVINOGraphExportError("OpenVINO conversion produced an empty IR component.")
    return OpenVINOGraphInfo(xml_path=target, bin_path=weights)


__all__ = [
    "OpenVINOGraphExportError",
    "OpenVINOGraphInfo",
    "convert_verified_onnx_to_openvino",
]
