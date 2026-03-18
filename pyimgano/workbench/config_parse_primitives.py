from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a dict/object, got {type(value).__name__}")
    return value


def _optional_int(value: Any, *, name: str) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception as exc:  # noqa: BLE001 - validation boundary
        raise ValueError(f"{name} must be int or null, got {value!r}") from exc


def _optional_float(value: Any, *, name: str) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception as exc:  # noqa: BLE001 - validation boundary
        raise ValueError(f"{name} must be float or null, got {value!r}") from exc


def _optional_nonempty_str(value: Any, *, name: str) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be a non-empty string or null")
    return text


def _optional_bool(value: Any, *, name: str) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes", "on"}:
            return True
        if text in {"false", "0", "no", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean or null, got {value!r}")


def _optional_int_sequence(value: Any, *, name: str) -> tuple[int, ...] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list/tuple of ints or null, got {value!r}")
    if len(value) == 0:
        raise ValueError(f"{name} must be a non-empty list/tuple of ints or null")
    out: list[int] = []
    for raw in value:
        try:
            out.append(int(raw))
        except Exception as exc:  # noqa: BLE001 - validation boundary
            raise ValueError(f"{name} must contain ints, got {value!r}") from exc
    return tuple(out)


def _parse_resize(value: Any, *, default: tuple[int, int]) -> tuple[int, int]:
    if value is None:
        return (int(default[0]), int(default[1]))

    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"resize must be a list/tuple of length 2, got {value!r}")
    try:
        h = int(value[0])
        w = int(value[1])
    except Exception as exc:  # noqa: BLE001 - validation boundary
        raise ValueError(f"resize must contain ints, got {value!r}") from exc
    if h <= 0 or w <= 0:
        raise ValueError(f"resize must be positive, got {(h, w)}")
    return (h, w)


def _parse_int_pair(
    value: Any,
    *,
    name: str,
    default: tuple[int, int],
) -> tuple[int, int]:
    if value is None:
        return (int(default[0]), int(default[1]))

    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be a list/tuple of length 2, got {value!r}")
    try:
        a = int(value[0])
        b = int(value[1])
    except Exception as exc:  # noqa: BLE001 - validation boundary
        raise ValueError(f"{name} must contain ints, got {value!r}") from exc
    if a <= 0 or b <= 0:
        raise ValueError(f"{name} must be positive ints, got {(a, b)}")
    return (a, b)


def _parse_percentile_range(
    value: Any,
    *,
    default: tuple[float, float] = (1.0, 99.0),
) -> tuple[float, float]:
    if value is None:
        return (float(default[0]), float(default[1]))
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"percentile_range must be a list/tuple of length 2, got {value!r}")
    try:
        low = float(value[0])
        high = float(value[1])
    except Exception as exc:  # noqa: BLE001 - validation boundary
        raise ValueError(f"percentile_range must contain floats, got {value!r}") from exc
    return (low, high)


def _parse_checkpoint_name(value: Any, *, default: str = "model.pt") -> str:
    if value is None:
        return str(default)
    name = str(value).strip()
    if not name:
        raise ValueError("training.checkpoint_name must be a non-empty filename")
    if name in (".", ".."):
        raise ValueError("training.checkpoint_name must be a filename, got '.'/'..'")
    if "/" in name or "\\" in name:
        raise ValueError("training.checkpoint_name must be a filename, not a path")
    p = Path(name)
    if p.is_absolute() or p.name != name:
        raise ValueError("training.checkpoint_name must be a filename, not a path")
    return name


def _parse_roi_xyxy_norm(value: Any) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("defects.roi_xyxy_norm must be a list/tuple of length 4 or null")

    try:
        x1, y1, x2, y2 = (float(v) for v in value)
    except Exception as exc:  # noqa: BLE001 - validation boundary
        raise ValueError(f"defects.roi_xyxy_norm must contain floats, got {value!r}") from exc

    def _clamp01(v: float) -> float:
        return float(min(max(v, 0.0), 1.0))

    x1c, y1c, x2c, y2c = (_clamp01(x1), _clamp01(y1), _clamp01(x2), _clamp01(y2))
    return (min(x1c, x2c), min(y1c, y2c), max(x1c, x2c), max(y1c, y2c))


__all__ = [
    "_require_mapping",
    "_optional_int",
    "_optional_float",
    "_optional_nonempty_str",
    "_optional_bool",
    "_optional_int_sequence",
    "_parse_resize",
    "_parse_int_pair",
    "_parse_percentile_range",
    "_parse_checkpoint_name",
    "_parse_roi_xyxy_norm",
]
