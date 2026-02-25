from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray


ImageInput = Union[str, Path, np.ndarray]
MapReduce = Literal["max", "mean", "hann", "gaussian"]
ScoreReduce = Literal["max", "mean", "topk_mean"]


@dataclass(frozen=True)
class Tile:
    """A single tile placement in the original image coordinate space."""

    y0: int
    x0: int
    y1: int
    x1: int
    tile_size: int

    @property
    def valid_h(self) -> int:
        return int(self.y1 - self.y0)

    @property
    def valid_w(self) -> int:
        return int(self.x1 - self.x0)


def iter_tile_coords(
    height: int,
    width: int,
    *,
    tile_size: int,
    stride: int,
) -> list[tuple[int, int]]:
    """Return tile (y0, x0) coordinates covering an image.

    Ensures the last tile always touches the bottom/right edges (even when the stride
    does not divide the image size).
    """

    h = int(height)
    w = int(width)
    tile = int(tile_size)
    step = int(stride)

    if h <= 0 or w <= 0:
        raise ValueError(f"height/width must be positive, got {(h, w)}")
    if tile <= 0:
        raise ValueError(f"tile_size must be positive, got {tile}")
    if step <= 0:
        raise ValueError(f"stride must be positive, got {step}")

    if h <= tile:
        y_positions = [0]
    else:
        y_positions = list(range(0, h - tile + 1, step))
        last = h - tile
        if y_positions[-1] != last:
            y_positions.append(last)

    if w <= tile:
        x_positions = [0]
    else:
        x_positions = list(range(0, w - tile + 1, step))
        last = w - tile
        if x_positions[-1] != last:
            x_positions.append(last)

    return [(y, x) for y in y_positions for x in x_positions]


def _safe_pad(
    array: NDArray,
    pad_spec: tuple[tuple[int, int], ...],
    *,
    pad_mode: str,
    pad_value: int,
) -> NDArray:
    mode = str(pad_mode)
    if mode == "constant":
        return np.pad(array, pad_spec, mode="constant", constant_values=int(pad_value))

    try:
        return np.pad(array, pad_spec, mode=mode)
    except Exception:
        # Modes like "reflect" can fail for very small tiles (pad width constraints).
        return np.pad(array, pad_spec, mode="edge")


def extract_tile(
    image: NDArray,
    *,
    y0: int,
    x0: int,
    tile_size: int,
    pad_mode: str = "reflect",
    pad_value: int = 0,
) -> tuple[NDArray, Tile]:
    """Extract a (tile_size, tile_size) tile from an image, padding at edges as needed."""

    img = np.asarray(image)
    if img.ndim < 2:
        raise ValueError(f"image must be at least 2D, got shape {img.shape}")

    h, w = int(img.shape[0]), int(img.shape[1])
    tile = int(tile_size)
    y0i, x0i = int(y0), int(x0)
    if y0i < 0 or x0i < 0:
        raise ValueError("y0/x0 must be non-negative")

    y1 = min(y0i + tile, h)
    x1 = min(x0i + tile, w)

    tile_arr = img[y0i:y1, x0i:x1]
    pad_bottom = tile - int(tile_arr.shape[0])
    pad_right = tile - int(tile_arr.shape[1])
    if pad_bottom > 0 or pad_right > 0:
        pad_spec: list[tuple[int, int]] = [(0, pad_bottom), (0, pad_right)]
        for _ in range(int(tile_arr.ndim) - 2):
            pad_spec.append((0, 0))
        tile_arr = _safe_pad(
            tile_arr,
            tuple(pad_spec),
            pad_mode=pad_mode,
            pad_value=int(pad_value),
        )

    return tile_arr, Tile(y0=y0i, x0=x0i, y1=int(y1), x1=int(x1), tile_size=tile)


def stitch_maps(
    tiles: Sequence[Tile],
    maps: Sequence[NDArray],
    *,
    out_shape: tuple[int, int],
    reduce: MapReduce = "max",
) -> NDArray:
    """Stitch tile anomaly maps back into an image-level map."""

    if len(tiles) != len(maps):
        raise ValueError(f"tiles/maps length mismatch: {len(tiles)} vs {len(maps)}")

    out_h, out_w = int(out_shape[0]), int(out_shape[1])
    if out_h <= 0 or out_w <= 0:
        raise ValueError(f"out_shape must be positive, got {out_shape}")

    reduce_mode = str(reduce).lower()
    if reduce_mode not in ("max", "mean", "hann", "gaussian"):
        raise ValueError(
            f"Unknown reduce mode: {reduce}. Choose from: max, mean, hann, gaussian"
        )

    if reduce_mode == "max":
        out = np.full((out_h, out_w), -np.inf, dtype=np.float32)
        for tile, tile_map in zip(tiles, maps):
            arr = np.asarray(tile_map, dtype=np.float32)
            if arr.ndim != 2:
                raise ValueError(f"tile_map must be 2D, got shape {arr.shape}")
            patch = arr[: tile.valid_h, : tile.valid_w]
            region = out[tile.y0 : tile.y1, tile.x0 : tile.x1]
            out[tile.y0 : tile.y1, tile.x0 : tile.x1] = np.maximum(region, patch)

        out[~np.isfinite(out)] = 0.0
        return out

    def _window(kind: str, tile_size: int) -> NDArray:
        t = int(tile_size)
        if t <= 0:
            raise ValueError(f"tile_size must be positive, got {t}")

        if kind == "mean":
            return np.ones((t, t), dtype=np.float32)

        if kind == "hann":
            w = np.hanning(t).astype(np.float32)
            if float(w.sum()) <= 0:  # pragma: no cover - tiny t edge cases
                w = np.ones((t,), dtype=np.float32)
            # Avoid exact zeros at borders to prevent division-by-zero when a
            # border pixel is only covered by one tile.
            w = np.maximum(w, 1e-3)
            return np.outer(w, w).astype(np.float32)

        if kind == "gaussian":
            if t == 1:
                return np.ones((1, 1), dtype=np.float32)
            yy, xx = np.mgrid[0:t, 0:t].astype(np.float32)
            cy = (t - 1) / 2.0
            cx = (t - 1) / 2.0
            y = (yy - cy) / max(cy, 1.0)
            x = (xx - cx) / max(cx, 1.0)
            r2 = x * x + y * y
            sigma2 = 0.5 * 0.5
            w = np.exp(-0.5 * r2 / sigma2).astype(np.float32)
            w = np.maximum(w, 1e-3)
            return w

        raise ValueError(f"Unknown window kind: {kind}")

    weights = _window(reduce_mode, int(tiles[0].tile_size)) if tiles else None

    accum = np.zeros((out_h, out_w), dtype=np.float32)
    weight_sum = np.zeros((out_h, out_w), dtype=np.float32)
    for tile, tile_map in zip(tiles, maps):
        arr = np.asarray(tile_map, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"tile_map must be 2D, got shape {arr.shape}")

        patch = arr[: tile.valid_h, : tile.valid_w]

        w = weights
        if w is None:  # pragma: no cover
            w_patch = 1.0
        else:
            if w.shape != (tile.tile_size, tile.tile_size):
                w = _window(reduce_mode, int(tile.tile_size))
            w_patch = w[: tile.valid_h, : tile.valid_w]

        region = (slice(tile.y0, tile.y1), slice(tile.x0, tile.x1))
        accum[region] += patch * w_patch
        weight_sum[region] += w_patch

    weight_sum = np.maximum(weight_sum, 1e-6)
    return accum / weight_sum


def _reduce_scores(scores: NDArray, *, mode: ScoreReduce, topk: float) -> float:
    arr = np.asarray(scores, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        raise ValueError("scores must be non-empty")

    mode_lower = str(mode).lower()
    if mode_lower == "max":
        return float(np.max(arr))
    if mode_lower == "mean":
        return float(np.mean(arr))
    if mode_lower == "topk_mean":
        frac = float(topk)
        if not (0.0 < frac <= 1.0):
            raise ValueError(f"topk must be in (0,1], got {topk}")
        k = max(1, int(np.ceil(arr.size * frac)))
        topk_vals = np.partition(arr, -k)[-k:]
        return float(np.mean(topk_vals))

    raise ValueError(f"Unknown score reduce mode: {mode}. Choose from: max, mean, topk_mean")


def _load_rgb_u8_hwc_from_path(path: Union[str, Path]) -> NDArray:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "opencv-python is required to load images from disk for tiled inference.\n"
            "Install it via:\n  pip install 'opencv-python'\n"
            f"Original error: {exc}"
        ) from exc

    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        raise ValueError(f"Failed to load image: {path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return np.asarray(img_rgb, dtype=np.uint8)


def _resize_map(anomaly_map: NDArray, *, target_h: int, target_w: int) -> NDArray:
    arr = np.asarray(anomaly_map, dtype=np.float32)
    if arr.shape == (int(target_h), int(target_w)):
        return arr

    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "opencv-python is required to resize anomaly maps.\n"
            "Install it via:\n  pip install 'opencv-python'\n"
            f"Original error: {exc}"
        ) from exc

    resized = cv2.resize(arr, (int(target_w), int(target_h)), interpolation=cv2.INTER_LINEAR)
    return np.asarray(resized, dtype=np.float32)


class TiledDetector:
    """Wrap a detector to run inference on overlapping tiles for high-resolution images."""

    def __init__(
        self,
        *,
        detector,
        tile_size: int = 512,
        stride: Optional[int] = None,
        pad_mode: str = "reflect",
        pad_value: int = 0,
        score_reduce: ScoreReduce = "max",
        score_topk: float = 0.1,
        map_reduce: MapReduce = "max",
    ) -> None:
        self.detector = detector
        self.tile_size = int(tile_size)
        self.stride = int(stride) if stride is not None else int(tile_size)
        self.pad_mode = str(pad_mode)
        self.pad_value = int(pad_value)
        self.score_reduce = score_reduce
        self.score_topk = float(score_topk)
        self.map_reduce = map_reduce

        if self.tile_size <= 0:
            raise ValueError(f"tile_size must be positive, got {self.tile_size}")
        if self.stride <= 0:
            raise ValueError(f"stride must be positive, got {self.stride}")

        # Cache tile coordinate grids per (H,W,tile_size,stride) to avoid recomputing
        # them for repeated inputs of the same size.
        self._tile_coords_cache: dict[
            tuple[int, int, int, int], tuple[tuple[int, int], ...]
        ] = {}

    def fit(self, X, y=None, **kwargs):
        try:
            return self.detector.fit(X, y=y, **kwargs)
        except TypeError:
            return self.detector.fit(X, y=y)

    def _to_image_array(self, item: ImageInput) -> NDArray:
        if isinstance(item, (str, Path)):
            return _load_rgb_u8_hwc_from_path(item)
        return np.asarray(item)

    def _iter_tiles(self, image: NDArray) -> tuple[list[NDArray], list[Tile]]:
        img = np.asarray(image)
        h, w = int(img.shape[0]), int(img.shape[1])
        cache_key = (h, w, int(self.tile_size), int(self.stride))
        coords = self._tile_coords_cache.get(cache_key)
        if coords is None:
            coords = tuple(iter_tile_coords(h, w, tile_size=self.tile_size, stride=self.stride))
            self._tile_coords_cache[cache_key] = coords

        tiles: list[NDArray] = []
        infos: list[Tile] = []
        for y0, x0 in coords:
            tile_arr, tile = extract_tile(
                img,
                y0=y0,
                x0=x0,
                tile_size=self.tile_size,
                pad_mode=self.pad_mode,
                pad_value=self.pad_value,
            )
            tiles.append(np.asarray(tile_arr))
            infos.append(tile)
        return tiles, infos

    def _call_decision_function(self, tiles: list[NDArray]) -> NDArray:
        # Try list-style first, then batched ndarray style.
        try:
            scores = self.detector.decision_function(tiles)
            return np.asarray(scores, dtype=np.float32).reshape(-1)
        except Exception:
            pass

        batch = np.stack(tiles, axis=0)
        scores = self.detector.decision_function(batch)
        return np.asarray(scores, dtype=np.float32).reshape(-1)

    def decision_function(self, X: Iterable[ImageInput]) -> NDArray:
        items = list(X)
        scores_out = np.zeros(len(items), dtype=np.float32)
        for i, item in enumerate(items):
            image = self._to_image_array(item)
            tiles, _infos = self._iter_tiles(image)
            tile_scores = self._call_decision_function(tiles)
            scores_out[i] = _reduce_scores(
                tile_scores, mode=self.score_reduce, topk=self.score_topk
            )
        return scores_out

    def _call_predict_anomaly_map(self, tiles: list[NDArray]) -> list[NDArray]:
        if hasattr(self.detector, "predict_anomaly_map"):
            try:
                maps = self.detector.predict_anomaly_map(tiles)
            except Exception:
                maps = self.detector.predict_anomaly_map(np.stack(tiles, axis=0))

            arr = np.asarray(maps)
            if arr.ndim == 3 and arr.shape[0] == len(tiles):
                return [np.asarray(arr[i], dtype=np.float32) for i in range(arr.shape[0])]

        if hasattr(self.detector, "get_anomaly_map"):
            out: list[NDArray] = []
            for tile in tiles:
                out.append(np.asarray(self.detector.get_anomaly_map(tile), dtype=np.float32))
            return out

        raise ValueError("Wrapped detector does not expose predict_anomaly_map/get_anomaly_map")

    def get_anomaly_map(self, item: ImageInput) -> NDArray:
        image = self._to_image_array(item)
        h, w = int(image.shape[0]), int(image.shape[1])

        tiles, infos = self._iter_tiles(image)
        tile_maps = self._call_predict_anomaly_map(tiles)
        if len(tile_maps) != len(infos):
            raise ValueError(
                "predict_anomaly_map must return one map per tile. "
                f"Got {len(tile_maps)} maps for {len(infos)} tiles."
            )

        normalized_maps: list[NDArray] = []
        for tile, tile_map in zip(infos, tile_maps):
            arr = np.asarray(tile_map, dtype=np.float32)
            arr = _resize_map(arr, target_h=self.tile_size, target_w=self.tile_size)
            normalized_maps.append(arr[: tile.valid_h, : tile.valid_w])

        stitched = stitch_maps(
            infos,
            normalized_maps,
            out_shape=(h, w),
            reduce=self.map_reduce,
        )
        return np.asarray(stitched, dtype=np.float32)

    def predict_anomaly_map(self, X: Iterable[ImageInput]) -> NDArray:
        items = list(X)
        maps = [self.get_anomaly_map(item) for item in items]
        if not maps:
            raise ValueError("X must be non-empty")

        first_shape = maps[0].shape
        for m in maps[1:]:
            if m.shape != first_shape:
                raise ValueError(
                    "Inconsistent anomaly map shapes; cannot stack. "
                    f"Expected {first_shape}, got {m.shape}."
                )
        return np.stack(maps, axis=0)

    def __getattr__(self, name: str):
        # Proxy unknown attributes (e.g. threshold_, decision_scores_) to the wrapped detector.
        return getattr(self.detector, name)
