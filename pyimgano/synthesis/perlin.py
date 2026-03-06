from __future__ import annotations

import math

import numpy as np


def _fade(t: np.ndarray) -> np.ndarray:
    # Ken Perlin's improved fade curve: 6t^5 - 15t^4 + 10t^3
    t = np.asarray(t, dtype=np.float32)
    return ((6.0 * t - 15.0) * t + 10.0) * t * t * t


def _lerp(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.float32) + np.asarray(t, dtype=np.float32) * (
        np.asarray(b, dtype=np.float32) - np.asarray(a, dtype=np.float32)
    )


def perlin_noise_2d(
    shape_hw: tuple[int, int],
    res_hw: tuple[int, int],
    *,
    rng: np.random.Generator,
    normalize: bool = True,
) -> np.ndarray:
    """Generate 2D Perlin noise (float32).

    Parameters
    ----------
    shape_hw:
        Output shape (H, W).
    res_hw:
        Number of noise grid cells (res_y, res_x). Higher = more high-frequency.
    rng:
        Numpy random Generator (deterministic given the seed).
    normalize:
        If True, rescale output to [0,1].

    Returns
    -------
    noise:
        Array of shape (H, W), dtype float32.
    """

    h, w = int(shape_hw[0]), int(shape_hw[1])
    ry, rx = int(res_hw[0]), int(res_hw[1])
    if h <= 0 or w <= 0:
        raise ValueError(f"shape_hw must be positive, got {shape_hw!r}")
    if ry <= 0 or rx <= 0:
        raise ValueError(f"res_hw must be positive, got {res_hw!r}")

    # Coordinates in "grid space".
    # endpoint=False ensures indices stay within [0, res-1].
    ys = np.linspace(0.0, float(ry), h, endpoint=False, dtype=np.float32)
    xs = np.linspace(0.0, float(rx), w, endpoint=False, dtype=np.float32)

    yi = np.floor(ys).astype(np.int32)
    xi = np.floor(xs).astype(np.int32)
    yf = ys - yi.astype(np.float32)
    xf = xs - xi.astype(np.float32)

    gy, gx = np.meshgrid(yi, xi, indexing="ij")
    fy, fx = np.meshgrid(yf, xf, indexing="ij")

    # Random unit gradient vectors for each grid vertex.
    angles = rng.uniform(0.0, 2.0 * math.pi, size=(ry + 1, rx + 1)).astype(np.float32)
    grads = np.stack((np.cos(angles), np.sin(angles)), axis=-1).astype(np.float32)

    g00 = grads[gy, gx]
    g10 = grads[gy + 1, gx]
    g01 = grads[gy, gx + 1]
    g11 = grads[gy + 1, gx + 1]

    d00 = np.stack((fy, fx), axis=-1)
    d10 = np.stack((fy - 1.0, fx), axis=-1)
    d01 = np.stack((fy, fx - 1.0), axis=-1)
    d11 = np.stack((fy - 1.0, fx - 1.0), axis=-1)

    s = np.sum(g00 * d00, axis=-1)
    t = np.sum(g10 * d10, axis=-1)
    u = np.sum(g01 * d01, axis=-1)
    v = np.sum(g11 * d11, axis=-1)

    wx = _fade(fx)
    wy = _fade(fy)

    a = _lerp(s, u, wx)
    b = _lerp(t, v, wx)
    out = _lerp(a, b, wy).astype(np.float32)

    if normalize:
        lo = float(np.min(out))
        hi = float(np.max(out))
        denom = hi - lo
        if denom <= 1e-12:
            out = np.zeros_like(out, dtype=np.float32)
        else:
            out = (out - lo) / denom

    return np.asarray(out, dtype=np.float32)


def fractal_perlin_noise_2d(
    shape_hw: tuple[int, int],
    res_hw: tuple[int, int],
    *,
    rng: np.random.Generator,
    octaves: int = 3,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    normalize: bool = True,
) -> np.ndarray:
    """Fractal Brownian motion (fBm) style Perlin noise."""

    h, w = int(shape_hw[0]), int(shape_hw[1])
    if h <= 0 or w <= 0:
        raise ValueError(f"shape_hw must be positive, got {shape_hw!r}")

    o = int(octaves)
    if o < 1:
        raise ValueError("octaves must be >= 1")

    amp = 1.0
    freq_y, freq_x = float(res_hw[0]), float(res_hw[1])
    noise = np.zeros((h, w), dtype=np.float32)
    amp_sum = 0.0

    for _ in range(o):
        ry = max(1, int(round(freq_y)))
        rx = max(1, int(round(freq_x)))
        # Clamp to keep the gradient grid reasonably sized.
        ry = min(ry, max(1, h))
        rx = min(rx, max(1, w))

        noise = noise + amp * perlin_noise_2d((h, w), (ry, rx), rng=rng, normalize=False)
        amp_sum += amp
        amp *= float(persistence)
        freq_y *= float(lacunarity)
        freq_x *= float(lacunarity)

    if amp_sum > 0.0:
        noise = noise / float(amp_sum)

    if normalize:
        lo = float(np.min(noise))
        hi = float(np.max(noise))
        denom = hi - lo
        if denom <= 1e-12:
            noise = np.zeros_like(noise, dtype=np.float32)
        else:
            noise = (noise - lo) / denom

    return np.asarray(noise, dtype=np.float32)
