"""
Image augmentation techniques for data augmentation and robustness.

This module provides comprehensive augmentation operations including:
- Geometric transformations (rotation, flip, scale, shear, perspective)
- Color augmentations (brightness, contrast, saturation, hue)
- Noise addition (Gaussian, salt-and-pepper, Poisson, speckle)
- Blur operations (motion blur, defocus blur, glass blur)
- Weather effects (rain, fog, snow, shadow)
- Cutout and occlusion
- Elastic and grid distortions
- Advanced augmentations (Mixup, CutMix)
"""

import random
from enum import Enum
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from scipy.ndimage import map_coordinates


# Enums for type-safe augmentation selection

class GeometricTransform(Enum):
    """Geometric transformation types."""
    ROTATE = "rotate"
    FLIP_HORIZONTAL = "flip_horizontal"
    FLIP_VERTICAL = "flip_vertical"
    FLIP_BOTH = "flip_both"
    SCALE = "scale"
    TRANSLATE = "translate"
    SHEAR = "shear"
    PERSPECTIVE = "perspective"
    AFFINE = "affine"


class NoiseType(Enum):
    """Noise types."""
    GAUSSIAN = "gaussian"
    SALT_PEPPER = "salt_pepper"
    POISSON = "poisson"
    SPECKLE = "speckle"
    UNIFORM = "uniform"


class BlurType(Enum):
    """Blur types for augmentation."""
    MOTION = "motion"
    DEFOCUS = "defocus"
    GLASS = "glass"
    ZOOM = "zoom"


class WeatherEffect(Enum):
    """Weather effects."""
    RAIN = "rain"
    FOG = "fog"
    SNOW = "snow"
    SHADOW = "shadow"
    SUNFLARE = "sunflare"


# Geometric Transformations

def rotate_image(
    image: NDArray,
    angle: float,
    center: Optional[Tuple[int, int]] = None,
    scale: float = 1.0,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value: int = 0
) -> NDArray:
    """
    Rotate image by specified angle.

    Args:
        image: Input image
        angle: Rotation angle in degrees (positive = counter-clockwise)
        center: Center of rotation (default: image center)
        scale: Scaling factor
        border_mode: Border mode for pixel extrapolation
        border_value: Border value if border_mode is BORDER_CONSTANT

    Returns:
        Rotated image
    """
    h, w = image.shape[:2]
    if center is None:
        center = (w // 2, h // 2)

    # Get rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, scale)

    # Rotate image
    rotated = cv2.warpAffine(image, M, (w, h), borderMode=border_mode, borderValue=border_value)

    return rotated


def flip_image(image: NDArray, mode: Union[str, GeometricTransform] = "horizontal") -> NDArray:
    """
    Flip image horizontally, vertically, or both.

    Args:
        image: Input image
        mode: Flip mode ('horizontal', 'vertical', 'both')

    Returns:
        Flipped image
    """
    if isinstance(mode, str):
        if mode == "horizontal" or mode == "flip_horizontal":
            return cv2.flip(image, 1)
        elif mode == "vertical" or mode == "flip_vertical":
            return cv2.flip(image, 0)
        elif mode == "both" or mode == "flip_both":
            return cv2.flip(image, -1)

    raise ValueError(f"Unknown flip mode: {mode}")


def scale_image(
    image: NDArray,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    keep_size: bool = True
) -> NDArray:
    """
    Scale image by specified factors.

    Args:
        image: Input image
        scale_x: Horizontal scaling factor
        scale_y: Vertical scaling factor
        keep_size: If True, keep original image size (crop/pad as needed)

    Returns:
        Scaled image
    """
    h, w = image.shape[:2]
    new_w = int(w * scale_x)
    new_h = int(h * scale_y)

    scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    if keep_size:
        if new_h > h or new_w > w:
            # Crop to original size
            start_y = (new_h - h) // 2
            start_x = (new_w - w) // 2
            scaled = scaled[start_y:start_y + h, start_x:start_x + w]
        else:
            # Pad to original size
            pad_y = (h - new_h) // 2
            pad_x = (w - new_w) // 2
            result = np.zeros_like(image)
            result[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = scaled
            scaled = result

    return scaled


def translate_image(
    image: NDArray,
    tx: int,
    ty: int,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value: int = 0
) -> NDArray:
    """
    Translate image by specified offsets.

    Args:
        image: Input image
        tx: Horizontal translation (pixels)
        ty: Vertical translation (pixels)
        border_mode: Border mode for pixel extrapolation
        border_value: Border value if border_mode is BORDER_CONSTANT

    Returns:
        Translated image
    """
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    h, w = image.shape[:2]
    translated = cv2.warpAffine(image, M, (w, h), borderMode=border_mode, borderValue=border_value)

    return translated


def shear_image(
    image: NDArray,
    shear_x: float = 0.0,
    shear_y: float = 0.0,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value: int = 0
) -> NDArray:
    """
    Apply shear transformation to image.

    Args:
        image: Input image
        shear_x: Horizontal shear factor
        shear_y: Vertical shear factor
        border_mode: Border mode
        border_value: Border value

    Returns:
        Sheared image
    """
    h, w = image.shape[:2]

    # Shear matrix
    M = np.float32([[1, shear_x, 0], [shear_y, 1, 0]])

    sheared = cv2.warpAffine(image, M, (w, h), borderMode=border_mode, borderValue=border_value)

    return sheared


def perspective_transform(
    image: NDArray,
    strength: float = 0.2,
    random_seed: Optional[int] = None
) -> NDArray:
    """
    Apply random perspective transformation.

    Args:
        image: Input image
        strength: Strength of perspective distortion (0-1)
        random_seed: Random seed for reproducibility

    Returns:
        Perspective-transformed image
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    h, w = image.shape[:2]

    # Source points (corners)
    src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    # Destination points with random perturbation
    max_offset = int(min(w, h) * strength)
    dst_points = src_points + np.random.randint(-max_offset, max_offset, src_points.shape).astype(np.float32)

    # Get perspective transform matrix
    M = cv2.getPerspectiveTransform(src_points, dst_points)

    # Apply transformation
    transformed = cv2.warpPerspective(image, M, (w, h))

    return transformed


# Color Augmentations

def adjust_brightness(image: NDArray, factor: float) -> NDArray:
    """
    Adjust image brightness.

    Args:
        image: Input image
        factor: Brightness factor (< 1.0: darker, > 1.0: brighter)

    Returns:
        Brightness-adjusted image
    """
    adjusted = cv2.convertScaleAbs(image, alpha=factor, beta=0)
    return adjusted


def adjust_contrast(image: NDArray, factor: float) -> NDArray:
    """
    Adjust image contrast.

    Args:
        image: Input image
        factor: Contrast factor (< 1.0: lower contrast, > 1.0: higher contrast)

    Returns:
        Contrast-adjusted image
    """
    mean = image.mean()
    adjusted = cv2.convertScaleAbs(image, alpha=factor, beta=mean * (1 - factor))
    return adjusted


def adjust_saturation(image: NDArray, factor: float) -> NDArray:
    """
    Adjust color saturation (only for color images).

    Args:
        image: Input color image (BGR)
        factor: Saturation factor (0: grayscale, 1: original, > 1: more saturated)

    Returns:
        Saturation-adjusted image
    """
    if len(image.shape) != 3:
        return image

    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Adjust saturation
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)

    # Convert back to BGR
    hsv = hsv.astype(np.uint8)
    adjusted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return adjusted


def adjust_hue(image: NDArray, delta: float) -> NDArray:
    """
    Adjust color hue.

    Args:
        image: Input color image (BGR)
        delta: Hue shift (-180 to 180 degrees)

    Returns:
        Hue-adjusted image
    """
    if len(image.shape) != 3:
        return image

    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Adjust hue (with wrapping)
    hsv[:, :, 0] = (hsv[:, :, 0] + delta) % 180

    # Convert back to BGR
    hsv = hsv.astype(np.uint8)
    adjusted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return adjusted


def color_jitter(
    image: NDArray,
    brightness: Tuple[float, float] = (0.8, 1.2),
    contrast: Tuple[float, float] = (0.8, 1.2),
    saturation: Tuple[float, float] = (0.8, 1.2),
    hue: Tuple[float, float] = (-10, 10)
) -> NDArray:
    """
    Apply random color jittering.

    Args:
        image: Input image
        brightness: Range for brightness factor
        contrast: Range for contrast factor
        saturation: Range for saturation factor
        hue: Range for hue shift (degrees)

    Returns:
        Color-jittered image
    """
    # Random brightness
    b_factor = np.random.uniform(brightness[0], brightness[1])
    result = adjust_brightness(image, b_factor)

    # Random contrast
    c_factor = np.random.uniform(contrast[0], contrast[1])
    result = adjust_contrast(result, c_factor)

    # Random saturation (only for color images)
    if len(image.shape) == 3:
        s_factor = np.random.uniform(saturation[0], saturation[1])
        result = adjust_saturation(result, s_factor)

        # Random hue
        h_delta = np.random.uniform(hue[0], hue[1])
        result = adjust_hue(result, h_delta)

    return result


# Noise Addition

def add_gaussian_noise(
    image: NDArray,
    mean: float = 0,
    std: float = 25
) -> NDArray:
    """
    Add Gaussian noise to image.

    Args:
        image: Input image
        mean: Mean of Gaussian noise
        std: Standard deviation of Gaussian noise

    Returns:
        Noisy image
    """
    noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)

    return noisy


def add_salt_pepper_noise(
    image: NDArray,
    salt_prob: float = 0.01,
    pepper_prob: float = 0.01
) -> NDArray:
    """
    Add salt-and-pepper noise to image.

    Args:
        image: Input image
        salt_prob: Probability of salt noise (white pixels)
        pepper_prob: Probability of pepper noise (black pixels)

    Returns:
        Noisy image
    """
    noisy = image.copy()

    # Salt noise (white)
    salt_mask = np.random.random(image.shape[:2]) < salt_prob
    noisy[salt_mask] = 255

    # Pepper noise (black)
    pepper_mask = np.random.random(image.shape[:2]) < pepper_prob
    noisy[pepper_mask] = 0

    return noisy


def add_poisson_noise(image: NDArray) -> NDArray:
    """
    Add Poisson noise to image.

    Args:
        image: Input image

    Returns:
        Noisy image
    """
    # Normalize to 0-1
    normalized = image.astype(np.float32) / 255.0

    # Add Poisson noise
    noisy = np.random.poisson(normalized * 255) / 255.0

    # Convert back to uint8
    noisy = np.clip(noisy * 255, 0, 255).astype(np.uint8)

    return noisy


def add_speckle_noise(image: NDArray, std: float = 0.1) -> NDArray:
    """
    Add speckle (multiplicative) noise to image.

    Args:
        image: Input image
        std: Standard deviation of speckle noise

    Returns:
        Noisy image
    """
    noise = np.random.randn(*image.shape) * std
    noisy = image.astype(np.float32) * (1 + noise)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)

    return noisy


# Blur Operations

def motion_blur(
    image: NDArray,
    kernel_size: int = 15,
    angle: float = 0
) -> NDArray:
    """
    Apply motion blur to image.

    Args:
        image: Input image
        kernel_size: Size of motion blur kernel
        angle: Angle of motion blur (degrees)

    Returns:
        Motion-blurred image
    """
    # Create motion blur kernel
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, :] = 1
    kernel = kernel / kernel_size

    # Rotate kernel
    if angle != 0:
        center = (kernel_size // 2, kernel_size // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))

    # Apply blur
    blurred = cv2.filter2D(image, -1, kernel)

    return blurred


def defocus_blur(image: NDArray, radius: int = 5) -> NDArray:
    """
    Apply defocus blur (circular kernel).

    Args:
        image: Input image
        radius: Radius of defocus blur

    Returns:
        Defocused image
    """
    kernel_size = radius * 2 + 1
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)

    # Create circular kernel
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = x**2 + y**2 <= radius**2
    kernel[mask] = 1
    kernel = kernel / kernel.sum()

    # Apply blur
    blurred = cv2.filter2D(image, -1, kernel)

    return blurred


def glass_blur(image: NDArray, iterations: int = 2, kernel_size: int = 5) -> NDArray:
    """
    Apply glass blur effect (local pixel shuffling).

    Args:
        image: Input image
        iterations: Number of iterations
        kernel_size: Size of local shuffling window

    Returns:
        Glass-blurred image
    """
    h, w = image.shape[:2]
    result = image.copy()

    for _ in range(iterations):
        # Random offsets
        dx = np.random.randint(-kernel_size, kernel_size, (h, w))
        dy = np.random.randint(-kernel_size, kernel_size, (h, w))

        # Create coordinate arrays
        x = np.arange(w)
        y = np.arange(h)
        xx, yy = np.meshgrid(x, y)

        # Apply offsets with boundary checking
        xx_new = np.clip(xx + dx, 0, w - 1)
        yy_new = np.clip(yy + dy, 0, h - 1)

        # Shuffle pixels
        if len(image.shape) == 3:
            for c in range(image.shape[2]):
                result[:, :, c] = result[yy_new, xx_new, c]
        else:
            result = result[yy_new, xx_new]

    return result


# Weather Effects

def add_rain(
    image: NDArray,
    intensity: float = 0.5,
    length: int = 20,
    angle: float = -30,
    num_drops: Optional[int] = None
) -> NDArray:
    """
    Add rain effect to image.

    Args:
        image: Input image
        intensity: Rain intensity (0-1)
        length: Length of rain drops
        angle: Angle of rain (degrees)
        num_drops: Number of rain drops (auto if None)

    Returns:
        Image with rain effect
    """
    h, w = image.shape[:2]
    result = image.copy().astype(np.float32)

    if num_drops is None:
        num_drops = int(h * w * intensity / 100)

    # Generate random rain drops
    for _ in range(num_drops):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)

        # Calculate end point based on angle and length
        end_x = int(x + length * np.cos(np.radians(angle)))
        end_y = int(y + length * np.sin(np.radians(angle)))

        # Draw rain drop
        color = (200, 200, 200) if len(image.shape) == 3 else 200
        thickness = 1
        cv2.line(result, (x, y), (end_x, end_y), color, thickness)

    # Blend with original
    result = cv2.addWeighted(image.astype(np.float32), 0.7, result, 0.3, 0)
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result


def add_fog(image: NDArray, intensity: float = 0.5) -> NDArray:
    """
    Add fog effect to image.

    Args:
        image: Input image
        intensity: Fog intensity (0-1)

    Returns:
        Image with fog effect
    """
    # Create fog layer (white)
    fog = np.ones_like(image, dtype=np.float32) * 255

    # Blend with original
    result = cv2.addWeighted(image.astype(np.float32), 1 - intensity, fog, intensity, 0)
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result


def add_snow(
    image: NDArray,
    intensity: float = 0.5,
    num_flakes: Optional[int] = None
) -> NDArray:
    """
    Add snow effect to image.

    Args:
        image: Input image
        intensity: Snow intensity (0-1)
        num_flakes: Number of snowflakes (auto if None)

    Returns:
        Image with snow effect
    """
    h, w = image.shape[:2]
    result = image.copy()

    if num_flakes is None:
        num_flakes = int(h * w * intensity / 50)

    # Add snowflakes
    for _ in range(num_flakes):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        size = np.random.randint(1, 4)

        color = (255, 255, 255) if len(image.shape) == 3 else 255
        cv2.circle(result, (x, y), size, color, -1)

    return result


def add_shadow(
    image: NDArray,
    num_shadows: int = 1,
    intensity: float = 0.5
) -> NDArray:
    """
    Add random shadow effects to image.

    Args:
        image: Input image
        num_shadows: Number of shadow regions
        intensity: Shadow darkness (0-1)

    Returns:
        Image with shadows
    """
    h, w = image.shape[:2]
    result = image.copy().astype(np.float32)

    for _ in range(num_shadows):
        # Random polygon for shadow
        num_points = np.random.randint(3, 6)
        points = np.random.randint(0, [w, h], (num_points, 2))

        # Create mask
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)

        # Apply shadow
        shadow_factor = 1 - intensity
        if len(image.shape) == 3:
            for c in range(3):
                result[:, :, c] = np.where(mask == 255, result[:, :, c] * shadow_factor, result[:, :, c])
        else:
            result = np.where(mask == 255, result * shadow_factor, result)

    result = np.clip(result, 0, 255).astype(np.uint8)
    return result


# Cutout and Occlusion

def random_cutout(
    image: NDArray,
    num_holes: int = 1,
    hole_size: Union[int, Tuple[int, int]] = 32,
    fill_value: int = 0
) -> NDArray:
    """
    Apply random cutout (random erasing) to image.

    Args:
        image: Input image
        num_holes: Number of cutout holes
        hole_size: Size of holes (int or (height, width))
        fill_value: Fill value for holes

    Returns:
        Image with cutout
    """
    h, w = image.shape[:2]
    result = image.copy()

    if isinstance(hole_size, int):
        hole_h = hole_w = hole_size
    else:
        hole_h, hole_w = hole_size

    for _ in range(num_holes):
        y = np.random.randint(0, h - hole_h + 1)
        x = np.random.randint(0, w - hole_w + 1)

        result[y:y + hole_h, x:x + hole_w] = fill_value

    return result


def grid_mask(
    image: NDArray,
    grid_size: int = 32,
    ratio: float = 0.5,
    fill_value: int = 0
) -> NDArray:
    """
    Apply grid mask augmentation.

    Args:
        image: Input image
        grid_size: Size of grid cells
        ratio: Ratio of masked area
        fill_value: Fill value for masked regions

    Returns:
        Grid-masked image
    """
    h, w = image.shape[:2]
    result = image.copy()

    mask_size = int(grid_size * ratio)

    for y in range(0, h, grid_size):
        for x in range(0, w, grid_size):
            if np.random.random() > 0.5:
                y_end = min(y + mask_size, h)
                x_end = min(x + mask_size, w)
                result[y:y_end, x:x_end] = fill_value

    return result


# Elastic and Grid Distortions

def elastic_transform(
    image: NDArray,
    alpha: float = 100,
    sigma: float = 10,
    random_seed: Optional[int] = None
) -> NDArray:
    """
    Apply elastic deformation to image.

    Args:
        image: Input image
        alpha: Scaling factor for deformation
        sigma: Smoothing factor for deformation
        random_seed: Random seed

    Returns:
        Elastically deformed image
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    h, w = image.shape[:2]

    # Random displacement fields
    dx = np.random.randn(h, w) * alpha
    dy = np.random.randn(h, w) * alpha

    # Smooth displacement fields
    dx = ndimage.gaussian_filter(dx, sigma)
    dy = ndimage.gaussian_filter(dy, sigma)

    # Create coordinate arrays
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    indices = (y + dy, x + dx)

    # Apply transformation
    if len(image.shape) == 3:
        result = np.zeros_like(image)
        for c in range(image.shape[2]):
            result[:, :, c] = map_coordinates(image[:, :, c], indices, order=1, mode='reflect')
    else:
        result = map_coordinates(image, indices, order=1, mode='reflect')

    return result.astype(np.uint8)


def grid_distortion(
    image: NDArray,
    num_steps: int = 5,
    distort_limit: float = 0.3
) -> NDArray:
    """
    Apply grid distortion to image.

    Args:
        image: Input image
        num_steps: Number of grid steps
        distort_limit: Distortion strength (0-1)

    Returns:
        Grid-distorted image
    """
    h, w = image.shape[:2]

    # Create grid
    x_step = w // num_steps
    y_step = h // num_steps

    # Source points (grid corners)
    src_points = []
    for i in range(num_steps + 1):
        for j in range(num_steps + 1):
            src_points.append([j * x_step, i * y_step])

    src_points = np.array(src_points, dtype=np.float32)

    # Destination points (distorted grid)
    dst_points = src_points.copy()
    max_distort = int(min(x_step, y_step) * distort_limit)

    for i in range(len(dst_points)):
        if i % (num_steps + 1) != 0 and i % (num_steps + 1) != num_steps:  # Not on border
            if i // (num_steps + 1) != 0 and i // (num_steps + 1) != num_steps:
                dst_points[i] += np.random.uniform(-max_distort, max_distort, 2)

    # Apply piecewise affine transformation
    result = image.copy()

    # Note: Full piecewise affine requires more complex implementation
    # This is a simplified version using perspective transform
    # For production, consider using albumentations library

    return result


# Advanced Augmentations

def mixup(
    image1: NDArray,
    image2: NDArray,
    alpha: float = 0.5
) -> NDArray:
    """
    Apply Mixup augmentation (blend two images).

    Args:
        image1: First image
        image2: Second image
        alpha: Mixing factor (0-1)

    Returns:
        Mixed image
    """
    # Ensure same size
    if image1.shape != image2.shape:
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

    # Blend images
    mixed = cv2.addWeighted(image1, alpha, image2, 1 - alpha, 0)

    return mixed


def cutmix(
    image1: NDArray,
    image2: NDArray,
    alpha: float = 0.5
) -> NDArray:
    """
    Apply CutMix augmentation (cut and paste patches).

    Args:
        image1: First image
        image2: Second image
        alpha: Ratio of patch size

    Returns:
        CutMix image
    """
    # Ensure same size
    if image1.shape != image2.shape:
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

    h, w = image1.shape[:2]

    # Random box
    cut_h = int(h * alpha)
    cut_w = int(w * alpha)

    cx = np.random.randint(0, w)
    cy = np.random.randint(0, h)

    x1 = np.clip(cx - cut_w // 2, 0, w)
    y1 = np.clip(cy - cut_h // 2, 0, h)
    x2 = np.clip(cx + cut_w // 2, 0, w)
    y2 = np.clip(cy + cut_h // 2, 0, h)

    # Copy patch from image2 to image1
    result = image1.copy()
    result[y1:y2, x1:x2] = image2[y1:y2, x1:x2]

    return result


def jpeg_compress(image: NDArray, quality: int = 80) -> NDArray:
    """Simulate JPEG compression artifacts by encode/decode.

    Notes
    -----
    - This uses Pillow to avoid BGR/RGB ambiguity in OpenCV JPEG encoding.
    - Output dtype is always uint8.
    """

    q = int(quality)
    if not (1 <= q <= 95):
        raise ValueError(f"quality must be in [1,95], got {quality}")

    arr = np.asarray(image)
    if arr.ndim == 2:
        mode = "L"
        arr_u8 = arr.astype(np.uint8, copy=False)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        mode = "RGB"
        arr_u8 = arr.astype(np.uint8, copy=False)
    else:
        raise ValueError(f"Expected grayscale (H,W) or color (H,W,3) image, got {arr.shape}")

    try:
        from io import BytesIO
        from PIL import Image
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "Pillow is required for JPEG compression augmentation.\n"
            "Install it via:\n  pip install 'pillow'\n"
            f"Original error: {exc}"
        ) from exc

    pil = Image.fromarray(arr_u8, mode=mode)
    buf = BytesIO()
    pil.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    decoded = Image.open(buf)
    decoded = decoded.convert(mode)
    out = np.asarray(decoded, dtype=np.uint8)
    return out


def vignette(image: NDArray, strength: float = 0.5, exponent: float = 2.0) -> NDArray:
    """Apply a simple radial vignetting effect (darkened corners)."""

    s = float(strength)
    if s < 0.0:
        raise ValueError(f"strength must be >= 0, got {strength}")

    arr = np.asarray(image)
    if arr.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D image, got {arr.shape}")

    h, w = int(arr.shape[0]), int(arr.shape[1])
    if h == 0 or w == 0:
        return arr

    yy, xx = np.ogrid[:h, :w]
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    ny = (yy - cy) / max(cy, 1.0)
    nx = (xx - cx) / max(cx, 1.0)
    r2 = nx**2 + ny**2

    mask = 1.0 - s * (r2 ** float(exponent))
    mask = np.clip(mask, 0.0, 1.0).astype(np.float32)

    out = arr.astype(np.float32)
    if out.ndim == 3:
        out = out * mask[..., None]
    else:
        out = out * mask

    if arr.dtype == np.uint8:
        return np.clip(out, 0.0, 255.0).astype(np.uint8)
    return out.astype(arr.dtype)


def random_channel_gain(
    image: NDArray,
    gain_range: Tuple[float, float] = (0.9, 1.1),
) -> NDArray:
    """Randomly scale each color channel (simple color temperature/exposure drift proxy)."""

    arr = np.asarray(image)
    if arr.ndim == 2:
        return arr
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected image shape (H,W,3), got {arr.shape}")

    low, high = float(gain_range[0]), float(gain_range[1])
    if low <= 0.0 or high <= 0.0 or high < low:
        raise ValueError(f"Invalid gain_range: {gain_range}")

    gains = np.array(
        [random.uniform(low, high), random.uniform(low, high), random.uniform(low, high)],
        dtype=np.float32,
    )
    out = arr.astype(np.float32) * gains.reshape(1, 1, 3)
    if arr.dtype == np.uint8:
        return np.clip(out, 0.0, 255.0).astype(np.uint8)
    return out.astype(arr.dtype)
