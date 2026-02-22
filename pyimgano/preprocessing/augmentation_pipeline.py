"""
Augmentation pipeline and composition utilities.

Provides flexible augmentation pipelines with:
- Sequential augmentation application
- Random augmentation selection
- Probability-based augmentation
- One-of augmentation (select one from multiple)
- Compose multiple augmentations
"""

import random
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from .augmentation import (
    GeometricTransform,
    NoiseType,
    BlurType,
    WeatherEffect,
    # Geometric
    rotate_image,
    flip_image,
    scale_image,
    translate_image,
    shear_image,
    perspective_transform,
    # Color
    adjust_brightness,
    adjust_contrast,
    adjust_saturation,
    adjust_hue,
    color_jitter,
    # Noise
    add_gaussian_noise,
    add_salt_pepper_noise,
    add_poisson_noise,
    add_speckle_noise,
    # Blur
    motion_blur,
    defocus_blur,
    glass_blur,
    # Weather
    add_rain,
    add_fog,
    add_snow,
    add_shadow,
    # Cutout
    random_cutout,
    grid_mask,
    # Distortion
    elastic_transform,
    grid_distortion,
    # Advanced
    mixup,
    cutmix,
    # Industrial camera robustness
    jpeg_compress,
    vignette,
    random_channel_gain,
)


class AugmentationTransform:
    """
    Base class for augmentation transforms.
    """

    def __init__(self, p: float = 1.0):
        """
        Initialize transform.

        Args:
            p: Probability of applying this transform (0-1)
        """
        self.p = p

    def __call__(self, image: NDArray, **kwargs) -> NDArray:
        """Apply transform with probability p."""
        if random.random() < self.p:
            return self.apply(image, **kwargs)
        return image

    def apply(self, image: NDArray, **kwargs) -> NDArray:
        """Apply the actual transformation."""
        raise NotImplementedError


class Compose:
    """
    Compose multiple augmentation transforms.
    """

    def __init__(self, transforms: List[Union[AugmentationTransform, Callable]]):
        """
        Initialize composition.

        Args:
            transforms: List of transforms to apply sequentially
        """
        self.transforms = transforms

    def __call__(self, image: NDArray, **kwargs) -> NDArray:
        """Apply all transforms sequentially."""
        for transform in self.transforms:
            image = transform(image, **kwargs)
        return image

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string


class OneOf:
    """
    Select one transform from multiple options.
    """

    def __init__(
        self,
        transforms: List[Union[AugmentationTransform, Callable]],
        p: float = 1.0
    ):
        """
        Initialize one-of selection.

        Args:
            transforms: List of transforms to choose from
            p: Probability of applying any transform
        """
        self.transforms = transforms
        self.p = p

    def __call__(self, image: NDArray, **kwargs) -> NDArray:
        """Apply one randomly selected transform."""
        if random.random() < self.p:
            transform = random.choice(self.transforms)
            return transform(image, **kwargs)
        return image


class RandomApply:
    """
    Apply a transform with specified probability.
    """

    def __init__(self, transform: Union[AugmentationTransform, Callable], p: float = 0.5):
        """
        Initialize random apply.

        Args:
            transform: Transform to apply
            p: Probability of applying transform
        """
        self.transform = transform
        self.p = p

    def __call__(self, image: NDArray, **kwargs) -> NDArray:
        """Apply transform with probability p."""
        if random.random() < self.p:
            return self.transform(image, **kwargs)
        return image


# Concrete Transform Classes

class RandomRotate(AugmentationTransform):
    """Random rotation augmentation."""

    def __init__(self, angle_range: Tuple[float, float] = (-30, 30), p: float = 0.5):
        super().__init__(p)
        self.angle_range = angle_range

    def apply(self, image: NDArray, **kwargs) -> NDArray:
        angle = random.uniform(*self.angle_range)
        return rotate_image(image, angle)


class RandomFlip(AugmentationTransform):
    """Random flip augmentation."""

    def __init__(self, mode: str = "horizontal", p: float = 0.5):
        super().__init__(p)
        self.mode = mode

    def apply(self, image: NDArray, **kwargs) -> NDArray:
        return flip_image(image, self.mode)


class RandomScale(AugmentationTransform):
    """Random scale augmentation."""

    def __init__(
        self,
        scale_range: Tuple[float, float] = (0.8, 1.2),
        keep_size: bool = True,
        p: float = 0.5
    ):
        super().__init__(p)
        self.scale_range = scale_range
        self.keep_size = keep_size

    def apply(self, image: NDArray, **kwargs) -> NDArray:
        scale = random.uniform(*self.scale_range)
        return scale_image(image, scale, scale, self.keep_size)


class RandomTranslate(AugmentationTransform):
    """Random translation augmentation."""

    def __init__(
        self,
        translate_range: Tuple[float, float] = (-0.1, 0.1),
        p: float = 0.5
    ):
        super().__init__(p)
        self.translate_range = translate_range

    def apply(self, image: NDArray, **kwargs) -> NDArray:
        h, w = image.shape[:2]
        tx = int(random.uniform(*self.translate_range) * w)
        ty = int(random.uniform(*self.translate_range) * h)
        return translate_image(image, tx, ty)


class RandomShear(AugmentationTransform):
    """Random shear augmentation."""

    def __init__(
        self,
        shear_range: Tuple[float, float] = (-0.2, 0.2),
        p: float = 0.5
    ):
        super().__init__(p)
        self.shear_range = shear_range

    def apply(self, image: NDArray, **kwargs) -> NDArray:
        shear_x = random.uniform(*self.shear_range)
        shear_y = random.uniform(*self.shear_range)
        return shear_image(image, shear_x, shear_y)


class RandomPerspective(AugmentationTransform):
    """Random perspective augmentation."""

    def __init__(self, strength: float = 0.2, p: float = 0.5):
        super().__init__(p)
        self.strength = strength

    def apply(self, image: NDArray, **kwargs) -> NDArray:
        return perspective_transform(image, self.strength)


class ColorJitter(AugmentationTransform):
    """Color jitter augmentation."""

    def __init__(
        self,
        brightness: Tuple[float, float] = (0.8, 1.2),
        contrast: Tuple[float, float] = (0.8, 1.2),
        saturation: Tuple[float, float] = (0.8, 1.2),
        hue: Tuple[float, float] = (-10, 10),
        p: float = 0.5
    ):
        super().__init__(p)
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def apply(self, image: NDArray, **kwargs) -> NDArray:
        return color_jitter(image, self.brightness, self.contrast, self.saturation, self.hue)


class GaussianNoise(AugmentationTransform):
    """Gaussian noise augmentation."""

    def __init__(self, std_range: Tuple[float, float] = (10, 30), p: float = 0.5):
        super().__init__(p)
        self.std_range = std_range

    def apply(self, image: NDArray, **kwargs) -> NDArray:
        std = random.uniform(*self.std_range)
        return add_gaussian_noise(image, 0, std)


class SaltPepperNoise(AugmentationTransform):
    """Salt-and-pepper noise augmentation."""

    def __init__(
        self,
        salt_prob: float = 0.01,
        pepper_prob: float = 0.01,
        p: float = 0.5
    ):
        super().__init__(p)
        self.salt_prob = salt_prob
        self.pepper_prob = pepper_prob

    def apply(self, image: NDArray, **kwargs) -> NDArray:
        return add_salt_pepper_noise(image, self.salt_prob, self.pepper_prob)


class MotionBlur(AugmentationTransform):
    """Motion blur augmentation."""

    def __init__(
        self,
        kernel_size_range: Tuple[int, int] = (3, 15),
        angle_range: Tuple[float, float] = (-45, 45),
        p: float = 0.5
    ):
        super().__init__(p)
        self.kernel_size_range = kernel_size_range
        self.angle_range = angle_range

    def apply(self, image: NDArray, **kwargs) -> NDArray:
        kernel_size = random.randint(*self.kernel_size_range)
        if kernel_size % 2 == 0:
            kernel_size += 1
        angle = random.uniform(*self.angle_range)
        return motion_blur(image, kernel_size, angle)


class DefocusBlur(AugmentationTransform):
    """Defocus blur augmentation."""

    def __init__(self, radius_range: Tuple[int, int] = (3, 7), p: float = 0.5):
        super().__init__(p)
        self.radius_range = radius_range

    def apply(self, image: NDArray, **kwargs) -> NDArray:
        radius = random.randint(*self.radius_range)
        return defocus_blur(image, radius)


class RandomRain(AugmentationTransform):
    """Random rain effect augmentation."""

    def __init__(
        self,
        intensity_range: Tuple[float, float] = (0.3, 0.7),
        p: float = 0.5
    ):
        super().__init__(p)
        self.intensity_range = intensity_range

    def apply(self, image: NDArray, **kwargs) -> NDArray:
        intensity = random.uniform(*self.intensity_range)
        return add_rain(image, intensity)


class RandomFog(AugmentationTransform):
    """Random fog effect augmentation."""

    def __init__(
        self,
        intensity_range: Tuple[float, float] = (0.3, 0.7),
        p: float = 0.5
    ):
        super().__init__(p)
        self.intensity_range = intensity_range

    def apply(self, image: NDArray, **kwargs) -> NDArray:
        intensity = random.uniform(*self.intensity_range)
        return add_fog(image, intensity)


class RandomSnow(AugmentationTransform):
    """Random snow effect augmentation."""

    def __init__(
        self,
        intensity_range: Tuple[float, float] = (0.3, 0.7),
        p: float = 0.5
    ):
        super().__init__(p)
        self.intensity_range = intensity_range

    def apply(self, image: NDArray, **kwargs) -> NDArray:
        intensity = random.uniform(*self.intensity_range)
        return add_snow(image, intensity)


class RandomShadow(AugmentationTransform):
    """Random shadow augmentation."""

    def __init__(
        self,
        num_shadows: int = 1,
        intensity_range: Tuple[float, float] = (0.3, 0.7),
        p: float = 0.5
    ):
        super().__init__(p)
        self.num_shadows = num_shadows
        self.intensity_range = intensity_range

    def apply(self, image: NDArray, **kwargs) -> NDArray:
        intensity = random.uniform(*self.intensity_range)
        return add_shadow(image, self.num_shadows, intensity)


class RandomCutout(AugmentationTransform):
    """Random cutout augmentation."""

    def __init__(
        self,
        num_holes: int = 1,
        hole_size: Union[int, Tuple[int, int]] = 32,
        fill_value: int = 0,
        p: float = 0.5
    ):
        super().__init__(p)
        self.num_holes = num_holes
        self.hole_size = hole_size
        self.fill_value = fill_value

    def apply(self, image: NDArray, **kwargs) -> NDArray:
        return random_cutout(image, self.num_holes, self.hole_size, self.fill_value)


class GridMask(AugmentationTransform):
    """Grid mask augmentation."""

    def __init__(
        self,
        grid_size: int = 32,
        ratio: float = 0.5,
        fill_value: int = 0,
        p: float = 0.5
    ):
        super().__init__(p)
        self.grid_size = grid_size
        self.ratio = ratio
        self.fill_value = fill_value

    def apply(self, image: NDArray, **kwargs) -> NDArray:
        return grid_mask(image, self.grid_size, self.ratio, self.fill_value)


class ElasticTransform(AugmentationTransform):
    """Elastic transform augmentation."""

    def __init__(self, alpha: float = 100, sigma: float = 10, p: float = 0.5):
        super().__init__(p)
        self.alpha = alpha
        self.sigma = sigma

    def apply(self, image: NDArray, **kwargs) -> NDArray:
        return elastic_transform(image, self.alpha, self.sigma)


# Preset Augmentation Pipelines

def get_light_augmentation() -> Compose:
    """
    Get light augmentation pipeline for training.

    Returns:
        Compose object with light augmentations
    """
    return Compose([
        RandomFlip(mode="horizontal", p=0.5),
        RandomRotate(angle_range=(-10, 10), p=0.3),
        ColorJitter(
            brightness=(0.9, 1.1),
            contrast=(0.9, 1.1),
            saturation=(0.9, 1.1),
            hue=(-5, 5),
            p=0.3
        ),
    ])


def get_medium_augmentation() -> Compose:
    """
    Get medium augmentation pipeline for training.

    Returns:
        Compose object with medium augmentations
    """
    return Compose([
        RandomFlip(mode="horizontal", p=0.5),
        RandomRotate(angle_range=(-20, 20), p=0.5),
        RandomScale(scale_range=(0.9, 1.1), p=0.3),
        ColorJitter(
            brightness=(0.8, 1.2),
            contrast=(0.8, 1.2),
            saturation=(0.8, 1.2),
            hue=(-10, 10),
            p=0.5
        ),
        OneOf([
            GaussianNoise(std_range=(10, 25), p=1.0),
            SaltPepperNoise(salt_prob=0.01, pepper_prob=0.01, p=1.0),
        ], p=0.3),
    ])


def get_heavy_augmentation() -> Compose:
    """
    Get heavy augmentation pipeline for training.

    Returns:
        Compose object with heavy augmentations
    """
    return Compose([
        RandomFlip(mode="horizontal", p=0.5),
        RandomRotate(angle_range=(-30, 30), p=0.5),
        RandomScale(scale_range=(0.8, 1.2), p=0.5),
        RandomShear(shear_range=(-0.2, 0.2), p=0.3),
        OneOf([
            RandomPerspective(strength=0.2, p=1.0),
            ElasticTransform(alpha=100, sigma=10, p=1.0),
        ], p=0.3),
        ColorJitter(
            brightness=(0.7, 1.3),
            contrast=(0.7, 1.3),
            saturation=(0.7, 1.3),
            hue=(-15, 15),
            p=0.5
        ),
        OneOf([
            GaussianNoise(std_range=(15, 35), p=1.0),
            SaltPepperNoise(salt_prob=0.02, pepper_prob=0.02, p=1.0),
        ], p=0.5),
        OneOf([
            MotionBlur(kernel_size_range=(5, 15), p=1.0),
            DefocusBlur(radius_range=(3, 7), p=1.0),
        ], p=0.3),
        RandomCutout(num_holes=1, hole_size=32, p=0.3),
    ])


def get_weather_augmentation() -> Compose:
    """
    Get weather effects augmentation pipeline.

    Returns:
        Compose object with weather augmentations
    """
    return Compose([
        OneOf([
            RandomRain(intensity_range=(0.3, 0.6), p=1.0),
            RandomFog(intensity_range=(0.2, 0.5), p=1.0),
            RandomSnow(intensity_range=(0.3, 0.6), p=1.0),
            RandomShadow(num_shadows=1, intensity_range=(0.3, 0.6), p=1.0),
        ], p=0.5),
    ])


def get_anomaly_augmentation() -> Compose:
    """
    Get augmentation pipeline specific for anomaly detection.

    Focuses on augmentations that preserve anomalies while
    improving model robustness.

    Returns:
        Compose object with anomaly-preserving augmentations
    """
    return Compose([
        RandomFlip(mode="horizontal", p=0.5),
        RandomRotate(angle_range=(-15, 15), p=0.5),
        ColorJitter(
            brightness=(0.85, 1.15),
            contrast=(0.85, 1.15),
            saturation=(0.85, 1.15),
            hue=(-8, 8),
            p=0.5
        ),
        OneOf([
            GaussianNoise(std_range=(5, 15), p=1.0),
            SaltPepperNoise(salt_prob=0.005, pepper_prob=0.005, p=1.0),
        ], p=0.3),
        # Avoid heavy distortions that might hide anomalies
    ])


def get_industrial_camera_robust_augmentation() -> Compose:
    """Augmentation preset focused on industrial camera/lighting drift robustness.

    This targets common production artifacts:
    - JPEG recompression (edge ringing / blocking)
    - vignetting (lens / lighting falloff)
    - mild per-channel gain drift (white balance / illumination changes)
    """

    def _jpeg(image):
        return jpeg_compress(image, quality=random.randint(30, 95))

    def _vignette(image):
        return vignette(image, strength=random.uniform(0.1, 0.5), exponent=2.0)

    def _gain(image):
        return random_channel_gain(image, gain_range=(0.85, 1.15))

    return Compose(
        [
            RandomFlip(mode="horizontal", p=0.5),
            RandomRotate(angle_range=(-10, 10), p=0.3),
            ColorJitter(
                brightness=(0.85, 1.15),
                contrast=(0.85, 1.15),
                saturation=(0.9, 1.1),
                hue=(-6, 6),
                p=0.4,
            ),
            RandomApply(_gain, p=0.35),
            RandomApply(_jpeg, p=0.25),
            RandomApply(_vignette, p=0.25),
            OneOf(
                [
                    GaussianNoise(std_range=(5, 20), p=1.0),
                    SaltPepperNoise(salt_prob=0.01, pepper_prob=0.01, p=1.0),
                ],
                p=0.25,
            ),
            OneOf(
                [
                    MotionBlur(kernel_size_range=(3, 9), p=1.0),
                    DefocusBlur(radius_range=(2, 6), p=1.0),
                ],
                p=0.2,
            ),
        ]
    )


class AugmentationPipeline:
    """
    Flexible augmentation pipeline with statistics tracking.
    """

    def __init__(self, transforms: List[Union[AugmentationTransform, Callable]]):
        """
        Initialize pipeline.

        Args:
            transforms: List of transforms
        """
        self.transforms = transforms
        self.stats = {
            'total_images': 0,
            'transform_applications': {}
        }

    def __call__(self, image: NDArray, **kwargs) -> NDArray:
        """Apply pipeline and track statistics."""
        self.stats['total_images'] += 1

        for transform in self.transforms:
            transform_name = transform.__class__.__name__
            if transform_name not in self.stats['transform_applications']:
                self.stats['transform_applications'][transform_name] = 0

            # Apply transform
            original_image = image.copy()
            image = transform(image, **kwargs)

            # Check if transform was applied (image changed)
            if not np.array_equal(image, original_image):
                self.stats['transform_applications'][transform_name] += 1

        return image

    def get_stats(self) -> dict:
        """Get pipeline statistics."""
        return self.stats.copy()

    def reset_stats(self):
        """Reset statistics."""
        self.stats = {
            'total_images': 0,
            'transform_applications': {}
        }
