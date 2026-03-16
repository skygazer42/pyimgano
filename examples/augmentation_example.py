"""
Image augmentation examples.

This example demonstrates comprehensive augmentation techniques including:
1. Geometric transformations
2. Color augmentations
3. Noise addition
4. Blur effects
5. Weather effects
6. Cutout and occlusion
7. Elastic and grid distortions
8. Advanced augmentations (Mixup, CutMix)
9. Augmentation pipelines
10. Preset augmentation strategies
"""

import os
import sys

import cv2
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pyimgano.preprocessing import (  # Pipeline classes; Transform classes; Preset pipelines
    AugmentationPipeline,
    ColorJitter,
    Compose,
    DefocusBlur,
    ElasticTransform,
    GaussianNoise,
    GridMask,
    MotionBlur,
    OneOf,
    RandomCutout,
    RandomFlip,
    RandomFog,
    RandomPerspective,
    RandomRain,
    RandomRotate,
    RandomScale,
    RandomShadow,
    RandomShear,
    RandomSnow,
    RandomTranslate,
    SaltPepperNoise,
    get_anomaly_augmentation,
    get_heavy_augmentation,
    get_light_augmentation,
    get_medium_augmentation,
    get_weather_augmentation,
)


def create_test_image():
    """Create a test image with various features."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)

    # Add colorful shapes
    cv2.rectangle(img, (50, 50), (150, 150), (255, 100, 100), -1)
    cv2.circle(img, (100, 100), 30, (100, 255, 100), -1)
    cv2.line(img, (0, 0), (200, 200), (100, 100, 255), 3)

    # Add text
    cv2.putText(img, "TEST", (60, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return img


def example_geometric_transforms():
    """Example 1: Geometric transformations."""
    print("\n" + "=" * 60)
    print("Example 1: Geometric Transformations")
    print("=" * 60)

    img = create_test_image()

    print("\n1. Random Rotation:")
    rotate = RandomRotate(angle_range=(-30, 30), p=1.0)
    rotated = rotate(img)
    print(f"   Applied rotation, result shape: {rotated.shape}")

    print("\n2. Random Flip:")
    flip = RandomFlip(mode="horizontal", p=1.0)
    flipped = flip(img)
    print(f"   Applied horizontal flip, result shape: {flipped.shape}")

    print("\n3. Random Scale:")
    scale = RandomScale(scale_range=(0.8, 1.2), p=1.0)
    scaled = scale(img)
    print(f"   Applied scaling, result shape: {scaled.shape}")

    print("\n4. Random Translation:")
    translate = RandomTranslate(translate_range=(-0.1, 0.1), p=1.0)
    translated = translate(img)
    print(f"   Applied translation, result shape: {translated.shape}")

    print("\n5. Random Shear:")
    shear = RandomShear(shear_range=(-0.2, 0.2), p=1.0)
    sheared = shear(img)
    print(f"   Applied shear, result shape: {sheared.shape}")

    print("\n6. Random Perspective:")
    perspective = RandomPerspective(strength=0.3, p=1.0)
    persp = perspective(img)
    print(f"   Applied perspective transform, result shape: {persp.shape}")


def example_color_augmentations():
    """Example 2: Color augmentations."""
    print("\n" + "=" * 60)
    print("Example 2: Color Augmentations")
    print("=" * 60)

    img = create_test_image()

    print("\n1. Color Jitter (random brightness, contrast, saturation, hue):")
    color_jitter = ColorJitter(
        brightness=(0.5, 1.5), contrast=(0.5, 1.5), saturation=(0.5, 1.5), hue=(-20, 20), p=1.0
    )
    jittered = color_jitter(img)
    print(f"   Applied color jitter, result shape: {jittered.shape}")
    print(f"   Mean pixel value change: {np.abs(img.mean() - jittered.mean()):.2f}")


def example_noise_augmentations():
    """Example 3: Noise augmentations."""
    print("\n" + "=" * 60)
    print("Example 3: Noise Augmentations")
    print("=" * 60)

    img = create_test_image()

    print("\n1. Gaussian Noise:")
    gaussian_noise = GaussianNoise(std_range=(10, 30), p=1.0)
    noisy_gaussian = gaussian_noise(img)
    print(f"   Applied Gaussian noise, result shape: {noisy_gaussian.shape}")

    print("\n2. Salt-and-Pepper Noise:")
    salt_pepper = SaltPepperNoise(salt_prob=0.02, pepper_prob=0.02, p=1.0)
    noisy_sp = salt_pepper(img)
    print(f"   Applied salt-and-pepper noise, result shape: {noisy_sp.shape}")


def example_blur_augmentations():
    """Example 4: Blur augmentations."""
    print("\n" + "=" * 60)
    print("Example 4: Blur Augmentations")
    print("=" * 60)

    img = create_test_image()

    print("\n1. Motion Blur:")
    motion_blur = MotionBlur(kernel_size_range=(5, 15), angle_range=(-45, 45), p=1.0)
    blurred_motion = motion_blur(img)
    print(f"   Applied motion blur, result shape: {blurred_motion.shape}")

    print("\n2. Defocus Blur:")
    defocus_blur = DefocusBlur(radius_range=(3, 7), p=1.0)
    blurred_defocus = defocus_blur(img)
    print(f"   Applied defocus blur, result shape: {blurred_defocus.shape}")


def example_weather_effects():
    """Example 5: Weather effects."""
    print("\n" + "=" * 60)
    print("Example 5: Weather Effects")
    print("=" * 60)

    img = create_test_image()

    print("\n1. Random Rain:")
    rain = RandomRain(intensity_range=(0.4, 0.6), p=1.0)
    rainy = rain(img)
    print(f"   Applied rain effect, result shape: {rainy.shape}")

    print("\n2. Random Fog:")
    fog = RandomFog(intensity_range=(0.3, 0.5), p=1.0)
    foggy = fog(img)
    print(f"   Applied fog effect, result shape: {foggy.shape}")

    print("\n3. Random Snow:")
    snow = RandomSnow(intensity_range=(0.4, 0.6), p=1.0)
    snowy = snow(img)
    print(f"   Applied snow effect, result shape: {snowy.shape}")

    print("\n4. Random Shadow:")
    shadow = RandomShadow(num_shadows=2, intensity_range=(0.4, 0.6), p=1.0)
    shadowed = shadow(img)
    print(f"   Applied shadow effect, result shape: {shadowed.shape}")


def example_cutout_augmentations():
    """Example 6: Cutout and occlusion."""
    print("\n" + "=" * 60)
    print("Example 6: Cutout and Occlusion")
    print("=" * 60)

    img = create_test_image()

    print("\n1. Random Cutout:")
    cutout = RandomCutout(num_holes=2, hole_size=40, fill_value=0, p=1.0)
    cutout_img = cutout(img)
    print(f"   Applied cutout (2 holes), result shape: {cutout_img.shape}")

    print("\n2. Grid Mask:")
    grid_mask = GridMask(grid_size=32, ratio=0.5, fill_value=0, p=1.0)
    grid_masked = grid_mask(img)
    print(f"   Applied grid mask, result shape: {grid_masked.shape}")


def example_elastic_transforms():
    """Example 7: Elastic and grid distortions."""
    print("\n" + "=" * 60)
    print("Example 7: Elastic Transforms")
    print("=" * 60)

    img = create_test_image()

    print("\n1. Elastic Transform:")
    elastic = ElasticTransform(alpha=100, sigma=10, p=1.0)
    elastic_img = elastic(img)
    print(f"   Applied elastic transform, result shape: {elastic_img.shape}")


def example_compose_pipeline():
    """Example 8: Compose multiple augmentations."""
    print("\n" + "=" * 60)
    print("Example 8: Compose Multiple Augmentations")
    print("=" * 60)

    img = create_test_image()

    print("\nCreating pipeline with 5 augmentations:")
    pipeline = Compose(
        [
            RandomRotate(angle_range=(-15, 15), p=0.7),
            RandomFlip(mode="horizontal", p=0.5),
            ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2), p=0.6),
            GaussianNoise(std_range=(5, 15), p=0.4),
            RandomCutout(num_holes=1, hole_size=32, p=0.3),
        ]
    )

    print("\nApplying pipeline 5 times:")
    for i in range(5):
        augmented = pipeline(img)
        print(f"   Iteration {i+1}: Result shape {augmented.shape}")


def example_one_of():
    """Example 9: OneOf augmentation (select one from multiple)."""
    print("\n" + "=" * 60)
    print("Example 9: OneOf Augmentation")
    print("=" * 60)

    img = create_test_image()

    print("\nOneOf: Apply one of three weather effects:")
    one_of_weather = OneOf(
        [
            RandomRain(intensity_range=(0.4, 0.6), p=1.0),
            RandomFog(intensity_range=(0.3, 0.5), p=1.0),
            RandomSnow(intensity_range=(0.4, 0.6), p=1.0),
        ],
        p=0.8,
    )

    print("\nApplying OneOf 10 times:")
    for i in range(10):
        _ = one_of_weather(img)
        # Check which effect was applied (simplified)
        print(f"   Iteration {i+1}: Applied random weather effect")


def example_preset_pipelines():
    """Example 10: Preset augmentation pipelines."""
    print("\n" + "=" * 60)
    print("Example 10: Preset Augmentation Pipelines")
    print("=" * 60)

    img = create_test_image()

    print("\n1. Light Augmentation Pipeline:")
    light_aug = get_light_augmentation()
    light_result = light_aug(img)
    print(f"   Result shape: {light_result.shape}")
    print("   Includes: flip, slight rotation, mild color jitter")

    print("\n2. Medium Augmentation Pipeline:")
    medium_aug = get_medium_augmentation()
    medium_result = medium_aug(img)
    print(f"   Result shape: {medium_result.shape}")
    print("   Includes: flip, rotation, scale, color jitter, noise")

    print("\n3. Heavy Augmentation Pipeline:")
    heavy_aug = get_heavy_augmentation()
    heavy_result = heavy_aug(img)
    print(f"   Result shape: {heavy_result.shape}")
    print("   Includes: all geometric, color, noise, blur, cutout")

    print("\n4. Weather Augmentation Pipeline:")
    weather_aug = get_weather_augmentation()
    weather_result = weather_aug(img)
    print(f"   Result shape: {weather_result.shape}")
    print("   Includes: rain, fog, snow, shadow effects")

    print("\n5. Anomaly Detection Augmentation:")
    anomaly_aug = get_anomaly_augmentation()
    anomaly_result = anomaly_aug(img)
    print(f"   Result shape: {anomaly_result.shape}")
    print("   Includes: anomaly-preserving augmentations")


def example_custom_pipeline_for_training():
    """Example 11: Custom training pipeline."""
    print("\n" + "=" * 60)
    print("Example 11: Custom Training Pipeline")
    print("=" * 60)

    img = create_test_image()

    print("\nBuilding custom augmentation pipeline for anomaly detection training:")

    # Custom pipeline
    train_pipeline = Compose(
        [
            # Geometric augmentations
            RandomFlip(mode="horizontal", p=0.5),
            RandomFlip(mode="vertical", p=0.2),
            RandomRotate(angle_range=(-20, 20), p=0.5),
            RandomScale(scale_range=(0.9, 1.1), p=0.3),
            # Color augmentations
            ColorJitter(
                brightness=(0.85, 1.15),
                contrast=(0.85, 1.15),
                saturation=(0.85, 1.15),
                hue=(-10, 10),
                p=0.5,
            ),
            # Noise (one of)
            OneOf(
                [
                    GaussianNoise(std_range=(5, 15), p=1.0),
                    SaltPepperNoise(salt_prob=0.01, pepper_prob=0.01, p=1.0),
                ],
                p=0.3,
            ),
            # Blur (one of)
            OneOf(
                [
                    MotionBlur(kernel_size_range=(3, 9), p=1.0),
                    DefocusBlur(radius_range=(2, 4), p=1.0),
                ],
                p=0.2,
            ),
            # Cutout
            RandomCutout(num_holes=1, hole_size=24, p=0.2),
        ]
    )

    print("\nApplying custom pipeline 5 times:")
    for i in range(5):
        augmented = train_pipeline(img)
        print(f"   Iteration {i+1}: Augmentation applied, shape {augmented.shape}")


def example_augmentation_statistics():
    """Example 12: Track augmentation statistics."""
    print("\n" + "=" * 60)
    print("Example 12: Augmentation Statistics")
    print("=" * 60)

    img = create_test_image()

    print("\nCreating augmentation pipeline with statistics tracking:")

    transforms = [
        RandomRotate(angle_range=(-15, 15), p=0.5),
        RandomFlip(mode="horizontal", p=0.5),
        ColorJitter(brightness=(0.8, 1.2), p=0.5),
        GaussianNoise(std_range=(10, 20), p=0.5),
    ]

    pipeline = AugmentationPipeline(transforms)

    print("\nApplying pipeline to 100 images:")
    for _ in range(100):
        _ = pipeline(img)

    print("\nAugmentation Statistics:")
    stats = pipeline.get_stats()
    print(f"   Total images processed: {stats['total_images']}")
    print("\n   Transform application counts:")
    for transform_name, count in stats["transform_applications"].items():
        percentage = (count / stats["total_images"]) * 100
        print(f"      {transform_name}: {count} times ({percentage:.1f}%)")


def example_batch_augmentation():
    """Example 13: Batch augmentation."""
    print("\n" + "=" * 60)
    print("Example 13: Batch Augmentation")
    print("=" * 60)

    # Create batch of images
    batch_size = 10
    images = [create_test_image() for _ in range(batch_size)]

    print(f"\nCreated batch of {batch_size} images")

    # Augmentation pipeline
    aug_pipeline = get_medium_augmentation()

    print("\nAugmenting batch:")
    augmented_batch = []
    for i, img in enumerate(images):
        augmented = aug_pipeline(img)
        augmented_batch.append(augmented)
        print(f"   Image {i+1}/{batch_size}: Augmented, shape {augmented.shape}")

    print(f"\nAugmented batch size: {len(augmented_batch)}")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("PyImgAno Image Augmentation Examples")
    print("=" * 60)

    try:
        example_geometric_transforms()
    except Exception as e:
        print(f"\nError in geometric transforms: {e}")

    try:
        example_color_augmentations()
    except Exception as e:
        print(f"\nError in color augmentations: {e}")

    try:
        example_noise_augmentations()
    except Exception as e:
        print(f"\nError in noise augmentations: {e}")

    try:
        example_blur_augmentations()
    except Exception as e:
        print(f"\nError in blur augmentations: {e}")

    try:
        example_weather_effects()
    except Exception as e:
        print(f"\nError in weather effects: {e}")

    try:
        example_cutout_augmentations()
    except Exception as e:
        print(f"\nError in cutout augmentations: {e}")

    try:
        example_elastic_transforms()
    except Exception as e:
        print(f"\nError in elastic transforms: {e}")

    try:
        example_compose_pipeline()
    except Exception as e:
        print(f"\nError in compose pipeline: {e}")

    try:
        example_one_of()
    except Exception as e:
        print(f"\nError in one-of: {e}")

    try:
        example_preset_pipelines()
    except Exception as e:
        print(f"\nError in preset pipelines: {e}")

    try:
        example_custom_pipeline_for_training()
    except Exception as e:
        print(f"\nError in custom pipeline: {e}")

    try:
        example_augmentation_statistics()
    except Exception as e:
        print(f"\nError in statistics: {e}")

    try:
        example_batch_augmentation()
    except Exception as e:
        print(f"\nError in batch augmentation: {e}")

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
