"""
Example usage of preprocessing module with anomaly detectors.

This example demonstrates:
1. Basic preprocessing operations
2. Preprocessing pipelines
3. Integration with anomaly detectors using PreprocessingMixin
4. Different preprocessing strategies for different scenarios
"""

import os
import sys

import cv2
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pyimgano.preprocessing import (
    ImageEnhancer,
    PreprocessingPipeline,
    PreprocessingMixin,
)
from pyimgano.models import ECOD


def create_sample_images():
    """Create sample images for demonstration."""
    images = []

    for i in range(10):
        # Create varied sample images
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        # Add some structure
        if i % 2 == 0:
            cv2.rectangle(img, (20, 20), (80, 80), (255, 255, 255), -1)
        else:
            cv2.circle(img, (50, 50), 30, (255, 255, 255), -1)

        # Add some anomalies to later images
        if i >= 8:
            cv2.rectangle(img, (10, 10), (30, 30), (0, 0, 255), -1)

        images.append(img)

    return images


def example_basic_operations():
    """Example 1: Basic preprocessing operations."""
    print("\n" + "="*60)
    print("Example 1: Basic Preprocessing Operations")
    print("="*60)

    # Create enhancer
    enhancer = ImageEnhancer()

    # Create sample image
    img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    cv2.rectangle(img, (30, 30), (70, 70), (255, 255, 255), -1)

    print("\nOriginal image shape:", img.shape)

    # Apply various operations
    print("\n1. Gaussian Blur:")
    blurred = enhancer.gaussian_blur(img, ksize=(5, 5))
    print(f"   Result shape: {blurred.shape}")

    print("\n2. Edge Detection (Canny):")
    edges = enhancer.detect_edges(img, method='canny')
    print(f"   Result shape: {edges.shape}")

    print("\n3. Morphological Erosion:")
    eroded = enhancer.erode(img, kernel_size=(5, 5))
    print(f"   Result shape: {eroded.shape}")

    print("\n4. Normalization:")
    normalized = enhancer.normalize(img, method='minmax')
    print(f"   Result shape: {normalized.shape}")
    print(f"   Value range: [{normalized.min():.3f}, {normalized.max():.3f}]")

    print("\n5. Sharpening:")
    sharpened = enhancer.sharpen(img)
    print(f"   Result shape: {sharpened.shape}")

    print("\n6. CLAHE (Contrast Limited Adaptive Histogram Equalization):")
    clahe_result = enhancer.clahe(img)
    print(f"   Result shape: {clahe_result.shape}")


def example_preprocessing_pipeline():
    """Example 2: Using preprocessing pipelines."""
    print("\n" + "="*60)
    print("Example 2: Preprocessing Pipeline")
    print("="*60)

    # Create sample image
    img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    cv2.circle(img, (50, 50), 30, (255, 255, 255), -1)

    # Build pipeline
    print("\nBuilding preprocessing pipeline:")
    pipeline = PreprocessingPipeline()

    # Add steps
    print("  1. Gaussian blur (reduce noise)")
    pipeline.add_step('gaussian_blur', ksize=(5, 5))

    print("  2. Edge detection (Canny)")
    pipeline.add_step('detect_edges', method='canny')

    print("  3. Normalization (MinMax)")
    pipeline.add_step('normalize', method='minmax')

    print(f"\nPipeline: {pipeline}")

    # Apply pipeline
    result = pipeline.transform(img)

    print(f"\nOriginal image shape: {img.shape}")
    print(f"Result shape: {result.shape}")
    print(f"Result range: [{result.min():.3f}, {result.max():.3f}]")


def example_detector_with_preprocessing():
    """Example 3: Detector with preprocessing using mixin."""
    print("\n" + "="*60)
    print("Example 3: Detector with Preprocessing Mixin")
    print("="*60)

    # Create custom detector with preprocessing
    class ECODWithPreprocessing(PreprocessingMixin, ECOD):
        """ECOD detector with preprocessing capabilities."""

        def __init__(self, **kwargs):
            # Initialize ECOD
            super().__init__(**kwargs)

            # Setup preprocessing
            self.setup_preprocessing(enable=True, use_pipeline=True)

            # Configure preprocessing pipeline
            self.add_preprocessing_step('gaussian_blur', ksize=(5, 5))
            self.add_preprocessing_step('normalize', method='minmax')

        def fit(self, X, y=None):
            """Fit with preprocessing."""
            print(f"Preprocessing {len(X)} training images...")
            X_processed = self.preprocess_images(X)

            # Flatten for ECOD
            X_flat = [img.flatten() for img in X_processed]
            return super().fit(X_flat, y)

        def predict(self, X):
            """Predict with preprocessing."""
            print(f"Preprocessing {len(X)} test images...")
            X_processed = self.preprocess_images(X)

            # Flatten for ECOD
            X_flat = [img.flatten() for img in X_processed]
            return super().predict(X_flat)

        def decision_function(self, X):
            """Compute anomaly scores with preprocessing."""
            print(f"Preprocessing {len(X)} test images...")
            X_processed = self.preprocess_images(X)

            # Flatten for ECOD
            X_flat = [img.flatten() for img in X_processed]
            return super().decision_function(X_flat)

    # Create sample data
    train_images = create_sample_images()[:8]
    test_images = create_sample_images()[8:]

    print("\nCreating detector with preprocessing...")
    detector = ECODWithPreprocessing()

    # Get preprocessing info
    info = detector.get_preprocessing_info()
    print(f"\nPreprocessing configuration:")
    print(f"  Enabled: {info['enabled']}")
    print(f"  Pipeline mode: {info['pipeline_mode']}")
    print(f"  Pipeline steps: {info['pipeline_steps']}")
    print(f"  Steps: {info['steps']}")

    # Train
    print("\nTraining detector...")
    detector.fit(train_images)

    # Predict
    print("\nPredicting anomaly scores...")
    scores = detector.decision_function(test_images)
    predictions = detector.predict(test_images)

    print(f"\nAnomaly scores: {scores}")
    print(f"Mean score: {scores.mean():.4f}")
    print(f"Predicted anomalies: {int(predictions.sum())}/{len(predictions)}")


def example_different_strategies():
    """Example 4: Different preprocessing strategies."""
    print("\n" + "="*60)
    print("Example 4: Different Preprocessing Strategies")
    print("="*60)

    img = create_sample_images()[0]

    # Strategy 1: Texture enhancement
    print("\nStrategy 1: Texture Enhancement")
    print("  Use case: Detecting surface defects")
    pipeline_texture = PreprocessingPipeline()
    pipeline_texture.add_step('gaussian_blur', ksize=(3, 3))
    pipeline_texture.add_step('unsharp_mask', sigma=1.0, amount=1.5)
    pipeline_texture.add_step('normalize', method='minmax')

    result1 = pipeline_texture.transform(img)
    print(f"  Result shape: {result1.shape}")

    # Strategy 2: Edge-based
    print("\nStrategy 2: Edge-Based Detection")
    print("  Use case: Detecting structural anomalies")
    pipeline_edge = PreprocessingPipeline()
    pipeline_edge.add_step('gaussian_blur', ksize=(5, 5))
    pipeline_edge.add_step('detect_edges', method='canny')
    pipeline_edge.add_step('dilate', kernel_size=(3, 3))
    pipeline_edge.add_step('normalize', method='minmax')

    result2 = pipeline_edge.transform(img)
    print(f"  Result shape: {result2.shape}")

    # Strategy 3: Morphology-based
    print("\nStrategy 3: Morphology-Based")
    print("  Use case: Detecting shape anomalies")
    pipeline_morph = PreprocessingPipeline()
    pipeline_morph.add_step('gaussian_blur', ksize=(5, 5))
    pipeline_morph.add_step('opening', kernel_size=(5, 5))
    pipeline_morph.add_step('closing', kernel_size=(5, 5))
    pipeline_morph.add_step('morphological_gradient', kernel_size=(3, 3))
    pipeline_morph.add_step('normalize', method='minmax')

    result3 = pipeline_morph.transform(img)
    print(f"  Result shape: {result3.shape}")

    # Strategy 4: Contrast enhancement
    print("\nStrategy 4: Contrast Enhancement")
    print("  Use case: Low-contrast images")
    pipeline_contrast = PreprocessingPipeline()
    pipeline_contrast.add_step('clahe', clip_limit=2.0)
    pipeline_contrast.add_step('normalize', method='robust')

    result4 = pipeline_contrast.transform(img)
    print(f"  Result shape: {result4.shape}")


def example_edge_detection_comparison():
    """Example 5: Comparing edge detection methods."""
    print("\n" + "="*60)
    print("Example 5: Edge Detection Method Comparison")
    print("="*60)

    enhancer = ImageEnhancer()
    img = create_sample_images()[0]

    methods = ['canny', 'sobel', 'laplacian', 'scharr', 'prewitt']

    print("\nComparing edge detection methods:")
    for method in methods:
        edges = enhancer.detect_edges(img, method=method)
        edge_density = (edges > 0).sum() / edges.size
        print(f"  {method:10s}: Edge density = {edge_density:.4f}")


def example_morphological_operations():
    """Example 6: Morphological operations workflow."""
    print("\n" + "="*60)
    print("Example 6: Morphological Operations Workflow")
    print("="*60)

    enhancer = ImageEnhancer()

    # Create binary image
    img = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(img, (30, 30), (70, 70), 255, -1)
    cv2.circle(img, (50, 50), 15, 0, -1)  # Hole in rectangle

    print("\nOriginal image: Rectangle with hole")
    print(f"  White pixels: {(img == 255).sum()}")

    # Erosion - shrinks objects
    print("\n1. Erosion:")
    eroded = enhancer.erode(img, kernel_size=(5, 5))
    print(f"   White pixels: {(eroded == 255).sum()} (decreased)")

    # Dilation - expands objects
    print("\n2. Dilation:")
    dilated = enhancer.dilate(img, kernel_size=(5, 5))
    print(f"   White pixels: {(dilated == 255).sum()} (increased)")

    # Opening - removes small objects
    print("\n3. Opening (erosion → dilation):")
    opened = enhancer.opening(img, kernel_size=(5, 5))
    print(f"   Removes small noise while preserving shape")

    # Closing - fills small holes
    print("\n4. Closing (dilation → erosion):")
    closed = enhancer.closing(img, kernel_size=(5, 5))
    print(f"   Fills holes in objects")

    # Morphological gradient - edge detection
    print("\n5. Morphological Gradient:")
    gradient = enhancer.morphological_gradient(img, kernel_size=(3, 3))
    print(f"   Highlights object boundaries")


def example_normalization_methods():
    """Example 7: Normalization method comparison."""
    print("\n" + "="*60)
    print("Example 7: Normalization Methods")
    print("="*60)

    enhancer = ImageEnhancer()

    # Create image with known statistics
    img = np.array([
        [0, 50, 100],
        [150, 200, 255],
    ], dtype=np.uint8)

    print("\nOriginal image values:")
    print(img)
    print(f"  Min: {img.min()}, Max: {img.max()}")
    print(f"  Mean: {img.mean():.2f}, Std: {img.std():.2f}")

    # MinMax normalization
    print("\n1. MinMax Normalization:")
    norm1 = enhancer.normalize(img, method='minmax')
    print(f"   Range: [{norm1.min():.3f}, {norm1.max():.3f}]")

    # Z-score normalization
    print("\n2. Z-Score Normalization:")
    norm2 = enhancer.normalize(img, method='zscore')
    print(f"   Mean: {norm2.mean():.3f}, Std: {norm2.std():.3f}")

    # L2 normalization
    print("\n3. L2 Normalization:")
    norm3 = enhancer.normalize(img, method='l2')
    print(f"   L2 norm: {np.linalg.norm(norm3):.3f}")

    # Robust normalization
    print("\n4. Robust Normalization:")
    norm4 = enhancer.normalize(img, method='robust')
    print(f"   Range: [{norm4.min():.3f}, {norm4.max():.3f}]")
    print(f"   (Uses IQR, robust to outliers)")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("PyImgAno Preprocessing Module Examples")
    print("="*60)

    example_basic_operations()
    example_preprocessing_pipeline()
    example_detector_with_preprocessing()
    example_different_strategies()
    example_edge_detection_comparison()
    example_morphological_operations()
    example_normalization_methods()

    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
