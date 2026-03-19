"""
Tests for preprocessing module.

Tests cover:
- Edge detection methods
- Morphological operations
- Filters
- Normalization
- Preprocessing pipelines
- Mixin integration
"""

import os
import tempfile

import cv2
import numpy as np
import pytest

from pyimgano.preprocessing import (
    ImageEnhancer,
    PreprocessingMixin,
    PreprocessingPipeline,
    apply_filter,
    edge_detection,
    morphological_operation,
    normalize_image,
)
from pyimgano.preprocessing.mixin import ExampleDetectorWithPreprocessing

# Fixtures


@pytest.fixture
def sample_image():
    """Create a sample grayscale test image."""
    img = np.zeros((100, 100), dtype=np.uint8)
    # Add some patterns
    cv2.rectangle(img, (20, 20), (80, 80), 255, -1)
    cv2.circle(img, (50, 50), 15, 128, -1)
    return img


@pytest.fixture
def sample_color_image():
    """Create a sample color test image."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Add colored patterns
    cv2.rectangle(img, (20, 20), (80, 80), (255, 0, 0), -1)  # Blue
    cv2.circle(img, (50, 50), 15, (0, 255, 0), -1)  # Green
    return img


@pytest.fixture
def temp_image_path(sample_color_image):
    """Create a temporary image file."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        cv2.imwrite(f.name, sample_color_image)
        yield f.name
    os.unlink(f.name)


# Test Edge Detection


class TestEdgeDetection:
    """Test edge detection methods."""

    def test_canny_edge_detection(self, sample_image):
        """Test Canny edge detection."""
        edges = edge_detection(sample_image, method="canny")
        assert edges.shape == sample_image.shape
        assert edges.dtype == np.uint8
        assert np.max(edges) <= 255

    def test_canny_custom_thresholds(self, sample_image):
        """Test Canny with custom thresholds."""
        edges = edge_detection(sample_image, method="canny", threshold1=100, threshold2=200)
        assert edges.shape == sample_image.shape

    def test_sobel_edge_detection(self, sample_image):
        """Test Sobel edge detection."""
        edges = edge_detection(sample_image, method="sobel")
        assert edges.shape == sample_image.shape
        assert edges.dtype == np.uint8

    def test_sobel_x_edge_detection(self, sample_image):
        """Test Sobel X edge detection."""
        edges = edge_detection(sample_image, method="sobel_x")
        assert edges.shape == sample_image.shape

    def test_sobel_y_edge_detection(self, sample_image):
        """Test Sobel Y edge detection."""
        edges = edge_detection(sample_image, method="sobel_y")
        assert edges.shape == sample_image.shape

    def test_laplacian_edge_detection(self, sample_image):
        """Test Laplacian edge detection."""
        edges = edge_detection(sample_image, method="laplacian")
        assert edges.shape == sample_image.shape

    def test_scharr_edge_detection(self, sample_image):
        """Test Scharr edge detection."""
        edges = edge_detection(sample_image, method="scharr")
        assert edges.shape == sample_image.shape

    def test_prewitt_edge_detection(self, sample_image):
        """Test Prewitt edge detection."""
        edges = edge_detection(sample_image, method="prewitt")
        assert edges.shape == sample_image.shape

    def test_invalid_method(self, sample_image):
        """Test invalid edge detection method."""
        with pytest.raises(ValueError):
            edge_detection(sample_image, method="invalid_method")

    def test_color_image_conversion(self, sample_color_image):
        """Test edge detection on color image (should convert to grayscale)."""
        edges = edge_detection(sample_color_image, method="canny")
        assert len(edges.shape) == 2  # Should be grayscale


# Test Morphological Operations


class TestMorphologicalOperations:
    """Test morphological operations."""

    def test_erosion(self, sample_image):
        """Test erosion operation."""
        result = morphological_operation(sample_image, operation="erosion")
        assert result.shape == sample_image.shape
        # Erosion should shrink bright regions
        assert np.sum(result) <= np.sum(sample_image)

    def test_dilation(self, sample_image):
        """Test dilation operation."""
        result = morphological_operation(sample_image, operation="dilation")
        assert result.shape == sample_image.shape
        # Dilation should expand bright regions
        assert np.sum(result) >= np.sum(sample_image)

    def test_opening(self, sample_image):
        """Test opening operation."""
        result = morphological_operation(sample_image, operation="opening")
        assert result.shape == sample_image.shape

    def test_closing(self, sample_image):
        """Test closing operation."""
        result = morphological_operation(sample_image, operation="closing")
        assert result.shape == sample_image.shape

    def test_gradient(self, sample_image):
        """Test morphological gradient."""
        result = morphological_operation(sample_image, operation="gradient")
        assert result.shape == sample_image.shape

    def test_tophat(self, sample_image):
        """Test top-hat operation."""
        result = morphological_operation(sample_image, operation="tophat")
        assert result.shape == sample_image.shape

    def test_blackhat(self, sample_image):
        """Test black-hat operation."""
        result = morphological_operation(sample_image, operation="blackhat")
        assert result.shape == sample_image.shape

    def test_custom_kernel_size(self, sample_image):
        """Test with custom kernel size."""
        result = morphological_operation(sample_image, operation="erosion", kernel_size=(5, 5))
        assert result.shape == sample_image.shape

    def test_ellipse_kernel(self, sample_image):
        """Test with ellipse kernel."""
        result = morphological_operation(sample_image, operation="erosion", kernel_shape="ellipse")
        assert result.shape == sample_image.shape

    def test_cross_kernel(self, sample_image):
        """Test with cross kernel."""
        result = morphological_operation(sample_image, operation="erosion", kernel_shape="cross")
        assert result.shape == sample_image.shape

    def test_multiple_iterations(self, sample_image):
        """Test with multiple iterations."""
        result = morphological_operation(sample_image, operation="erosion", iterations=3)
        assert result.shape == sample_image.shape

    def test_invalid_operation(self, sample_image):
        """Test invalid morphological operation."""
        with pytest.raises(ValueError):
            morphological_operation(sample_image, operation="invalid_op")


# Test Filters


class TestFilters:
    """Test filter operations."""

    def test_gaussian_blur(self, sample_image):
        """Test Gaussian blur."""
        result = apply_filter(sample_image, filter_type="gaussian")
        assert result.shape == sample_image.shape
        assert result.dtype == sample_image.dtype

    def test_gaussian_custom_kernel(self, sample_image):
        """Test Gaussian blur with custom kernel."""
        result = apply_filter(sample_image, filter_type="gaussian", ksize=(7, 7), sigma=2.0)
        assert result.shape == sample_image.shape

    def test_bilateral_filter(self, sample_color_image):
        """Test bilateral filter."""
        result = apply_filter(sample_color_image, filter_type="bilateral")
        assert result.shape == sample_color_image.shape

    def test_median_blur(self, sample_image):
        """Test median blur."""
        result = apply_filter(sample_image, filter_type="median")
        assert result.shape == sample_image.shape

    def test_median_custom_ksize(self, sample_image):
        """Test median blur with custom kernel size."""
        result = apply_filter(sample_image, filter_type="median", ksize=7)
        assert result.shape == sample_image.shape

    def test_box_filter(self, sample_image):
        """Test box/mean filter."""
        result = apply_filter(sample_image, filter_type="box")
        assert result.shape == sample_image.shape

    def test_invalid_filter(self, sample_image):
        """Test invalid filter type."""
        with pytest.raises(ValueError):
            apply_filter(sample_image, filter_type="invalid_filter")


# Test Normalization


class TestNormalization:
    """Test normalization methods."""

    def test_minmax_normalization(self):
        """Test min-max normalization."""
        img = np.array([[0, 50, 100], [150, 200, 255]], dtype=np.uint8)
        result = normalize_image(img, method="minmax")
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_zscore_normalization(self):
        """Test z-score normalization."""
        img = np.array([[0, 50, 100], [150, 200, 255]], dtype=np.uint8)
        result = normalize_image(img, method="zscore")
        assert abs(result.mean()) < 0.1  # Should be close to 0
        assert abs(result.std() - 1.0) < 0.1  # Should be close to 1

    def test_l2_normalization(self):
        """Test L2 normalization."""
        img = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        result = normalize_image(img, method="l2")
        # L2 norm should be 1
        assert abs(np.linalg.norm(result) - 1.0) < 1e-5

    def test_robust_normalization(self):
        """Test robust (IQR-based) normalization."""
        img = np.array([[0, 50, 100], [150, 200, 255]], dtype=np.uint8)
        result = normalize_image(img, method="robust")
        assert result.dtype == np.float32

    def test_invalid_method(self):
        """Test invalid normalization method."""
        img = np.array([[0, 1, 2]], dtype=np.uint8)
        with pytest.raises(ValueError):
            normalize_image(img, method="invalid_method")


# Test ImageEnhancer Class


class TestImageEnhancer:
    """Test ImageEnhancer class."""

    def test_initialization(self):
        """Test ImageEnhancer initialization."""
        enhancer = ImageEnhancer()
        assert enhancer is not None

    def test_detect_edges(self, sample_image):
        """Test edge detection through enhancer."""
        enhancer = ImageEnhancer()
        edges = enhancer.detect_edges(sample_image, method="canny")
        assert edges.shape == sample_image.shape

    def test_gaussian_blur(self, sample_image):
        """Test Gaussian blur through enhancer."""
        enhancer = ImageEnhancer()
        blurred = enhancer.gaussian_blur(sample_image)
        assert blurred.shape == sample_image.shape

    def test_erode(self, sample_image):
        """Test erosion through enhancer."""
        enhancer = ImageEnhancer()
        eroded = enhancer.erode(sample_image)
        assert eroded.shape == sample_image.shape

    def test_dilate(self, sample_image):
        """Test dilation through enhancer."""
        enhancer = ImageEnhancer()
        dilated = enhancer.dilate(sample_image)
        assert dilated.shape == sample_image.shape

    def test_sharpen(self, sample_image):
        """Test sharpening through enhancer."""
        enhancer = ImageEnhancer()
        sharpened = enhancer.sharpen(sample_image)
        assert sharpened.shape == sample_image.shape

    def test_unsharp_mask(self, sample_image):
        """Test unsharp mask through enhancer."""
        enhancer = ImageEnhancer()
        result = enhancer.unsharp_mask(sample_image)
        assert result.shape == sample_image.shape

    def test_clahe(self, sample_image):
        """Test CLAHE through enhancer."""
        enhancer = ImageEnhancer()
        result = enhancer.clahe(sample_image)
        assert result.shape == sample_image.shape

    def test_normalize(self, sample_image):
        """Test normalization through enhancer."""
        enhancer = ImageEnhancer()
        result = enhancer.normalize(sample_image)
        assert result.min() >= 0.0
        assert result.max() <= 1.0


# Test PreprocessingPipeline


class TestPreprocessingPipeline:
    """Test preprocessing pipeline."""

    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        pipeline = PreprocessingPipeline()
        assert len(pipeline) == 0

    def test_add_single_step(self):
        """Test adding a single step."""
        pipeline = PreprocessingPipeline()
        pipeline.add_step("gaussian_blur", ksize=(5, 5))
        assert len(pipeline) == 1

    def test_add_multiple_steps(self):
        """Test adding multiple steps."""
        pipeline = PreprocessingPipeline()
        pipeline.add_step("gaussian_blur", ksize=(5, 5))
        pipeline.add_step("detect_edges", method="canny")
        pipeline.add_step("normalize", method="minmax")
        assert len(pipeline) == 3

    def test_pipeline_transform(self, sample_image):
        """Test pipeline transformation."""
        pipeline = PreprocessingPipeline()
        pipeline.add_step("gaussian_blur", ksize=(5, 5))
        pipeline.add_step("normalize", method="minmax")

        result = pipeline.transform(sample_image)
        assert result.shape == sample_image.shape

    def test_pipeline_chaining(self, sample_image):
        """Test pipeline method chaining."""
        pipeline = (
            PreprocessingPipeline()
            .add_step("gaussian_blur", ksize=(5, 5))
            .add_step("detect_edges", method="canny")
            .add_step("normalize", method="minmax")
        )

        result = pipeline.transform(sample_image)
        assert result is not None

    def test_pipeline_clear(self):
        """Test clearing pipeline."""
        pipeline = PreprocessingPipeline()
        pipeline.add_step("gaussian_blur")
        pipeline.add_step("normalize")
        assert len(pipeline) > 0

        pipeline.clear()
        assert len(pipeline) == 0

    def test_pipeline_repr(self):
        """Test pipeline string representation."""
        pipeline = PreprocessingPipeline()
        pipeline.add_step("gaussian_blur")
        repr_str = repr(pipeline)
        assert "PreprocessingPipeline" in repr_str
        assert "1 step" in repr_str

    def test_invalid_operation(self):
        """Test adding invalid operation."""
        pipeline = PreprocessingPipeline()
        with pytest.raises(ValueError):
            pipeline.add_step("invalid_operation")


# Test PreprocessingMixin


class TestPreprocessingMixin:
    """Test preprocessing mixin."""

    def test_setup_preprocessing_disabled(self):
        """Test setup with preprocessing disabled."""

        class TestDetector(PreprocessingMixin):
            pass

        detector = TestDetector()
        detector.setup_preprocessing(enable=False)

        assert not detector.preprocessing_enabled
        assert not hasattr(detector, "enhancer")

    def test_setup_preprocessing_enabled(self):
        """Test setup with preprocessing enabled."""

        class TestDetector(PreprocessingMixin):
            pass

        detector = TestDetector()
        detector.setup_preprocessing(enable=True, use_pipeline=False)

        assert detector.preprocessing_enabled
        assert hasattr(detector, "enhancer")
        assert not detector.use_preprocessing_pipeline

    def test_setup_preprocessing_with_pipeline(self):
        """Test setup with pipeline mode."""

        class TestDetector(PreprocessingMixin):
            pass

        detector = TestDetector()
        detector.setup_preprocessing(enable=True, use_pipeline=True)

        assert detector.preprocessing_enabled
        assert detector.use_preprocessing_pipeline
        assert hasattr(detector, "preprocessing_pipeline")

    def test_add_preprocessing_step(self):
        """Test adding preprocessing step."""

        class TestDetector(PreprocessingMixin):
            pass

        detector = TestDetector()
        detector.setup_preprocessing(enable=True, use_pipeline=True)
        detector.add_preprocessing_step("gaussian_blur", ksize=(5, 5))

        assert len(detector.preprocessing_pipeline) == 1

    def test_add_step_without_pipeline(self):
        """Test adding step without pipeline mode."""

        class TestDetector(PreprocessingMixin):
            pass

        detector = TestDetector()
        detector.setup_preprocessing(enable=True, use_pipeline=False)

        with pytest.raises(RuntimeError):
            detector.add_preprocessing_step("gaussian_blur")

    def test_add_step_without_setup(self):
        """Test adding step without setup."""

        class TestDetector(PreprocessingMixin):
            pass

        detector = TestDetector()
        detector.preprocessing_enabled = False

        with pytest.raises(RuntimeError):
            detector.add_preprocessing_step("gaussian_blur")

    def test_preprocess_image_disabled(self, sample_color_image, temp_image_path):
        """Test preprocessing when disabled."""

        class TestDetector(PreprocessingMixin):
            pass

        detector = TestDetector()
        detector.setup_preprocessing(enable=False)

        # With path
        result = detector.preprocess_image(temp_image_path)
        assert result is not None

        # With array
        result = detector.preprocess_image(sample_color_image)
        assert np.array_equal(result, sample_color_image)

    def test_preprocess_image_single_operation(self, sample_color_image):
        """Test preprocessing with single operation."""

        class TestDetector(PreprocessingMixin):
            pass

        detector = TestDetector()
        detector.setup_preprocessing(enable=True, use_pipeline=False)

        result = detector.preprocess_image(
            sample_color_image, operation="gaussian_blur", ksize=(5, 5)
        )
        assert result.shape == sample_color_image.shape

    def test_preprocess_image_pipeline(self, sample_color_image):
        """Test preprocessing with pipeline."""

        class TestDetector(PreprocessingMixin):
            pass

        detector = TestDetector()
        detector.setup_preprocessing(enable=True, use_pipeline=True)
        detector.add_preprocessing_step("gaussian_blur", ksize=(5, 5))
        detector.add_preprocessing_step("normalize", method="minmax")

        result = detector.preprocess_image(sample_color_image)
        assert result is not None

    def test_preprocess_image_from_path(self, temp_image_path):
        """Test preprocessing from image path."""

        class TestDetector(PreprocessingMixin):
            pass

        detector = TestDetector()
        detector.setup_preprocessing(enable=True, use_pipeline=True)
        detector.add_preprocessing_step("gaussian_blur", ksize=(5, 5))

        result = detector.preprocess_image(temp_image_path)
        assert result is not None

    def test_preprocess_images_batch(self, temp_image_path):
        """Test preprocessing multiple images."""

        class TestDetector(PreprocessingMixin):
            pass

        detector = TestDetector()
        detector.setup_preprocessing(enable=True, use_pipeline=True)
        detector.add_preprocessing_step("gaussian_blur", ksize=(5, 5))

        images = [temp_image_path, temp_image_path, temp_image_path]
        results = detector.preprocess_images(images)

        assert len(results) == 3
        for result in results:
            assert result is not None

    def test_convenience_method_edges(self, sample_color_image):
        """Test convenience method for edge detection."""

        class TestDetector(PreprocessingMixin):
            pass

        detector = TestDetector()
        detector.setup_preprocessing(enable=True)

        edges = detector.preprocess_with_edges(sample_color_image, method="canny")
        assert edges is not None

    def test_convenience_method_blur(self, sample_color_image):
        """Test convenience method for blur."""

        class TestDetector(PreprocessingMixin):
            pass

        detector = TestDetector()
        detector.setup_preprocessing(enable=True)

        blurred = detector.preprocess_with_blur(sample_color_image, filter_type="gaussian")
        assert blurred is not None

    def test_convenience_method_morphology(self, sample_image):
        """Test convenience method for morphology."""

        class TestDetector(PreprocessingMixin):
            pass

        detector = TestDetector()
        detector.setup_preprocessing(enable=True)

        result = detector.preprocess_with_morphology(sample_image, operation="erosion")
        assert result is not None

    def test_clear_pipeline(self):
        """Test clearing preprocessing pipeline."""

        class TestDetector(PreprocessingMixin):
            pass

        detector = TestDetector()
        detector.setup_preprocessing(enable=True, use_pipeline=True)
        detector.add_preprocessing_step("gaussian_blur")
        detector.add_preprocessing_step("normalize")

        assert len(detector.preprocessing_pipeline) == 2

        detector.clear_preprocessing_pipeline()
        assert len(detector.preprocessing_pipeline) == 0

    def test_get_preprocessing_info(self):
        """Test getting preprocessing info."""

        class TestDetector(PreprocessingMixin):
            pass

        detector = TestDetector()
        detector.setup_preprocessing(enable=True, use_pipeline=True)
        detector.add_preprocessing_step("gaussian_blur")
        detector.add_preprocessing_step("normalize")

        info = detector.get_preprocessing_info()
        assert info["enabled"] is True
        assert info["pipeline_mode"] is True
        assert info["pipeline_steps"] == 2
        assert "gaussian_blur" in info["steps"]
        assert "normalize" in info["steps"]


# Integration Tests


class TestIntegration:
    """Integration tests for preprocessing module."""

    def test_full_pipeline_workflow(self, sample_color_image):
        """Test complete preprocessing workflow."""
        # Create pipeline
        pipeline = PreprocessingPipeline()
        pipeline.add_step("gaussian_blur", ksize=(5, 5))
        pipeline.add_step("detect_edges", method="canny")
        pipeline.add_step("normalize", method="minmax")

        # Apply pipeline
        result = pipeline.transform(sample_color_image)

        assert result is not None
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_detector_with_preprocessing(self, temp_image_path):
        """Test detector with preprocessing capabilities."""

        class MockDetector(PreprocessingMixin):
            def __init__(self):
                self.setup_preprocessing(enable=True, use_pipeline=True)
                self.add_preprocessing_step("gaussian_blur", ksize=(5, 5))
                self.add_preprocessing_step("normalize", method="minmax")

            def fit(self, x):
                x_processed = self.preprocess_images(x)
                return x_processed

        detector = MockDetector()
        images = [temp_image_path, temp_image_path]
        processed = detector.fit(images)

        assert len(processed) == 2
        for img in processed:
            assert img is not None

    def test_example_detector_predict_returns_finite_scores(self, sample_color_image):
        """Example detector should return one finite score per preprocessed image."""
        detector = ExampleDetectorWithPreprocessing()
        scores = detector.predict([sample_color_image, sample_color_image, sample_color_image])

        assert scores.shape == (3,)
        assert np.all(np.isfinite(scores))

    def test_example_detector_predict_does_not_mutate_global_numpy_state(
        self, sample_color_image
    ):
        """Example detector prediction should not touch the module-level NumPy RNG."""
        detector = ExampleDetectorWithPreprocessing()
        global_rng = np.random.mtrand._rand
        before = global_rng.get_state()

        try:
            detector.predict([sample_color_image, sample_color_image])
            after = global_rng.get_state()
        finally:
            global_rng.set_state(before)

        assert after[0] == before[0]
        assert np.array_equal(after[1], before[1])
        assert after[2:] == before[2:]

    def test_edge_detection_workflow(self, sample_color_image):
        """Test edge detection workflow."""
        enhancer = ImageEnhancer()

        # Blur to reduce noise
        blurred = enhancer.gaussian_blur(sample_color_image, ksize=(5, 5))

        # Detect edges
        edges = enhancer.detect_edges(blurred, method="canny")

        # Normalize
        normalized = enhancer.normalize(edges, method="minmax")

        assert normalized is not None
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0

    def test_morphological_workflow(self, sample_image):
        """Test morphological operations workflow."""
        enhancer = ImageEnhancer()

        # Opening (erosion then dilation) - removes small objects
        opened = enhancer.opening(sample_image, kernel_size=(5, 5))

        # Closing (dilation then erosion) - closes small holes
        closed = enhancer.closing(opened, kernel_size=(5, 5))

        assert closed.shape == sample_image.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
