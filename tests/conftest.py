"""
Pytest configuration and fixtures for PyImgAno tests.
"""

import os
import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pytest
from PIL import Image

_GLOBAL_NUMPY_RNG = np.random.mtrand._rand


def _fixture_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _torch_cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False

    cuda = getattr(torch, "cuda", None)
    is_available = getattr(cuda, "is_available", None)
    return bool(callable(is_available) and is_available())


def _seed_torch_if_available(seed: int) -> None:
    try:
        import torch
    except ImportError:
        return

    manual_seed = getattr(torch, "manual_seed", None)
    if callable(manual_seed):
        manual_seed(seed)

    cuda = getattr(torch, "cuda", None)
    is_available = getattr(cuda, "is_available", None)
    manual_seed_cuda = getattr(cuda, "manual_seed", None)
    if callable(is_available) and is_available() and callable(manual_seed_cuda):
        manual_seed_cuda(seed)


@pytest.fixture(scope="session")
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_image() -> np.ndarray:
    """Create a sample image for testing."""
    # Create a simple 100x100 RGB image
    image = _fixture_rng(0).integers(0, 255, (100, 100, 3), dtype=np.uint8)
    return image


@pytest.fixture
def sample_image_path(temp_dir: Path, sample_image: np.ndarray) -> Path:
    """Create and save a sample image, return its path."""
    image_path = temp_dir / "test_image.jpg"
    pil_image = Image.fromarray(sample_image)
    pil_image.save(image_path)
    return image_path


@pytest.fixture
def sample_image_paths(temp_dir: Path) -> list[Path]:
    """Create multiple sample images and return their paths."""
    paths = []
    rng = _fixture_rng(1)
    for i in range(5):
        image = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
        image_path = temp_dir / f"test_image_{i}.jpg"
        pil_image = Image.fromarray(image)
        pil_image.save(image_path)
        paths.append(image_path)
    return paths


@pytest.fixture
def sample_grayscale_image() -> np.ndarray:
    """Create a sample grayscale image for testing."""
    image = _fixture_rng(2).integers(0, 255, (100, 100), dtype=np.uint8)
    return image


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test for reproducibility."""
    _GLOBAL_NUMPY_RNG.seed(42)
    _seed_torch_if_available(42)


@pytest.fixture
def mock_dataset_paths(temp_dir: Path) -> dict[str, list[Path]]:
    """Create a mock dataset structure with train and test images."""
    train_dir = temp_dir / "train"
    test_dir = temp_dir / "test"
    train_dir.mkdir()
    test_dir.mkdir()

    train_paths = []
    test_paths = []
    rng = _fixture_rng(3)

    # Create training images
    for i in range(10):
        image = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
        image_path = train_dir / f"train_{i}.jpg"
        pil_image = Image.fromarray(image)
        pil_image.save(image_path)
        train_paths.append(image_path)

    # Create test images
    for i in range(5):
        image = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
        image_path = test_dir / f"test_{i}.jpg"
        pil_image = Image.fromarray(image)
        pil_image.save(image_path)
        test_paths.append(image_path)

    return {"train": train_paths, "test": test_paths}


# Markers for different test categories
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "requires_gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "requires_torch: marks tests that require PyTorch")


# Skip GPU tests if CUDA is not available
def pytest_collection_modifyitems(config, items):
    """Modify test collection to skip GPU tests if CUDA is not available."""
    del config
    has_cuda = _torch_cuda_available()

    skip_gpu = pytest.mark.skip(reason="CUDA not available")

    for item in items:
        if "requires_gpu" in item.keywords and not has_cuda:
            item.add_marker(skip_gpu)
