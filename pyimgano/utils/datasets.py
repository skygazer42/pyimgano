"""
Dataset loading utilities for standard anomaly detection benchmarks.

Provides easy-to-use loaders for popular datasets:
- MVTec AD
- BTAD
- VisA
- Custom datasets

Example:
    >>> from pyimgano.utils.datasets import MVTecDataset
    >>> dataset = MVTecDataset(root='./mvtec_ad', category='bottle')
    >>> train_data = dataset.get_train_data()
    >>> test_data, test_labels = dataset.get_test_data()
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass

from pyimgano.io.image import read_image, resize_image


@dataclass
class DatasetInfo:
    """Dataset information."""
    name: str
    categories: List[str]
    num_train: int
    num_test: int
    image_size: Tuple[int, int]
    description: str


class BaseDataset:
    """Base class for anomaly detection datasets."""

    def __init__(self, root: str, category: Optional[str] = None):
        self.root = Path(root)
        self.category = category

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {root}")

    def get_train_data(self) -> NDArray:
        """Get training data (normal only).

        Returns:
            Training images [N, H, W, C]
        """
        raise NotImplementedError

    def get_test_data(self) -> Tuple[NDArray, NDArray, Optional[NDArray]]:
        """Get test data with labels and masks.

        Returns:
            test_images: Test images [N, H, W, C]
            test_labels: Binary labels [N] (0=normal, 1=anomaly)
            test_masks: Ground truth masks [N, H, W] or None
        """
        raise NotImplementedError

    def get_info(self) -> DatasetInfo:
        """Get dataset information.

        Returns:
            Dataset information
        """
        raise NotImplementedError


class MVTecDataset(BaseDataset):
    """MVTec AD dataset loader.

    MVTec AD is a widely-used benchmark for industrial anomaly detection.
    Contains 15 categories with texture and object classes.

    Categories:
        Textures: carpet, grid, leather, tile, wood
        Objects: bottle, cable, capsule, hazelnut, metal_nut,
                 pill, screw, toothbrush, transistor, zipper

    Args:
        root: Path to MVTec AD dataset root
        category: Category name (e.g., 'bottle', 'carpet')
        resize: Target size for images (H, W)
        load_masks: Whether to load ground truth masks

    Example:
        >>> dataset = MVTecDataset(
        ...     root='./mvtec_ad',
        ...     category='bottle',
        ...     resize=(256, 256)
        ... )
        >>> train_imgs = dataset.get_train_data()
        >>> test_imgs, labels, masks = dataset.get_test_data()
    """

    CATEGORIES = [
        'bottle', 'cable', 'capsule', 'carpet', 'grid',
        'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
        'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
    ]

    def __init__(
        self,
        root: str,
        category: str,
        resize: Optional[Tuple[int, int]] = None,
        load_masks: bool = True
    ):
        super().__init__(root, category)

        if category not in self.CATEGORIES:
            raise ValueError(f"Invalid category. Choose from: {self.CATEGORIES}")

        self.resize = resize
        self.load_masks = load_masks
        self.category_path = self.root / category

    def _load_images(self, path: Path) -> List[NDArray]:
        """Load all images from a directory."""
        images = []

        for img_path in sorted(path.glob('*.png')):
            img = read_image(img_path, color="rgb")

            if self.resize is not None:
                img = resize_image(img, self.resize)

            images.append(img)

        return images

    def get_train_data(self) -> NDArray:
        """Get training data (normal images only)."""
        train_path = self.category_path / 'train' / 'good'

        if not train_path.exists():
            raise FileNotFoundError(f"Training data not found: {train_path}")

        images = self._load_images(train_path)
        return np.array(images)

    def get_test_data(self) -> Tuple[NDArray, NDArray, Optional[NDArray]]:
        """Get test data with labels and optionally masks."""
        test_path = self.category_path / 'test'
        ground_truth_path = self.category_path / 'ground_truth'

        if not test_path.exists():
            raise FileNotFoundError(f"Test data not found: {test_path}")

        images = []
        labels = []
        masks = [] if self.load_masks else None

        # Load normal test images
        normal_path = test_path / 'good'
        if normal_path.exists():
            normal_imgs = self._load_images(normal_path)
            images.extend(normal_imgs)
            labels.extend([0] * len(normal_imgs))

            if self.load_masks:
                # Normal images have no masks (all zeros)
                for img in normal_imgs:
                    masks.append(np.zeros(img.shape[:2], dtype=np.uint8))

        # Load anomaly test images
        for defect_dir in sorted(test_path.iterdir()):
            if defect_dir.name == 'good':
                continue

            if not defect_dir.is_dir():
                continue

            defect_imgs = self._load_images(defect_dir)
            images.extend(defect_imgs)
            labels.extend([1] * len(defect_imgs))

            # Load masks if requested
            if self.load_masks and ground_truth_path.exists():
                mask_dir = ground_truth_path / defect_dir.name
                if mask_dir.exists():
                    for img_path in sorted(defect_dir.glob('*.png')):
                        mask_path = mask_dir / f"{img_path.stem}_mask.png"
                        if mask_path.exists():
                            mask = read_image(mask_path, color="gray")
                            if self.resize is not None:
                                mask = resize_image(mask, self.resize, is_mask=True)
                            # Binary mask
                            mask = (mask > 127).astype(np.uint8)
                            masks.append(mask)
                        else:
                            # If mask not found, create zero mask
                            masks.append(np.zeros(self.resize or defect_imgs[0].shape[:2], dtype=np.uint8))

        images = np.array(images)
        labels = np.array(labels)
        masks = np.array(masks) if self.load_masks else None

        return images, labels, masks

    def get_info(self) -> DatasetInfo:
        """Get dataset information."""
        train_data = self.get_train_data()
        test_data, test_labels, _ = self.get_test_data()

        return DatasetInfo(
            name='MVTec AD',
            categories=self.CATEGORIES,
            num_train=len(train_data),
            num_test=len(test_data),
            image_size=train_data[0].shape[:2],
            description=f'MVTec AD - {self.category} category'
        )

    def get_train_paths(self) -> List[str]:
        """Get training image paths (normal only)."""
        train_path = self.category_path / 'train' / 'good'
        if not train_path.exists():
            raise FileNotFoundError(f"Training data not found: {train_path}")
        paths = [str(p) for p in sorted(train_path.glob('*.png'))]
        if not paths:
            raise ValueError(f"No training images found in: {train_path}")
        return paths

    def get_test_paths(self) -> Tuple[List[str], NDArray, Optional[NDArray]]:
        """Get test image paths with labels and optionally masks."""
        test_path = self.category_path / 'test'
        ground_truth_path = self.category_path / 'ground_truth'

        if not test_path.exists():
            raise FileNotFoundError(f"Test data not found: {test_path}")

        image_paths: List[str] = []
        labels: List[int] = []
        masks: Optional[List[NDArray]] = [] if self.load_masks else None

        # Normal test images
        normal_dir = test_path / 'good'
        normal_paths = sorted(normal_dir.glob('*.png')) if normal_dir.exists() else []
        image_paths.extend([str(p) for p in normal_paths])
        labels.extend([0] * len(normal_paths))

        if self.load_masks:
            for img_path in normal_paths:
                shape = self.resize
                if shape is None:
                    try:
                        img = read_image(img_path, color="bgr")
                    except FileNotFoundError:
                        img = None
                    shape = img.shape[:2] if img is not None else (256, 256)
                masks.append(np.zeros(shape, dtype=np.uint8))

        # Anomaly test images (all subdirs except good)
        for defect_dir in sorted(test_path.iterdir()):
            if defect_dir.name == 'good' or not defect_dir.is_dir():
                continue

            defect_paths = sorted(defect_dir.glob('*.png'))
            image_paths.extend([str(p) for p in defect_paths])
            labels.extend([1] * len(defect_paths))

            if not self.load_masks:
                continue

            mask_dir = ground_truth_path / defect_dir.name
            for img_path in defect_paths:
                mask_path = mask_dir / f"{img_path.stem}_mask.png"
                mask = None
                if mask_path.exists():
                    try:
                        mask = read_image(mask_path, color="gray")
                    except FileNotFoundError:
                        mask = None
                if mask is None:
                    shape = self.resize
                    if shape is None:
                        try:
                            img = read_image(img_path, color="bgr")
                        except FileNotFoundError:
                            img = None
                        shape = img.shape[:2] if img is not None else (256, 256)
                    mask = np.zeros(shape, dtype=np.uint8)
                elif self.resize is not None:
                    mask = resize_image(mask, self.resize, is_mask=True)
                masks.append((mask > 127).astype(np.uint8))

        return image_paths, np.array(labels), np.array(masks) if self.load_masks else None

    @staticmethod
    def list_categories() -> List[str]:
        """List all available categories."""
        return MVTecDataset.CATEGORIES


class MVTecLOCODataset(BaseDataset):
    """MVTec LOCO AD dataset loader.

    LOCO extends MVTec-style industrial inspection with:
    - structural anomalies
    - logical anomalies

    Expected structure (per category):
        <root>/<category>/
            train/good/*.png
            validation/good/*.png               (optional)
            test/good/*.png
            test/logical_anomalies/**.png
            test/structural_anomalies/**.png
            ground_truth/logical_anomalies/**.png     (optional)
            ground_truth/structural_anomalies/**.png  (optional)

    Notes
    -----
    - Some exports nest anomaly images one-level deeper (e.g. by defect id/name). This loader scans
      recursively under the anomaly directories.
    - Mask naming differs across exports. This loader tries multiple candidates and falls back to zeros.
    """

    CATEGORIES = [
        "breakfast_box",
        "juice_bottle",
        "pushpins",
        "screw_bag",
        "splicing_connectors",
    ]

    _ANOMALY_SPLITS = ("logical_anomalies", "structural_anomalies")

    def __init__(
        self,
        root: str,
        category: str,
        resize: Optional[Tuple[int, int]] = None,
        load_masks: bool = True,
        include_validation_in_train: bool = False,
    ) -> None:
        super().__init__(root, category)

        if category not in self.CATEGORIES:
            raise ValueError(f"Invalid category. Choose from: {self.CATEGORIES}")

        self.resize = resize
        self.load_masks = load_masks
        self.include_validation_in_train = bool(include_validation_in_train)
        self.category_path = self.root / category

        if not self.category_path.exists():
            raise FileNotFoundError(f"MVTec LOCO category path not found: {self.category_path}")

    @staticmethod
    def _scan_images(directory: Path) -> List[Path]:
        if not directory.exists():
            return []
        paths: List[Path] = []
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp"]:
            paths.extend(sorted(directory.rglob(ext)))
        return paths

    def _load_images(self, paths: List[Path]) -> List[NDArray]:
        images: List[NDArray] = []
        for img_path in paths:
            img = read_image(img_path, color="rgb")
            if self.resize is not None:
                img = resize_image(img, self.resize)
            images.append(img)
        return images

    def _zeros_mask_for_image(self, image_path: Path) -> NDArray:
        if self.resize is not None:
            shape = self.resize
        else:
            try:
                img = read_image(image_path, color="bgr")
            except FileNotFoundError:
                img = None
            shape = img.shape[:2] if img is not None else (256, 256)
        return np.zeros(shape, dtype=np.uint8)

    def _load_mask_for_test_image(self, image_path: Path) -> NDArray:
        if not self.load_masks:
            raise RuntimeError("load_masks is False")

        test_dir = self.category_path / "test"
        gt_dir = self.category_path / "ground_truth"

        try:
            rel = image_path.relative_to(test_dir)
        except Exception:
            return self._zeros_mask_for_image(image_path)

        # Candidate 1: same relative path under ground_truth
        candidates: List[Path] = []
        candidates.append(gt_dir / rel)

        # Candidate 2: same dir, stem + _mask (MVTec AD-style)
        candidates.append((gt_dir / rel).with_name(f"{image_path.stem}_mask{image_path.suffix}"))

        # Candidate 3: some exports store masks with the same name (no suffix) under ground_truth
        candidates.append((gt_dir / rel).with_name(image_path.name))

        # Candidate 4: some exports add an extra "<stem>/" folder level
        candidates.append(gt_dir / rel.parent / image_path.stem / image_path.name)
        candidates.append(
            gt_dir
            / rel.parent
            / image_path.stem
            / f"{image_path.stem}_mask{image_path.suffix}"
        )

        mask = None
        for candidate in candidates:
            if candidate.exists():
                try:
                    mask = read_image(candidate, color="gray")
                except FileNotFoundError:
                    mask = None
                else:
                    break

        if mask is None:
            mask = self._zeros_mask_for_image(image_path)
        elif self.resize is not None:
            mask = resize_image(mask, self.resize, is_mask=True)

        return (mask > 127).astype(np.uint8)

    def get_train_data(self) -> NDArray:
        train_good_dir = self.category_path / "train" / "good"
        if not train_good_dir.exists():
            raise FileNotFoundError(f"Training data not found: {train_good_dir}")

        train_paths = self._scan_images(train_good_dir)
        if self.include_validation_in_train:
            val_dir = self.category_path / "validation" / "good"
            train_paths.extend(self._scan_images(val_dir))

        images = self._load_images(train_paths)
        if not images:
            raise ValueError(f"No training images found in: {train_good_dir}")
        return np.array(images)

    def get_test_data(self) -> Tuple[NDArray, NDArray, Optional[NDArray]]:
        test_dir = self.category_path / "test"
        if not test_dir.exists():
            raise FileNotFoundError(f"Test data not found: {test_dir}")

        good_paths = self._scan_images(test_dir / "good")
        anomaly_paths: List[Path] = []
        for split in self._ANOMALY_SPLITS:
            anomaly_paths.extend(self._scan_images(test_dir / split))

        test_paths = good_paths + anomaly_paths
        images = self._load_images(test_paths)
        labels = np.array([0] * len(good_paths) + [1] * len(anomaly_paths))

        if not self.load_masks:
            return np.array(images), labels, None

        masks: List[NDArray] = []
        for p in good_paths:
            masks.append(self._zeros_mask_for_image(p))
        for p in anomaly_paths:
            masks.append(self._load_mask_for_test_image(p))

        return np.array(images), labels, np.array(masks)

    def get_info(self) -> DatasetInfo:
        train_data = self.get_train_data()
        test_data, _, _ = self.get_test_data()
        return DatasetInfo(
            name="MVTec LOCO AD",
            categories=self.CATEGORIES,
            num_train=len(train_data),
            num_test=len(test_data),
            image_size=train_data[0].shape[:2],
            description=f"MVTec LOCO AD - {self.category} category",
        )

    def get_train_paths(self) -> List[str]:
        train_good_dir = self.category_path / "train" / "good"
        if not train_good_dir.exists():
            raise FileNotFoundError(f"Training data not found: {train_good_dir}")

        train_paths = self._scan_images(train_good_dir)
        if self.include_validation_in_train:
            val_dir = self.category_path / "validation" / "good"
            train_paths.extend(self._scan_images(val_dir))

        out = [str(p) for p in sorted(train_paths)]
        if not out:
            raise ValueError(f"No training images found in: {train_good_dir}")
        return out

    def get_test_paths(self) -> Tuple[List[str], NDArray, Optional[NDArray]]:
        test_dir = self.category_path / "test"
        if not test_dir.exists():
            raise FileNotFoundError(f"Test data not found: {test_dir}")

        good_paths = self._scan_images(test_dir / "good")
        anomaly_paths: List[Path] = []
        for split in self._ANOMALY_SPLITS:
            anomaly_paths.extend(self._scan_images(test_dir / split))

        test_paths = [str(p) for p in list(good_paths) + list(anomaly_paths)]
        labels = np.array([0] * len(good_paths) + [1] * len(anomaly_paths))

        if not self.load_masks:
            return test_paths, labels, None

        masks: List[NDArray] = []
        for p in good_paths:
            masks.append(self._zeros_mask_for_image(p))
        for p in anomaly_paths:
            masks.append(self._load_mask_for_test_image(p))

        return test_paths, labels, np.array(masks)

    @staticmethod
    def list_categories() -> List[str]:
        return list(MVTecLOCODataset.CATEGORIES)


class MVTecAD2Dataset(BaseDataset):
    """MVTec AD 2 dataset loader (paths-first).

    Expected structure (per category):
        <root>/<category>/
            train/good/*.png
            validation/good/*.png
            test_public/good/*.png
            test_public/bad/*.png
            test_public/ground_truth/bad/*.png   (optional; usually *_mask.png)

    Notes
    -----
    - AD2 also provides private test splits that do not ship public GT. This loader defaults to
      `split="test_public"` for evaluation.
    """

    def __init__(
        self,
        root: str,
        category: str,
        *,
        split: str = "test_public",
        resize: Optional[Tuple[int, int]] = None,
        load_masks: bool = True,
    ) -> None:
        super().__init__(root, category)
        self.split = str(split)
        self.resize = resize
        self.load_masks = bool(load_masks)

        self.category_path = self.root / category
        if not self.category_path.exists():
            raise FileNotFoundError(f"MVTec AD 2 category path not found: {self.category_path}")

    @staticmethod
    def _scan_images(directory: Path) -> List[Path]:
        if not directory.exists():
            return []
        paths: List[Path] = []
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp"]:
            paths.extend(sorted(directory.rglob(ext)))
        return paths

    def _load_images(self, paths: List[Path]) -> List[NDArray]:
        images: List[NDArray] = []
        for img_path in paths:
            img = read_image(img_path, color="rgb")
            if self.resize is not None:
                img = resize_image(img, self.resize)
            images.append(img)
        return images

    def _zeros_mask_for_image(self, image_path: Path) -> NDArray:
        if self.resize is not None:
            shape = self.resize
        else:
            try:
                img = read_image(image_path, color="bgr")
            except FileNotFoundError:
                img = None
            shape = img.shape[:2] if img is not None else (256, 256)
        return np.zeros(shape, dtype=np.uint8)

    def _load_mask_for_bad_image(self, bad_image_path: Path) -> NDArray:
        if not self.load_masks:
            raise RuntimeError("load_masks is False")

        split_dir = self.category_path / self.split
        gt_bad_dir = split_dir / "ground_truth" / "bad"

        candidates: List[Path] = []
        candidates.append(gt_bad_dir / bad_image_path.name)
        candidates.append(gt_bad_dir / f"{bad_image_path.stem}_mask{bad_image_path.suffix}")
        candidates.append(gt_bad_dir / f"{bad_image_path.stem}_mask.png")
        candidates.append(gt_bad_dir / f"{bad_image_path.stem}.png")

        mask = None
        for candidate in candidates:
            if candidate.exists():
                try:
                    mask = read_image(candidate, color="gray")
                except FileNotFoundError:
                    mask = None
                else:
                    break

        if mask is None:
            mask = self._zeros_mask_for_image(bad_image_path)
        elif self.resize is not None:
            mask = resize_image(mask, self.resize, is_mask=True)

        return (mask > 127).astype(np.uint8)

    def get_train_data(self) -> NDArray:
        train_good_dir = self.category_path / "train" / "good"
        if not train_good_dir.exists():
            raise FileNotFoundError(f"Training data not found: {train_good_dir}")
        train_paths = self._scan_images(train_good_dir)
        images = self._load_images(train_paths)
        if not images:
            raise ValueError(f"No training images found in: {train_good_dir}")
        return np.array(images)

    def get_test_data(self) -> Tuple[NDArray, NDArray, Optional[NDArray]]:
        split_dir = self.category_path / self.split
        if not split_dir.exists():
            raise FileNotFoundError(f"Test split not found: {split_dir}")

        good_paths = self._scan_images(split_dir / "good")
        bad_paths = self._scan_images(split_dir / "bad")
        test_paths = good_paths + bad_paths
        images = self._load_images(test_paths)
        labels = np.array([0] * len(good_paths) + [1] * len(bad_paths))

        if not self.load_masks:
            return np.array(images), labels, None

        masks: List[NDArray] = []
        for p in good_paths:
            masks.append(self._zeros_mask_for_image(p))
        for p in bad_paths:
            masks.append(self._load_mask_for_bad_image(p))

        return np.array(images), labels, np.array(masks)

    def get_info(self) -> DatasetInfo:
        train_data = self.get_train_data()
        test_data, _, _ = self.get_test_data()
        return DatasetInfo(
            name="MVTec AD 2",
            categories=[self.category] if self.category else [],
            num_train=len(train_data),
            num_test=len(test_data),
            image_size=train_data[0].shape[:2],
            description=f"MVTec AD 2 - {self.category} category ({self.split})",
        )

    def get_train_paths(self) -> List[str]:
        train_good_dir = self.category_path / "train" / "good"
        if not train_good_dir.exists():
            raise FileNotFoundError(f"Training data not found: {train_good_dir}")
        paths = [str(p) for p in self._scan_images(train_good_dir)]
        if not paths:
            raise ValueError(f"No training images found in: {train_good_dir}")
        return paths

    def get_test_paths(self) -> Tuple[List[str], NDArray, Optional[NDArray]]:
        split_dir = self.category_path / self.split
        if not split_dir.exists():
            raise FileNotFoundError(f"Test split not found: {split_dir}")

        good_paths = self._scan_images(split_dir / "good")
        bad_paths = self._scan_images(split_dir / "bad")
        test_paths = [str(p) for p in good_paths + bad_paths]
        labels = np.array([0] * len(good_paths) + [1] * len(bad_paths))

        if not self.load_masks:
            return test_paths, labels, None

        masks: List[NDArray] = []
        for p in good_paths:
            masks.append(self._zeros_mask_for_image(p))
        for p in bad_paths:
            masks.append(self._load_mask_for_bad_image(p))
        return test_paths, labels, np.array(masks)

    @staticmethod
    def list_categories(root: str) -> List[str]:
        root_path = Path(root)
        if not root_path.exists():
            return []
        return sorted([p.name for p in root_path.iterdir() if p.is_dir()])


class BTADDataset(BaseDataset):
    """BTAD (BeanTech Anomaly Detection) dataset loader.

    BTAD contains 3 industrial product categories.

    Categories:
        01: Industrial product 1
        02: Industrial product 2
        03: Industrial product 3

    Args:
        root: Path to BTAD dataset root
        category: Category name ('01', '02', or '03')
        resize: Target size for images (H, W)

    Example:
        >>> dataset = BTADDataset(root='./btad', category='01')
        >>> train_imgs = dataset.get_train_data()
    """

    CATEGORIES = ['01', '02', '03']

    def __init__(
        self,
        root: str,
        category: str,
        resize: Optional[Tuple[int, int]] = None,
        load_masks: bool = False,
    ):
        super().__init__(root, category)

        if category not in self.CATEGORIES:
            raise ValueError(f"Invalid category. Choose from: {self.CATEGORIES}")

        self.resize = resize
        self.load_masks = bool(load_masks)  # BTAD does not ship masks; keep for API compatibility.
        self.category_path = self.root / category

    def _load_images(self, path: Path) -> List[NDArray]:
        """Load all images from a directory."""
        images = []

        for ext in ['*.png', '*.jpg', '*.bmp']:
            for img_path in sorted(path.glob(ext)):
                img = read_image(img_path, color="rgb")

                if self.resize is not None:
                    img = resize_image(img, self.resize)

                images.append(img)

        return images

    def get_train_data(self) -> NDArray:
        """Get training data (normal images only)."""
        train_path = self.category_path / 'train' / 'ok'

        if not train_path.exists():
            raise FileNotFoundError(f"Training data not found: {train_path}")

        images = self._load_images(train_path)
        return np.array(images)

    def get_test_data(self) -> Tuple[NDArray, NDArray, Optional[NDArray]]:
        """Get test data with labels."""
        test_path = self.category_path / 'test'

        if not test_path.exists():
            raise FileNotFoundError(f"Test data not found: {test_path}")

        images = []
        labels = []

        # Load normal test images
        normal_path = test_path / 'ok'
        if normal_path.exists():
            normal_imgs = self._load_images(normal_path)
            images.extend(normal_imgs)
            labels.extend([0] * len(normal_imgs))

        # Load anomaly test images
        defect_path = test_path / 'ko'
        if defect_path.exists():
            defect_imgs = self._load_images(defect_path)
            images.extend(defect_imgs)
            labels.extend([1] * len(defect_imgs))

        images = np.array(images)
        labels = np.array(labels)

        return images, labels, None

    def get_info(self) -> DatasetInfo:
        """Get dataset information."""
        train_data = self.get_train_data()
        test_data, test_labels, _ = self.get_test_data()

        return DatasetInfo(
            name='BTAD',
            categories=self.CATEGORIES,
            num_train=len(train_data),
            num_test=len(test_data),
            image_size=train_data[0].shape[:2],
            description=f'BTAD - Category {self.category}'
        )

    def get_train_paths(self) -> List[str]:
        train_path = self.category_path / 'train' / 'ok'
        if not train_path.exists():
            raise FileNotFoundError(f"Training data not found: {train_path}")
        paths: List[str] = []
        for ext in ['*.png', '*.jpg', '*.bmp']:
            paths.extend([str(p) for p in sorted(train_path.glob(ext))])
        if not paths:
            raise ValueError(f"No training images found in: {train_path}")
        return paths

    def get_test_paths(self) -> Tuple[List[str], NDArray, Optional[NDArray]]:
        test_path = self.category_path / 'test'
        if not test_path.exists():
            raise FileNotFoundError(f"Test data not found: {test_path}")

        ok_dir = test_path / 'ok'
        ko_dir = test_path / 'ko'

        ok_paths: List[str] = []
        ko_paths: List[str] = []
        for ext in ['*.png', '*.jpg', '*.bmp']:
            if ok_dir.exists():
                ok_paths.extend([str(p) for p in sorted(ok_dir.glob(ext))])
            if ko_dir.exists():
                ko_paths.extend([str(p) for p in sorted(ko_dir.glob(ext))])

        test_paths = ok_paths + ko_paths
        labels = np.array([0] * len(ok_paths) + [1] * len(ko_paths))
        return test_paths, labels, None


class VisADataset(BaseDataset):
    """VisA (Visual Anomaly) dataset loader.

    Supports a common folder-based layout (often named `visa_pytorch/`):

        root/
            visa_pytorch/
                <category>/
                    train/good/*.png
                    test/good/*.png
                    test/bad/*.png
                    ground_truth/bad/*.png   (optional)

    You may also pass `root` directly as the `visa_pytorch/` directory.
    If your VisA export uses a different structure, use `CustomDataset` or
    convert it into the layout above.
    """

    def __init__(
        self,
        root: str,
        category: str,
        resize: Optional[Tuple[int, int]] = None,
        load_masks: bool = True,
    ):
        super().__init__(root, category)
        self.resize = resize
        self.load_masks = load_masks

        base_root = self.root / "visa_pytorch" if (self.root / "visa_pytorch").exists() else self.root
        self.category_path = base_root / category

        if not self.category_path.exists():
            raise FileNotFoundError(f"VisA category path not found: {self.category_path}")

    def _load_images(self, path: Path) -> List[NDArray]:
        images: List[NDArray] = []

        for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
            for img_path in sorted(path.glob(ext)):
                try:
                    img = read_image(img_path, color="rgb")
                except FileNotFoundError:
                    continue

                if self.resize is not None:
                    img = resize_image(img, self.resize)

                images.append(img)

        return images

    @staticmethod
    def _resolve_dir(parent: Path, preferred: str, fallbacks: List[str]) -> Path:
        preferred_dir = parent / preferred
        if preferred_dir.exists():
            return preferred_dir
        for name in fallbacks:
            candidate = parent / name
            if candidate.exists():
                return candidate
        return preferred_dir

    @staticmethod
    def _scan_images(directory: Path) -> List[Path]:
        paths: List[Path] = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
            paths.extend(sorted(directory.glob(ext)))
        return paths

    def get_train_data(self) -> NDArray:
        train_dir = self.category_path / "train"
        good_dir = self._resolve_dir(train_dir, "good", ["ok", "normal"])

        if not good_dir.exists():
            raise FileNotFoundError(f"Training data not found: {good_dir}")

        images = self._load_images(good_dir)
        if not images:
            raise ValueError(f"No training images found in: {good_dir}")
        return np.array(images)

    def get_test_data(self) -> Tuple[NDArray, NDArray, Optional[NDArray]]:
        test_dir = self.category_path / "test"
        if not test_dir.exists():
            raise FileNotFoundError(f"Test data not found: {test_dir}")

        good_dir = self._resolve_dir(test_dir, "good", ["ok", "normal"])
        bad_dir = self._resolve_dir(test_dir, "bad", ["ko", "anomaly"])

        images: List[NDArray] = []
        labels: List[int] = []
        masks: Optional[List[NDArray]] = [] if self.load_masks else None

        # Normal images
        if good_dir.exists():
            good_imgs = self._load_images(good_dir)
            images.extend(good_imgs)
            labels.extend([0] * len(good_imgs))
            if self.load_masks:
                for img in good_imgs:
                    masks.append(np.zeros(img.shape[:2], dtype=np.uint8))

        # Anomalous images
        bad_img_paths = self._scan_images(bad_dir) if bad_dir.exists() else []
        bad_imgs: List[NDArray] = []
        for img_path in bad_img_paths:
            try:
                img = read_image(img_path, color="rgb")
            except FileNotFoundError:
                continue
            if self.resize is not None:
                img = resize_image(img, self.resize)
            bad_imgs.append(img)

        images.extend(bad_imgs)
        labels.extend([1] * len(bad_imgs))

        if self.load_masks:
            gt_dir = self.category_path / "ground_truth"
            gt_bad_dir = self._resolve_dir(gt_dir, "bad", ["ko", "anomaly"])
            for img_path, img in zip(bad_img_paths, bad_imgs):
                mask = None
                if gt_bad_dir.exists():
                    candidates = [
                        gt_bad_dir / img_path.name,
                        gt_bad_dir / f"{img_path.stem}.png",
                        gt_bad_dir / f"{img_path.stem}_mask.png",
                    ]
                    for candidate in candidates:
                        if candidate.exists():
                            try:
                                mask = read_image(candidate, color="gray")
                            except FileNotFoundError:
                                mask = None
                            else:
                                break
                if mask is None:
                    mask = np.zeros(img.shape[:2], dtype=np.uint8)
                elif self.resize is not None:
                    mask = resize_image(mask, self.resize, is_mask=True)
                masks.append((mask > 127).astype(np.uint8))

        return (
            np.array(images),
            np.array(labels),
            np.array(masks) if self.load_masks else None,
        )

    def get_info(self) -> DatasetInfo:
        train_data = self.get_train_data()
        test_data, _, _ = self.get_test_data()

        return DatasetInfo(
            name='VisA',
            categories=[self.category] if self.category else [],
            num_train=len(train_data),
            num_test=len(test_data),
            image_size=train_data[0].shape[:2],
            description=f'VisA - {self.category} category',
        )

    def get_train_paths(self) -> List[str]:
        train_dir = self.category_path / "train"
        good_dir = self._resolve_dir(train_dir, "good", ["ok", "normal"])
        if not good_dir.exists():
            raise FileNotFoundError(f"Training data not found: {good_dir}")
        paths = [str(p) for p in self._scan_images(good_dir)]
        if not paths:
            raise ValueError(f"No training images found in: {good_dir}")
        return paths

    def get_test_paths(self) -> Tuple[List[str], NDArray, Optional[NDArray]]:
        test_dir = self.category_path / "test"
        if not test_dir.exists():
            raise FileNotFoundError(f"Test data not found: {test_dir}")

        good_dir = self._resolve_dir(test_dir, "good", ["ok", "normal"])
        bad_dir = self._resolve_dir(test_dir, "bad", ["ko", "anomaly"])

        good_paths = self._scan_images(good_dir) if good_dir.exists() else []
        bad_paths = self._scan_images(bad_dir) if bad_dir.exists() else []

        test_paths = [str(p) for p in good_paths + bad_paths]
        labels = np.array([0] * len(good_paths) + [1] * len(bad_paths))

        if not self.load_masks:
            return test_paths, labels, None

        masks: List[NDArray] = []
        for _ in good_paths:
            shape = self.resize or (256, 256)
            masks.append(np.zeros(shape, dtype=np.uint8))

        gt_dir = self.category_path / "ground_truth"
        gt_bad_dir = self._resolve_dir(gt_dir, "bad", ["ko", "anomaly"])
        for img_path in bad_paths:
            mask = None
            if gt_bad_dir.exists():
                candidates = [
                    gt_bad_dir / img_path.name,
                    gt_bad_dir / f"{img_path.stem}.png",
                    gt_bad_dir / f"{img_path.stem}_mask.png",
                ]
                for candidate in candidates:
                    if candidate.exists():
                        try:
                            mask = read_image(candidate, color="gray")
                        except FileNotFoundError:
                            mask = None
                        else:
                            break
            if mask is None:
                shape = self.resize or (256, 256)
                mask = np.zeros(shape, dtype=np.uint8)
            elif self.resize is not None:
                mask = resize_image(mask, self.resize, is_mask=True)
            masks.append((mask > 127).astype(np.uint8))

        return test_paths, labels, np.array(masks)

    @staticmethod
    def list_categories(root: str) -> List[str]:
        root_path = Path(root)
        base_root = root_path / "visa_pytorch" if (root_path / "visa_pytorch").exists() else root_path
        if not base_root.exists():
            return []
        return sorted([p.name for p in base_root.iterdir() if p.is_dir()])


class CustomDataset(BaseDataset):
    """Custom dataset loader for user-defined datasets.

    Expected directory structure:
        root/
            train/
                normal/
                    img1.png
                    img2.png
                    ...
            test/
                normal/
                    img1.png
                    ...
                anomaly/
                    img1.png
                    ...
            (optional) ground_truth/
                anomaly/
                    img1_mask.png
                    ...

    Args:
        root: Path to dataset root
        resize: Target size for images (H, W)
        load_masks: Whether to load ground truth masks

    Example:
        >>> dataset = CustomDataset(root='./my_dataset')
        >>> train_imgs = dataset.get_train_data()
    """

    def __init__(
        self,
        root: str,
        resize: Optional[Tuple[int, int]] = None,
        load_masks: bool = False
    ):
        super().__init__(root)
        self.resize = resize
        self.load_masks = load_masks

    def validate_structure(self) -> None:
        """Validate the expected custom dataset folder structure.

        Expected layout:
            root/
              train/normal/*.png|*.jpg|*.jpeg|*.bmp
              test/normal/*.png|*.jpg|*.jpeg|*.bmp
              test/anomaly/*.png|*.jpg|*.jpeg|*.bmp
              ground_truth/anomaly/<stem>_mask.png   (required when load_masks=True)
        """

        def _scan_images(directory: Path) -> List[Path]:
            if not directory.exists():
                return []
            paths: List[Path] = []
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
                paths.extend(sorted(directory.glob(ext)))
            return paths

        errors: List[str] = []

        train_dir = self.root / "train" / "normal"
        test_normal_dir = self.root / "test" / "normal"
        test_anomaly_dir = self.root / "test" / "anomaly"

        if not train_dir.exists():
            errors.append(f"Missing directory: {train_dir}")
        if not test_normal_dir.exists():
            errors.append(f"Missing directory: {test_normal_dir}")
        if not test_anomaly_dir.exists():
            errors.append(f"Missing directory: {test_anomaly_dir}")

        train_imgs = _scan_images(train_dir)
        test_normal_imgs = _scan_images(test_normal_dir)
        test_anomaly_imgs = _scan_images(test_anomaly_dir)

        if train_dir.exists() and not train_imgs:
            errors.append(f"No training images found in: {train_dir}")
        if test_normal_dir.exists() and not test_normal_imgs:
            errors.append(f"No normal test images found in: {test_normal_dir}")
        if test_anomaly_dir.exists() and not test_anomaly_imgs:
            errors.append(f"No anomaly test images found in: {test_anomaly_dir}")

        if self.load_masks:
            gt_dir = self.root / "ground_truth" / "anomaly"
            if not gt_dir.exists():
                errors.append(
                    f"Missing directory: {gt_dir} (required when load_masks=True)"
                )
            else:
                missing_masks: List[str] = []
                for img_path in test_anomaly_imgs:
                    mask_path = gt_dir / f"{img_path.stem}_mask.png"
                    if not mask_path.exists():
                        missing_masks.append(str(mask_path))
                        if len(missing_masks) >= 3:
                            break
                if missing_masks:
                    preview = ", ".join(missing_masks)
                    errors.append(
                        "Missing ground-truth masks for anomaly images (expected '<stem>_mask.png'). "
                        f"Examples: {preview}"
                    )

        if errors:
            details = "\n- ".join(errors)
            raise ValueError(f"Invalid custom dataset structure:\n- {details}")

    def _load_images(self, path: Path) -> List[NDArray]:
        """Load all images from a directory."""
        images = []

        for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
            for img_path in sorted(path.glob(ext)):
                img = read_image(img_path, color="rgb")

                if self.resize is not None:
                    img = resize_image(img, self.resize)

                images.append(img)

        return images

    def get_train_data(self) -> NDArray:
        """Get training data."""
        train_path = self.root / 'train' / 'normal'

        if not train_path.exists():
            raise FileNotFoundError(f"Training data not found: {train_path}")

        images = self._load_images(train_path)
        return np.array(images)

    def get_test_data(self) -> Tuple[NDArray, NDArray, Optional[NDArray]]:
        """Get test data with labels and optionally masks."""
        test_path = self.root / 'test'

        if not test_path.exists():
            raise FileNotFoundError(f"Test data not found: {test_path}")

        images = []
        labels = []
        masks = [] if self.load_masks else None

        # Load normal test images
        normal_path = test_path / 'normal'
        if normal_path.exists():
            normal_imgs = self._load_images(normal_path)
            images.extend(normal_imgs)
            labels.extend([0] * len(normal_imgs))

            if self.load_masks:
                for img in normal_imgs:
                    masks.append(np.zeros(img.shape[:2], dtype=np.uint8))

        # Load anomaly test images
        anomaly_path = test_path / 'anomaly'
        if anomaly_path.exists():
            anomaly_imgs = self._load_images(anomaly_path)
            images.extend(anomaly_imgs)
            labels.extend([1] * len(anomaly_imgs))

            # Load masks if requested
            if self.load_masks:
                gt_path = self.root / 'ground_truth' / 'anomaly'
                if gt_path.exists():
                    for img_path in sorted(anomaly_path.glob('*.png')):
                        mask_path = gt_path / f"{img_path.stem}_mask.png"
                        if mask_path.exists():
                            mask = read_image(mask_path, color="gray")
                            if self.resize is not None:
                                mask = resize_image(mask, self.resize, is_mask=True)
                            mask = (mask > 127).astype(np.uint8)
                            masks.append(mask)
                        else:
                            masks.append(np.zeros(self.resize or anomaly_imgs[0].shape[:2], dtype=np.uint8))
                else:
                    # No masks available
                    for img in anomaly_imgs:
                        masks.append(np.zeros(img.shape[:2], dtype=np.uint8))

        images = np.array(images)
        labels = np.array(labels)
        masks = np.array(masks) if self.load_masks else None

        return images, labels, masks

    def get_info(self) -> DatasetInfo:
        """Get dataset information."""
        train_data = self.get_train_data()
        test_data, test_labels, _ = self.get_test_data()

        return DatasetInfo(
            name='Custom Dataset',
            categories=['custom'],
            num_train=len(train_data),
            num_test=len(test_data),
            image_size=train_data[0].shape[:2],
            description='User-defined custom dataset'
        )

    def get_train_paths(self) -> List[str]:
        train_path = self.root / 'train' / 'normal'
        if not train_path.exists():
            raise FileNotFoundError(f"Training data not found: {train_path}")
        paths: List[str] = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
            paths.extend([str(p) for p in sorted(train_path.glob(ext))])
        if not paths:
            raise ValueError(f"No training images found in: {train_path}")
        return paths

    def get_test_paths(self) -> Tuple[List[str], NDArray, Optional[NDArray]]:
        test_path = self.root / 'test'
        if not test_path.exists():
            raise FileNotFoundError(f"Test data not found: {test_path}")

        normal_dir = test_path / 'normal'
        anomaly_dir = test_path / 'anomaly'

        normal_paths: List[Path] = []
        anomaly_paths: List[Path] = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
            if normal_dir.exists():
                normal_paths.extend(sorted(normal_dir.glob(ext)))
            if anomaly_dir.exists():
                anomaly_paths.extend(sorted(anomaly_dir.glob(ext)))

        test_paths = [str(p) for p in normal_paths + anomaly_paths]
        labels = np.array([0] * len(normal_paths) + [1] * len(anomaly_paths))

        if not self.load_masks:
            return test_paths, labels, None

        masks: List[NDArray] = []
        for img_path in normal_paths:
            shape = self.resize
            if shape is None:
                try:
                    img = read_image(img_path, color="bgr")
                except FileNotFoundError:
                    img = None
                shape = img.shape[:2] if img is not None else (256, 256)
            masks.append(np.zeros(shape, dtype=np.uint8))

        gt_dir = self.root / 'ground_truth' / 'anomaly'
        for img_path in anomaly_paths:
            mask = None
            mask_path = gt_dir / f"{img_path.stem}_mask.png"
            if mask_path.exists():
                try:
                    mask = read_image(mask_path, color="gray")
                except FileNotFoundError:
                    mask = None
            if mask is None:
                shape = self.resize
                if shape is None:
                    try:
                        img = read_image(img_path, color="bgr")
                    except FileNotFoundError:
                        img = None
                    shape = img.shape[:2] if img is not None else (256, 256)
                mask = np.zeros(shape, dtype=np.uint8)
            elif self.resize is not None:
                mask = resize_image(mask, self.resize, is_mask=True)
            masks.append((mask > 127).astype(np.uint8))

        return test_paths, labels, np.array(masks)


def load_dataset(
    name: str,
    root: str,
    category: Optional[str] = None,
    **kwargs
) -> BaseDataset:
    """Factory function to load datasets.

    Args:
        name: Dataset name ('mvtec', 'mvtec_loco', 'mvtec_ad2', 'btad', 'visa', 'custom')
        root: Path to dataset root
        category: Category name (required for mvtec and btad)
        **kwargs: Additional arguments for dataset

    Returns:
        Dataset instance

    Example:
        >>> dataset = load_dataset('mvtec', './mvtec_ad', category='bottle')
        >>> train_data = dataset.get_train_data()
    """
    datasets = {
        'mvtec': MVTecDataset,
        'mvtec_ad': MVTecDataset,
        'mvtec_loco': MVTecLOCODataset,
        'mvtec_ad2': MVTecAD2Dataset,
        'btad': BTADDataset,
        'visa': VisADataset,
        'custom': CustomDataset,
    }

    name_lower = name.lower()
    if name_lower not in datasets:
        raise ValueError(f"Unknown dataset: {name}. Choose from: {list(datasets.keys())}")

    dataset_class = datasets[name_lower]

    if name_lower in ['mvtec', 'mvtec_ad', 'mvtec_loco', 'mvtec_ad2', 'btad', 'visa']:
        if category is None:
            raise ValueError(f"Category is required for {name} dataset")
        return dataset_class(root=root, category=category, **kwargs)
    else:
        return dataset_class(root=root, **kwargs)
