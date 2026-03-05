"""
Geometry and Camera Models

Features:
- Camera calibration (intrinsic/extrinsic parameters)
- Lens distortion correction (radial/tangential)
- Homography transformations
- Perspective transformations
- ROI operations (crop, pad, resize)
- Coordinate transformations
"""

from typing import Optional, Tuple, List, Union
import numpy as np
from numpy.typing import NDArray

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False


class CameraCalibration:
    """Camera calibration utilities."""

    @staticmethod
    def calibrate_camera(
        object_points: List[NDArray],
        image_points: List[NDArray],
        image_size: Tuple[int, int],
        flags: int = 0
    ) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        """
        Calibrate camera from checkerboard patterns.

        Parameters
        ----------
        object_points : list of ndarray
            3D points in real world space for each calibration image
        image_points : list of ndarray
            2D points in image plane for each calibration image
        image_size : tuple
            Image size (width, height)
        flags : int, default=0
            Calibration flags (OpenCV cv2.CALIB_* constants)

        Returns
        -------
        camera_matrix : ndarray
            Intrinsic camera matrix (3x3)
        dist_coeffs : ndarray
            Distortion coefficients
        rvecs : list of ndarray
            Rotation vectors for each calibration image
        tvecs : list of ndarray
            Translation vectors for each calibration image
        """
        if not HAS_OPENCV:
            raise NotImplementedError("Camera calibration requires OpenCV")

        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            object_points,
            image_points,
            image_size,
            None,
            None,
            flags=flags
        )

        return camera_matrix, dist_coeffs, rvecs, tvecs

    @staticmethod
    def generate_checkerboard_points(
        board_size: Tuple[int, int],
        square_size: float = 1.0
    ) -> NDArray:
        """
        Generate 3D points for checkerboard pattern.

        Parameters
        ----------
        board_size : tuple
            Number of inner corners (width, height)
        square_size : float, default=1.0
            Size of each square in real-world units

        Returns
        -------
        points : ndarray
            3D object points (N, 3)
        """
        points = np.zeros((board_size[0] * board_size[1], 3), np.float32)
        points[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
        points *= square_size
        return points


class DistortionCorrection:
    """Lens distortion correction."""

    @staticmethod
    def undistort(
        image: NDArray,
        camera_matrix: NDArray,
        dist_coeffs: NDArray,
        new_camera_matrix: Optional[NDArray] = None
    ) -> NDArray:
        """
        Correct lens distortion.

        Parameters
        ----------
        image : ndarray
            Input distorted image
        camera_matrix : ndarray
            Camera intrinsic matrix (3x3)
        dist_coeffs : ndarray
            Distortion coefficients [k1, k2, p1, p2, k3, ...]
        new_camera_matrix : ndarray, optional
            New camera matrix for output image

        Returns
        -------
        undistorted : ndarray
            Undistorted image
        """
        if not HAS_OPENCV:
            raise NotImplementedError("Undistortion requires OpenCV")

        h, w = image.shape[:2]
        if new_camera_matrix is None:
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                camera_matrix, dist_coeffs, (w, h), 1, (w, h)
            )

        undistorted = cv2.undistort(
            image,
            camera_matrix,
            dist_coeffs,
            None,
            new_camera_matrix
        )

        return undistorted

    @staticmethod
    def get_undistort_maps(
        image_size: Tuple[int, int],
        camera_matrix: NDArray,
        dist_coeffs: NDArray,
        new_camera_matrix: Optional[NDArray] = None
    ) -> Tuple[NDArray, NDArray]:
        """
        Compute undistortion maps for faster processing.

        Parameters
        ----------
        image_size : tuple
            Image size (width, height)
        camera_matrix : ndarray
            Camera intrinsic matrix
        dist_coeffs : ndarray
            Distortion coefficients
        new_camera_matrix : ndarray, optional
            New camera matrix

        Returns
        -------
        map1, map2 : ndarray
            Remapping maps for cv2.remap()
        """
        if not HAS_OPENCV:
            raise NotImplementedError("Undistortion maps require OpenCV")

        if new_camera_matrix is None:
            new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
                camera_matrix, dist_coeffs, image_size, 1, image_size
            )

        map1, map2 = cv2.initUndistortRectifyMap(
            camera_matrix,
            dist_coeffs,
            None,
            new_camera_matrix,
            image_size,
            cv2.CV_32FC1
        )

        return map1, map2


class HomographyTransform:
    """Homography and perspective transformations."""

    @staticmethod
    def find_homography(
        src_points: NDArray,
        dst_points: NDArray,
        method: str = 'ransac',
        ransac_threshold: float = 3.0
    ) -> Tuple[NDArray, NDArray]:
        """
        Find homography matrix from point correspondences.

        Parameters
        ----------
        src_points : ndarray
            Source points (N, 2)
        dst_points : ndarray
            Destination points (N, 2)
        method : str, default='ransac'
            Method: 'ransac', 'lmeds', or 'rho'
        ransac_threshold : float, default=3.0
            RANSAC reprojection threshold

        Returns
        -------
        H : ndarray
            Homography matrix (3x3)
        mask : ndarray
            Inlier mask (N,)
        """
        if not HAS_OPENCV:
            # Simple DLT (Direct Linear Transform) implementation
            if len(src_points) < 4:
                raise ValueError("Need at least 4 point correspondences")

            H = HomographyTransform._compute_homography_dlt(src_points, dst_points)
            mask = np.ones(len(src_points), dtype=np.uint8)
            return H, mask

        method_map = {
            'ransac': cv2.RANSAC,
            'lmeds': cv2.LMEDS,
            'rho': cv2.RHO
        }

        H, mask = cv2.findHomography(
            src_points,
            dst_points,
            method_map.get(method, cv2.RANSAC),
            ransac_threshold
        )

        return H, mask

    @staticmethod
    def _compute_homography_dlt(
        src_points: NDArray,
        dst_points: NDArray
    ) -> NDArray:
        """Compute homography using Direct Linear Transform."""
        n = len(src_points)
        A = np.zeros((2 * n, 9))

        for i in range(n):
            x, y = src_points[i]
            u, v = dst_points[i]

            A[2*i] = [-x, -y, -1, 0, 0, 0, u*x, u*y, u]
            A[2*i+1] = [0, 0, 0, -x, -y, -1, v*x, v*y, v]

        # Solve Ah = 0 using SVD
        _, _, Vt = np.linalg.svd(A)
        H = Vt[-1].reshape(3, 3)
        H = H / H[2, 2]  # Normalize

        return H

    @staticmethod
    def warp_perspective(
        image: NDArray,
        H: NDArray,
        output_size: Tuple[int, int],
        interpolation: str = 'linear'
    ) -> NDArray:
        """
        Apply perspective transformation.

        Parameters
        ----------
        image : ndarray
            Input image
        H : ndarray
            Homography matrix (3x3)
        output_size : tuple
            Output image size (width, height)
        interpolation : str, default='linear'
            Interpolation method: 'nearest', 'linear', 'cubic'

        Returns
        -------
        warped : ndarray
            Warped image
        """
        if not HAS_OPENCV:
            raise NotImplementedError("Perspective warp requires OpenCV")

        interp_map = {
            'nearest': cv2.INTER_NEAREST,
            'linear': cv2.INTER_LINEAR,
            'cubic': cv2.INTER_CUBIC
        }

        warped = cv2.warpPerspective(
            image,
            H,
            output_size,
            flags=interp_map.get(interpolation, cv2.INTER_LINEAR)
        )

        return warped

    @staticmethod
    def four_point_transform(
        image: NDArray,
        pts: NDArray
    ) -> NDArray:
        """
        Perform bird's-eye view transform from four corner points.

        Parameters
        ----------
        image : ndarray
            Input image
        pts : ndarray
            Four corner points (4, 2) in order: top-left, top-right,
            bottom-right, bottom-left

        Returns
        -------
        warped : ndarray
            Bird's-eye view warped image
        """
        # Compute width and height of new image
        (tl, tr, br, bl) = pts

        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = max(int(heightA), int(heightB))

        # Destination points
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype=np.float32)

        # Compute homography
        H, _ = HomographyTransform.find_homography(pts, dst)

        # Warp
        warped = HomographyTransform.warp_perspective(
            image, H, (maxWidth, maxHeight)
        )

        return warped


class ROIOperations:
    """Region of Interest operations."""

    @staticmethod
    def crop(
        image: NDArray,
        x: int,
        y: int,
        width: int,
        height: int
    ) -> NDArray:
        """
        Crop region of interest.

        Parameters
        ----------
        image : ndarray
            Input image
        x, y : int
            Top-left corner coordinates
        width, height : int
            ROI dimensions

        Returns
        -------
        roi : ndarray
            Cropped region
        """
        h, w = image.shape[:2]

        # Clamp coordinates
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w, x + width)
        y2 = min(h, y + height)

        return image[y1:y2, x1:x2].copy()

    @staticmethod
    def crop_center(
        image: NDArray,
        crop_size: Tuple[int, int]
    ) -> NDArray:
        """
        Crop center region.

        Parameters
        ----------
        image : ndarray
            Input image
        crop_size : tuple
            Crop size (width, height)

        Returns
        -------
        cropped : ndarray
            Center-cropped image
        """
        h, w = image.shape[:2]
        crop_w, crop_h = crop_size

        x = (w - crop_w) // 2
        y = (h - crop_h) // 2

        return ROIOperations.crop(image, x, y, crop_w, crop_h)

    @staticmethod
    def pad(
        image: NDArray,
        padding: Union[int, Tuple[int, int, int, int]],
        mode: str = 'constant',
        value: Union[int, float, Tuple] = 0
    ) -> NDArray:
        """
        Pad image.

        Parameters
        ----------
        image : ndarray
            Input image
        padding : int or tuple
            Padding size. If int, pad all sides equally.
            If tuple, (top, bottom, left, right)
        mode : str, default='constant'
            Padding mode: 'constant', 'edge', 'reflect', 'symmetric'
        value : scalar or tuple, default=0
            Fill value for constant padding

        Returns
        -------
        padded : ndarray
            Padded image
        """
        if isinstance(padding, int):
            padding = (padding, padding, padding, padding)

        top, bottom, left, right = padding

        if mode == 'constant':
            if len(image.shape) == 3:
                pad_width = ((top, bottom), (left, right), (0, 0))
            else:
                pad_width = ((top, bottom), (left, right))

            if isinstance(value, tuple):
                # Multi-channel constant value
                padded = np.pad(image, pad_width, mode='constant')
                for c, v in enumerate(value):
                    # Fill border with channel-specific value
                    padded[:top, :, c] = v
                    padded[-bottom:, :, c] = v
                    padded[:, :left, c] = v
                    padded[:, -right:, c] = v
            else:
                padded = np.pad(image, pad_width, mode='constant', constant_values=value)
        else:
            if len(image.shape) == 3:
                pad_width = ((top, bottom), (left, right), (0, 0))
            else:
                pad_width = ((top, bottom), (left, right))

            padded = np.pad(image, pad_width, mode=mode)

        return padded

    @staticmethod
    def resize(
        image: NDArray,
        size: Tuple[int, int],
        interpolation: str = 'linear',
        keep_aspect_ratio: bool = False
    ) -> NDArray:
        """
        Resize image.

        Parameters
        ----------
        image : ndarray
            Input image
        size : tuple
            Target size (width, height)
        interpolation : str, default='linear'
            Interpolation: 'nearest', 'linear', 'cubic', 'area', 'lanczos'
        keep_aspect_ratio : bool, default=False
            If True, maintain aspect ratio and pad

        Returns
        -------
        resized : ndarray
            Resized image
        """
        if not HAS_OPENCV:
            raise NotImplementedError("Resize requires OpenCV")

        h, w = image.shape[:2]
        target_w, target_h = size

        if keep_aspect_ratio:
            # Compute scaling factor
            scale = min(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)

            # Resize
            interp_map = {
                'nearest': cv2.INTER_NEAREST,
                'linear': cv2.INTER_LINEAR,
                'cubic': cv2.INTER_CUBIC,
                'area': cv2.INTER_AREA,
                'lanczos': cv2.INTER_LANCZOS4
            }

            resized = cv2.resize(
                image,
                (new_w, new_h),
                interpolation=interp_map.get(interpolation, cv2.INTER_LINEAR)
            )

            # Pad to target size
            pad_w = target_w - new_w
            pad_h = target_h - new_h
            top = pad_h // 2
            bottom = pad_h - top
            left = pad_w // 2
            right = pad_w - left

            resized = ROIOperations.pad(resized, (top, bottom, left, right))
        else:
            interp_map = {
                'nearest': cv2.INTER_NEAREST,
                'linear': cv2.INTER_LINEAR,
                'cubic': cv2.INTER_CUBIC,
                'area': cv2.INTER_AREA,
                'lanczos': cv2.INTER_LANCZOS4
            }

            resized = cv2.resize(
                image,
                size,
                interpolation=interp_map.get(interpolation, cv2.INTER_LINEAR)
            )

        return resized


class CoordinateTransform:
    """Coordinate system transformations."""

    @staticmethod
    def pixel_to_normalized(
        points: NDArray,
        image_size: Tuple[int, int]
    ) -> NDArray:
        """
        Convert pixel coordinates to normalized coordinates [-1, 1].

        Parameters
        ----------
        points : ndarray
            Pixel coordinates (N, 2)
        image_size : tuple
            Image size (width, height)

        Returns
        -------
        normalized : ndarray
            Normalized coordinates (N, 2)
        """
        w, h = image_size
        normalized = points.copy().astype(np.float32)
        normalized[:, 0] = (points[:, 0] / w) * 2 - 1
        normalized[:, 1] = (points[:, 1] / h) * 2 - 1
        return normalized

    @staticmethod
    def normalized_to_pixel(
        points: NDArray,
        image_size: Tuple[int, int]
    ) -> NDArray:
        """
        Convert normalized coordinates to pixel coordinates.

        Parameters
        ----------
        points : ndarray
            Normalized coordinates (N, 2) in range [-1, 1]
        image_size : tuple
            Image size (width, height)

        Returns
        -------
        pixels : ndarray
            Pixel coordinates (N, 2)
        """
        w, h = image_size
        pixels = points.copy().astype(np.float32)
        pixels[:, 0] = (points[:, 0] + 1) * w / 2
        pixels[:, 1] = (points[:, 1] + 1) * h / 2
        return pixels.astype(np.int32)

    @staticmethod
    def apply_affine(
        points: NDArray,
        matrix: NDArray
    ) -> NDArray:
        """
        Apply affine transformation to points.

        Parameters
        ----------
        points : ndarray
            Input points (N, 2)
        matrix : ndarray
            Affine transformation matrix (2, 3)

        Returns
        -------
        transformed : ndarray
            Transformed points (N, 2)
        """
        # Add homogeneous coordinate
        ones = np.ones((len(points), 1))
        points_h = np.hstack([points, ones])

        # Apply transformation
        transformed = points_h @ matrix.T

        return transformed


# Convenience functions
def undistort_image(
    image: NDArray,
    camera_matrix: NDArray,
    dist_coeffs: NDArray
) -> NDArray:
    """
    Convenience function to undistort image.

    Parameters
    ----------
    image : ndarray
        Input distorted image
    camera_matrix : ndarray
        Camera intrinsic matrix
    dist_coeffs : ndarray
        Distortion coefficients

    Returns
    -------
    undistorted : ndarray
        Undistorted image
    """
    return DistortionCorrection.undistort(image, camera_matrix, dist_coeffs)


def crop_roi(
    image: NDArray,
    bbox: Tuple[int, int, int, int]
) -> NDArray:
    """
    Crop region of interest from bounding box.

    Parameters
    ----------
    image : ndarray
        Input image
    bbox : tuple
        Bounding box (x, y, width, height)

    Returns
    -------
    roi : ndarray
        Cropped region
    """
    x, y, w, h = bbox
    return ROIOperations.crop(image, x, y, w, h)


def resize_image(
    image: NDArray,
    size: Tuple[int, int],
    keep_aspect_ratio: bool = False
) -> NDArray:
    """
    Resize image to target size.

    Parameters
    ----------
    image : ndarray
        Input image
    size : tuple
        Target size (width, height)
    keep_aspect_ratio : bool, default=False
        Whether to maintain aspect ratio

    Returns
    -------
    resized : ndarray
        Resized image
    """
    return ROIOperations.resize(image, size, keep_aspect_ratio=keep_aspect_ratio)
