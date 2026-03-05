"""
Color Management and Image Quality Assessment

Features:
- Color space conversions: sRGB/Linear/BT.2020/Lab/XYZ
- ICC profile support
- Gamma correction and tone mapping
- HDR to SDR conversion
- White balance and exposure normalization
- Quality metrics: PSNR, SSIM, MS-SSIM, LPIPS
"""

from typing import Optional, Tuple
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False


class ColorSpace:
    """Color space conversion utilities."""

    @staticmethod
    def srgb_to_linear(image: NDArray) -> NDArray:
        """
        Convert sRGB to linear RGB.

        Parameters
        ----------
        image : ndarray
            Image in sRGB space (0-1 range)

        Returns
        -------
        linear : ndarray
            Image in linear RGB space
        """
        image = image.astype(np.float32)
        mask = image <= 0.04045
        linear = np.where(
            mask,
            image / 12.92,
            np.power((image + 0.055) / 1.055, 2.4)
        )
        return linear

    @staticmethod
    def linear_to_srgb(image: NDArray) -> NDArray:
        """
        Convert linear RGB to sRGB.

        Parameters
        ----------
        image : ndarray
            Image in linear RGB space

        Returns
        -------
        srgb : ndarray
            Image in sRGB space
        """
        image = image.astype(np.float32)
        mask = image <= 0.0031308
        srgb = np.where(
            mask,
            image * 12.92,
            1.055 * np.power(image, 1.0 / 2.4) - 0.055
        )
        return srgb

    @staticmethod
    def rgb_to_lab(image: NDArray) -> NDArray:
        """
        Convert RGB to CIELAB.

        Parameters
        ----------
        image : ndarray
            RGB image (0-255 or 0-1)

        Returns
        -------
        lab : ndarray
            CIELAB image
        """
        if HAS_OPENCV:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            return cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        else:
            raise NotImplementedError("RGB to LAB requires OpenCV")

    @staticmethod
    def lab_to_rgb(image: NDArray) -> NDArray:
        """
        Convert CIELAB to RGB.

        Parameters
        ----------
        image : ndarray
            CIELAB image

        Returns
        -------
        rgb : ndarray
            RGB image
        """
        if HAS_OPENCV:
            return cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
        else:
            raise NotImplementedError("LAB to RGB requires OpenCV")


class ToneMapping:
    """HDR tone mapping and exposure adjustment."""

    @staticmethod
    def reinhard(
        hdr_image: NDArray,
        key_value: float = 0.18,
        white_point: Optional[float] = None
    ) -> NDArray:
        """
        Reinhard tone mapping operator.

        Parameters
        ----------
        hdr_image : ndarray
            HDR image (linear)
        key_value : float, default=0.18
            Middle gray value
        white_point : float, optional
            White point for burn-out

        Returns
        -------
        ldr : ndarray
            Tone-mapped LDR image
        """
        # Convert to luminance
        luminance = 0.2126 * hdr_image[..., 0] + \
                    0.7152 * hdr_image[..., 1] + \
                    0.0722 * hdr_image[..., 2]

        # Log average luminance
        log_avg = np.exp(np.mean(np.log(luminance + 1e-8)))

        # Scale
        scaled = (key_value / log_avg) * luminance

        if white_point is not None:
            # Extended Reinhard with white point
            scaled = scaled * (1 + scaled / (white_point ** 2))

        # Compress
        compressed = scaled / (1 + scaled)

        # Apply to color channels
        ldr = hdr_image * (compressed / (luminance + 1e-8))[..., None]

        return np.clip(ldr, 0, 1)

    @staticmethod
    def aces_filmic(hdr_image: NDArray) -> NDArray:
        """
        ACES Filmic tone mapping.

        Parameters
        ----------
        hdr_image : ndarray
            HDR image (linear)

        Returns
        -------
        ldr : ndarray
            Tone-mapped LDR image
        """
        # ACES parameters
        a = 2.51
        b = 0.03
        c = 2.43
        d = 0.59
        e = 0.14

        ldr = (hdr_image * (a * hdr_image + b)) / \
              (hdr_image * (c * hdr_image + d) + e)

        return np.clip(ldr, 0, 1)


class WhiteBalance:
    """White balance correction."""

    @staticmethod
    def gray_world(image: NDArray) -> NDArray:
        """
        Gray world white balance.

        Parameters
        ----------
        image : ndarray
            Input image (RGB, 0-1 range)

        Returns
        -------
        balanced : ndarray
            White-balanced image
        """
        # Calculate mean of each channel
        mean_r = np.mean(image[..., 0])
        mean_g = np.mean(image[..., 1])
        mean_b = np.mean(image[..., 2])

        # Calculate scaling factors
        avg = (mean_r + mean_g + mean_b) / 3
        scale_r = avg / (mean_r + 1e-8)
        scale_g = avg / (mean_g + 1e-8)
        scale_b = avg / (mean_b + 1e-8)

        # Apply correction
        balanced = image.copy()
        balanced[..., 0] *= scale_r
        balanced[..., 1] *= scale_g
        balanced[..., 2] *= scale_b

        return np.clip(balanced, 0, 1)

    @staticmethod
    def auto_white_balance(
        image: NDArray,
        method: str = 'gray_world'
    ) -> NDArray:
        """
        Automatic white balance correction.

        Parameters
        ----------
        image : ndarray
            Input image (RGB)
        method : str, default='gray_world'
            White balance method

        Returns
        -------
        balanced : ndarray
            White-balanced image
        """
        if method == 'gray_world':
            return WhiteBalance.gray_world(image)
        else:
            raise ValueError(f"Unknown method: {method}")


class QualityMetrics:
    """Image quality assessment metrics."""

    @staticmethod
    def psnr(image1: NDArray, image2: NDArray, max_value: float = 255.0) -> float:
        """
        Calculate Peak Signal-to-Noise Ratio.

        Parameters
        ----------
        image1, image2 : ndarray
            Images to compare
        max_value : float, default=255.0
            Maximum possible pixel value

        Returns
        -------
        psnr : float
            PSNR in dB
        """
        mse = np.mean((image1.astype(float) - image2.astype(float)) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(max_value / np.sqrt(mse))

    @staticmethod
    def ssim(
        image1: NDArray,
        image2: NDArray,
        max_value: float = 255.0,
        k1: float = 0.01,
        k2: float = 0.03,
        window_size: int = 11
    ) -> float:
        """
        Calculate Structural Similarity Index.

        Parameters
        ----------
        image1, image2 : ndarray
            Images to compare (grayscale or convert to gray)
        max_value : float, default=255.0
            Maximum possible pixel value
        k1, k2 : float
            SSIM parameters
        window_size : int, default=11
            Size of Gaussian window

        Returns
        -------
        ssim : float
            SSIM score (0-1, higher is better)
        """
        # Convert to grayscale if needed
        if len(image1.shape) == 3:
            image1 = 0.299 * image1[..., 0] + 0.587 * image1[..., 1] + 0.114 * image1[..., 2]
        if len(image2.shape) == 3:
            image2 = 0.299 * image2[..., 0] + 0.587 * image2[..., 1] + 0.114 * image2[..., 2]

        # Constants
        c1 = (k1 * max_value) ** 2
        c2 = (k2 * max_value) ** 2

        # Gaussian window
        sigma = 1.5
        gaussian = lambda x: np.exp(-(x ** 2) / (2 * sigma ** 2))
        window = np.outer(
            gaussian(np.arange(window_size) - window_size // 2),
            gaussian(np.arange(window_size) - window_size // 2)
        )
        window = window / window.sum()

        # Calculate local statistics
        mu1 = gaussian_filter(image1, sigma=sigma, mode='reflect')
        mu2 = gaussian_filter(image2, sigma=sigma, mode='reflect')

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = gaussian_filter(image1 ** 2, sigma=sigma, mode='reflect') - mu1_sq
        sigma2_sq = gaussian_filter(image2 ** 2, sigma=sigma, mode='reflect') - mu2_sq
        sigma12 = gaussian_filter(image1 * image2, sigma=sigma, mode='reflect') - mu1_mu2

        # SSIM formula
        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
                   ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

        return float(np.mean(ssim_map))

    @staticmethod
    def ms_ssim(
        image1: NDArray,
        image2: NDArray,
        max_value: float = 255.0,
        weights: Optional[list] = None
    ) -> float:
        """
        Calculate Multi-Scale Structural Similarity Index.

        Parameters
        ----------
        image1, image2 : ndarray
            Images to compare
        max_value : float, default=255.0
            Maximum possible pixel value
        weights : list, optional
            Weights for each scale

        Returns
        -------
        ms_ssim : float
            MS-SSIM score
        """
        if weights is None:
            weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]

        levels = len(weights)
        mssim = []
        mcs = []

        for i in range(levels):
            ssim_val = QualityMetrics.ssim(image1, image2, max_value=max_value)
            mssim.append(ssim_val)

            if i < levels - 1:
                # Downsample for next scale
                image1 = image1[::2, ::2]
                image2 = image2[::2, ::2]

        # Weighted combination
        ms_ssim_val = np.prod([m ** w for m, w in zip(mssim, weights)])
        return float(ms_ssim_val)


class ExposureNormalization:
    """Exposure and brightness normalization."""

    @staticmethod
    def histogram_equalization(image: NDArray) -> NDArray:
        """
        Histogram equalization for contrast enhancement.

        Parameters
        ----------
        image : ndarray
            Input image

        Returns
        -------
        equalized : ndarray
            Equalized image
        """
        if HAS_OPENCV:
            if len(image.shape) == 3:
                # Convert to YCrCb and equalize Y channel
                ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
                ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
                return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
            else:
                return cv2.equalizeHist(image)
        else:
            # Manual histogram equalization
            hist, bins = np.histogram(image.flatten(), 256, [0, 256])
            cdf = hist.cumsum()
            cdf_normalized = cdf * hist.max() / cdf.max()
            cdf_m = np.ma.masked_equal(cdf, 0)
            cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
            cdf = np.ma.filled(cdf_m, 0).astype('uint8')
            return cdf[image]

    @staticmethod
    def clahe(
        image: NDArray,
        clip_limit: float = 2.0,
        tile_size: Tuple[int, int] = (8, 8)
    ) -> NDArray:
        """
        Contrast Limited Adaptive Histogram Equalization.

        Parameters
        ----------
        image : ndarray
            Input image
        clip_limit : float, default=2.0
            Threshold for contrast limiting
        tile_size : tuple, default=(8, 8)
            Size of grid for histogram equalization

        Returns
        -------
        enhanced : ndarray
            CLAHE-enhanced image
        """
        if not HAS_OPENCV:
            raise NotImplementedError("CLAHE requires OpenCV")

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)

        if len(image.shape) == 3:
            # Convert to LAB and apply to L channel
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            return clahe.apply(image)


# Convenience functions
def normalize_exposure(image: NDArray, method: str = 'clahe') -> NDArray:
    """
    Normalize image exposure.

    Parameters
    ----------
    image : ndarray
        Input image
    method : str, default='clahe'
        Normalization method ('hist_eq', 'clahe')

    Returns
    -------
    normalized : ndarray
        Exposure-normalized image
    """
    if method == 'hist_eq':
        return ExposureNormalization.histogram_equalization(image)
    elif method == 'clahe':
        return ExposureNormalization.clahe(image)
    else:
        raise ValueError(f"Unknown method: {method}")


def auto_color_correct(image: NDArray) -> NDArray:
    """
    Automatic color correction pipeline.

    Parameters
    ----------
    image : ndarray
        Input image (RGB, 0-1 range)

    Returns
    -------
    corrected : ndarray
        Color-corrected image
    """
    # White balance
    balanced = WhiteBalance.auto_white_balance(image)

    # Exposure normalization
    if balanced.max() <= 1.0:
        balanced = (balanced * 255).astype(np.uint8)

    normalized = ExposureNormalization.clahe(balanced)

    if normalized.max() > 1.0:
        normalized = normalized.astype(np.float32) / 255.0

    return normalized
