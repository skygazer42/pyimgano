"""
Security and Robustness

Features:
- Input validation
- Path traversal prevention
- Resource limit enforcement
- Error handling and recovery
- Secure temporary file handling
- Format validation
- Hash verification
"""

from typing import Optional, Tuple, List, Any, Dict
from pathlib import Path
import hashlib
import tempfile
import shutil
import os
from enum import Enum


class ErrorCode(Enum):
    """Error codes for common issues."""
    SUCCESS = 0
    INVALID_INPUT = 1
    FILE_NOT_FOUND = 2
    PERMISSION_DENIED = 3
    RESOURCE_LIMIT_EXCEEDED = 4
    FORMAT_ERROR = 5
    SECURITY_VIOLATION = 6
    UNKNOWN_ERROR = 99


class SecurityValidator:
    """Security validation utilities."""

    # Maximum safe file sizes
    MAX_IMAGE_SIZE = 100 * 1024 * 1024  # 100 MB
    MAX_VIDEO_SIZE = 1024 * 1024 * 1024  # 1 GB
    MAX_PIXELS = 178 * 1024 * 1024  # 178 megapixels (prevent decompression bombs)

    # Allowed file extensions
    ALLOWED_IMAGE_EXTENSIONS = {
        '.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif', '.webp'
    }
    ALLOWED_VIDEO_EXTENSIONS = {
        '.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'
    }

    @staticmethod
    def validate_path(path: str, base_dir: Optional[str] = None) -> Tuple[bool, str]:
        """
        Validate path for security issues.

        Parameters
        ----------
        path : str
            Path to validate
        base_dir : str, optional
            Base directory to restrict access to

        Returns
        -------
        valid : bool
            Whether path is valid
        error : str
            Error message if invalid
        """
        try:
            path = Path(path).resolve()

            # Check for path traversal
            if base_dir:
                base_dir = Path(base_dir).resolve()
                if not str(path).startswith(str(base_dir)):
                    return False, "Path traversal attempt detected"

            # Check for suspicious patterns
            suspicious = ['..', '~', '$']
            path_str = str(path)
            for pattern in suspicious:
                if pattern in path_str:
                    return False, f"Suspicious pattern '{pattern}' in path"

            return True, ""

        except Exception as e:
            return False, f"Path validation error: {e}"

    @staticmethod
    def validate_file_size(
        file_path: str,
        max_size: Optional[int] = None,
        file_type: str = 'image'
    ) -> Tuple[bool, str]:
        """
        Validate file size.

        Parameters
        ----------
        file_path : str
            File path
        max_size : int, optional
            Maximum size in bytes
        file_type : str, default='image'
            File type: 'image', 'video'

        Returns
        -------
        valid : bool
            Whether size is valid
        error : str
            Error message if invalid
        """
        if max_size is None:
            if file_type == 'image':
                max_size = SecurityValidator.MAX_IMAGE_SIZE
            elif file_type == 'video':
                max_size = SecurityValidator.MAX_VIDEO_SIZE
            else:
                max_size = SecurityValidator.MAX_IMAGE_SIZE

        try:
            size = os.path.getsize(file_path)
            if size > max_size:
                return False, f"File size {size} exceeds maximum {max_size}"
            return True, ""
        except Exception as e:
            return False, f"Size validation error: {e}"

    @staticmethod
    def validate_image_dimensions(
        width: int,
        height: int,
        max_pixels: Optional[int] = None
    ) -> Tuple[bool, str]:
        """
        Validate image dimensions to prevent decompression bombs.

        Parameters
        ----------
        width : int
            Image width
        height : int
            Image height
        max_pixels : int, optional
            Maximum number of pixels

        Returns
        -------
        valid : bool
            Whether dimensions are valid
        error : str
            Error message if invalid
        """
        if max_pixels is None:
            max_pixels = SecurityValidator.MAX_PIXELS

        total_pixels = width * height
        if total_pixels > max_pixels:
            return False, f"Image dimensions {width}x{height} exceed maximum {max_pixels} pixels"

        return True, ""

    @staticmethod
    def validate_file_extension(
        file_path: str,
        allowed_extensions: Optional[List[str]] = None,
        file_type: str = 'image'
    ) -> Tuple[bool, str]:
        """
        Validate file extension.

        Parameters
        ----------
        file_path : str
            File path
        allowed_extensions : list, optional
            List of allowed extensions
        file_type : str, default='image'
            File type: 'image', 'video'

        Returns
        -------
        valid : bool
            Whether extension is valid
        error : str
            Error message if invalid
        """
        if allowed_extensions is None:
            if file_type == 'image':
                allowed_extensions = SecurityValidator.ALLOWED_IMAGE_EXTENSIONS
            elif file_type == 'video':
                allowed_extensions = SecurityValidator.ALLOWED_VIDEO_EXTENSIONS
            else:
                allowed_extensions = SecurityValidator.ALLOWED_IMAGE_EXTENSIONS

        ext = Path(file_path).suffix.lower()
        if ext not in allowed_extensions:
            return False, f"File extension '{ext}' not allowed"

        return True, ""


class InputValidator:
    """Input validation utilities."""

    @staticmethod
    def validate_bbox(
        bbox: Tuple[float, float, float, float],
        image_size: Tuple[int, int]
    ) -> Tuple[bool, str]:
        """
        Validate bounding box.

        Parameters
        ----------
        bbox : tuple
            Bounding box (x, y, width, height)
        image_size : tuple
            Image size (width, height)

        Returns
        -------
        valid : bool
            Whether bbox is valid
        error : str
            Error message if invalid
        """
        x, y, w, h = bbox
        img_w, img_h = image_size

        if w <= 0 or h <= 0:
            return False, "Bounding box dimensions must be positive"

        if x < 0 or y < 0:
            return False, "Bounding box coordinates cannot be negative"

        if x + w > img_w or y + h > img_h:
            return False, "Bounding box exceeds image boundaries"

        return True, ""

    @staticmethod
    def validate_range(
        value: float,
        min_val: float,
        max_val: float,
        name: str = "value"
    ) -> Tuple[bool, str]:
        """
        Validate value is in range.

        Parameters
        ----------
        value : float
            Value to validate
        min_val : float
            Minimum value
        max_val : float
            Maximum value
        name : str, default="value"
            Parameter name

        Returns
        -------
        valid : bool
            Whether value is valid
        error : str
            Error message if invalid
        """
        if not (min_val <= value <= max_val):
            return False, f"{name} must be in range [{min_val}, {max_val}], got {value}"

        return True, ""

    @staticmethod
    def validate_positive(value: float, name: str = "value") -> Tuple[bool, str]:
        """
        Validate value is positive.

        Parameters
        ----------
        value : float
            Value to validate
        name : str, default="value"
            Parameter name

        Returns
        -------
        valid : bool
            Whether value is valid
        error : str
            Error message if invalid
        """
        if value <= 0:
            return False, f"{name} must be positive, got {value}"

        return True, ""


class ResourceLimiter:
    """Resource limit enforcement."""

    @staticmethod
    def check_memory_usage(max_memory_mb: float = 1024.0) -> Tuple[bool, float]:
        """
        Check current memory usage.

        Parameters
        ----------
        max_memory_mb : float, default=1024.0
            Maximum memory in MB

        Returns
        -------
        ok : bool
            Whether memory usage is acceptable
        usage_mb : float
            Current memory usage in MB
        """
        try:
            import psutil
            process = psutil.Process()
            usage_mb = process.memory_info().rss / 1024 / 1024
            return usage_mb < max_memory_mb, usage_mb
        except ImportError:
            # psutil not available, can't check
            return True, 0.0

    @staticmethod
    def estimate_array_memory(shape: Tuple, dtype: str = 'float32') -> float:
        """
        Estimate memory required for array.

        Parameters
        ----------
        shape : tuple
            Array shape
        dtype : str, default='float32'
            Data type

        Returns
        -------
        memory_mb : float
            Estimated memory in MB
        """
        import numpy as np

        dtype_sizes = {
            'uint8': 1,
            'int8': 1,
            'uint16': 2,
            'int16': 2,
            'float16': 2,
            'uint32': 4,
            'int32': 4,
            'float32': 4,
            'float64': 8,
        }

        size_bytes = np.prod(shape) * dtype_sizes.get(dtype, 4)
        return size_bytes / 1024 / 1024


class SecureTempFile:
    """Secure temporary file handling."""

    def __init__(
        self,
        prefix: str = 'pyimgano_',
        suffix: str = '',
        dir: Optional[str] = None
    ):
        """
        Initialize secure temporary file.

        Parameters
        ----------
        prefix : str, default='pyimgano_'
            Filename prefix
        suffix : str, default=''
            Filename suffix
        dir : str, optional
            Temporary directory
        """
        self.prefix = prefix
        self.suffix = suffix
        self.dir = dir
        self.path = None
        self.fd = None

    def __enter__(self):
        """Create temporary file."""
        self.fd, self.path = tempfile.mkstemp(
            prefix=self.prefix,
            suffix=self.suffix,
            dir=self.dir
        )
        return self.path

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up temporary file."""
        if self.fd:
            os.close(self.fd)
        if self.path and os.path.exists(self.path):
            os.unlink(self.path)


class SecureTempDir:
    """Secure temporary directory handling."""

    def __init__(
        self,
        prefix: str = 'pyimgano_',
        dir: Optional[str] = None
    ):
        """
        Initialize secure temporary directory.

        Parameters
        ----------
        prefix : str, default='pyimgano_'
            Directory prefix
        dir : str, optional
            Parent directory
        """
        self.prefix = prefix
        self.dir = dir
        self.path = None

    def __enter__(self):
        """Create temporary directory."""
        self.path = tempfile.mkdtemp(prefix=self.prefix, dir=self.dir)
        return self.path

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up temporary directory."""
        if self.path and os.path.exists(self.path):
            shutil.rmtree(self.path, ignore_errors=True)


class FileHasher:
    """File integrity verification."""

    @staticmethod
    def compute_hash(
        file_path: str,
        algorithm: str = 'sha256',
        chunk_size: int = 8192
    ) -> str:
        """
        Compute file hash.

        Parameters
        ----------
        file_path : str
            File path
        algorithm : str, default='sha256'
            Hash algorithm: 'md5', 'sha1', 'sha256', 'sha512'
        chunk_size : int, default=8192
            Read chunk size

        Returns
        -------
        hash : str
            Hex digest of file hash
        """
        hasher = hashlib.new(algorithm)

        with open(file_path, 'rb') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                hasher.update(chunk)

        return hasher.hexdigest()

    @staticmethod
    def verify_hash(
        file_path: str,
        expected_hash: str,
        algorithm: str = 'sha256'
    ) -> bool:
        """
        Verify file hash.

        Parameters
        ----------
        file_path : str
            File path
        expected_hash : str
            Expected hash value
        algorithm : str, default='sha256'
            Hash algorithm

        Returns
        -------
        valid : bool
            Whether hash matches
        """
        actual_hash = FileHasher.compute_hash(file_path, algorithm)
        return actual_hash.lower() == expected_hash.lower()


class ErrorHandler:
    """Error handling utilities."""

    @staticmethod
    def safe_execute(
        func: callable,
        *args,
        default: Any = None,
        log_errors: bool = True,
        **kwargs
    ) -> Tuple[Any, Optional[Exception]]:
        """
        Safely execute function with error handling.

        Parameters
        ----------
        func : callable
            Function to execute
        *args, **kwargs
            Function arguments
        default : Any, default=None
            Default return value on error
        log_errors : bool, default=True
            Log errors

        Returns
        -------
        result : Any
            Function result or default
        error : Exception or None
            Exception if error occurred
        """
        try:
            result = func(*args, **kwargs)
            return result, None
        except Exception as e:
            if log_errors:
                print(f"Error in {func.__name__}: {e}")
            return default, e

    @staticmethod
    def retry_on_failure(
        func: callable,
        max_attempts: int = 3,
        delay: float = 1.0,
        backoff: float = 2.0,
        *args,
        **kwargs
    ) -> Any:
        """
        Retry function on failure.

        Parameters
        ----------
        func : callable
            Function to execute
        max_attempts : int, default=3
            Maximum retry attempts
        delay : float, default=1.0
            Initial delay in seconds
        backoff : float, default=2.0
            Backoff multiplier
        *args, **kwargs
            Function arguments

        Returns
        -------
        result : Any
            Function result

        Raises
        ------
        Exception
            If all attempts fail
        """
        import time

        last_error = None
        current_delay = delay

        for attempt in range(max_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < max_attempts - 1:
                    time.sleep(current_delay)
                    current_delay *= backoff

        raise last_error


# Convenience functions
def validate_image_file(
    file_path: str,
    base_dir: Optional[str] = None,
    max_size: Optional[int] = None
) -> Tuple[ErrorCode, str]:
    """
    Validate image file for security issues.

    Parameters
    ----------
    file_path : str
        Image file path
    base_dir : str, optional
        Base directory to restrict access
    max_size : int, optional
        Maximum file size in bytes

    Returns
    -------
    code : ErrorCode
        Error code
    message : str
        Error message (empty if valid)
    """
    # Validate path
    valid, msg = SecurityValidator.validate_path(file_path, base_dir)
    if not valid:
        return ErrorCode.SECURITY_VIOLATION, msg

    # Check file exists
    if not os.path.exists(file_path):
        return ErrorCode.FILE_NOT_FOUND, "File not found"

    # Validate extension
    valid, msg = SecurityValidator.validate_file_extension(file_path, file_type='image')
    if not valid:
        return ErrorCode.FORMAT_ERROR, msg

    # Validate size
    valid, msg = SecurityValidator.validate_file_size(file_path, max_size, 'image')
    if not valid:
        return ErrorCode.RESOURCE_LIMIT_EXCEEDED, msg

    return ErrorCode.SUCCESS, ""


def secure_load_image(file_path: str, **kwargs) -> Tuple[Optional[Any], ErrorCode]:
    """
    Securely load image with validation.

    Parameters
    ----------
    file_path : str
        Image file path
    **kwargs
        Additional validation parameters

    Returns
    -------
    image : ndarray or None
        Loaded image
    code : ErrorCode
        Error code
    """
    # Validate file
    code, msg = validate_image_file(file_path, **kwargs)
    if code != ErrorCode.SUCCESS:
        print(f"Validation failed: {msg}")
        return None, code

    # Load image
    try:
        import cv2
        image = cv2.imread(file_path)
        if image is None:
            return None, ErrorCode.FORMAT_ERROR

        # Validate dimensions
        h, w = image.shape[:2]
        valid, msg = SecurityValidator.validate_image_dimensions(w, h)
        if not valid:
            print(f"Dimension validation failed: {msg}")
            return None, ErrorCode.RESOURCE_LIMIT_EXCEEDED

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, ErrorCode.SUCCESS

    except Exception as e:
        print(f"Error loading image: {e}")
        return None, ErrorCode.UNKNOWN_ERROR
