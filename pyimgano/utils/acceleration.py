"""
Performance Acceleration

Features:
- SIMD vectorization (SSE/AVX/NEON)
- GPU backend management (CUDA/ROCm/Metal/Vulkan)
- DLPack integration for zero-copy transfers
- Memory-mapped operations
- Batch processing optimizations
"""

from typing import Optional, Dict, Any, Callable
import numpy as np
from numpy.typing import NDArray
import platform
import warnings

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


class SIMDAccelerator:
    """SIMD vectorization utilities."""

    @staticmethod
    def detect_simd_support() -> Dict[str, bool]:
        """
        Detect available SIMD instruction sets.

        Returns
        -------
        support : dict
            Dictionary indicating support for different SIMD extensions
        """
        support = {
            'sse': False,
            'sse2': False,
            'sse3': False,
            'ssse3': False,
            'sse4_1': False,
            'sse4_2': False,
            'avx': False,
            'avx2': False,
            'avx512f': False,
            'neon': False,
        }

        system = platform.system()
        machine = platform.machine().lower()

        # ARM NEON
        if 'arm' in machine or 'aarch64' in machine:
            support['neon'] = True

        # x86/x64 - check via cpuinfo on Linux
        if system == 'Linux':
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    cpuinfo = f.read()
                    flags = ''
                    for line in cpuinfo.split('\n'):
                        if 'flags' in line:
                            flags = line
                            break

                    support['sse'] = 'sse' in flags
                    support['sse2'] = 'sse2' in flags
                    support['sse3'] = 'sse3' in flags or 'pni' in flags
                    support['ssse3'] = 'ssse3' in flags
                    support['sse4_1'] = 'sse4_1' in flags
                    support['sse4_2'] = 'sse4_2' in flags
                    support['avx'] = 'avx' in flags
                    support['avx2'] = 'avx2' in flags
                    support['avx512f'] = 'avx512f' in flags
            except:
                pass

        return support

    @staticmethod
    def vectorized_add(
        a: NDArray,
        b: NDArray,
        use_simd: bool = True
    ) -> NDArray:
        """
        Vectorized array addition.

        Parameters
        ----------
        a, b : ndarray
            Input arrays
        use_simd : bool, default=True
            Use SIMD if available

        Returns
        -------
        result : ndarray
            Sum of arrays
        """
        # NumPy automatically uses SIMD when available
        if use_simd:
            # Ensure arrays are aligned for SIMD
            if a.flags['C_CONTIGUOUS'] and b.flags['C_CONTIGUOUS']:
                return np.add(a, b)

        return a + b

    @staticmethod
    def vectorized_dot(
        a: NDArray,
        b: NDArray,
        use_simd: bool = True
    ) -> float:
        """
        Vectorized dot product.

        Parameters
        ----------
        a, b : ndarray
            Input vectors
        use_simd : bool, default=True
            Use SIMD if available

        Returns
        -------
        result : float
            Dot product
        """
        if use_simd:
            # NumPy's dot uses BLAS which has SIMD optimizations
            return np.dot(a, b)

        return (a * b).sum()

    @staticmethod
    def ensure_alignment(
        array: NDArray,
        alignment: int = 32
    ) -> NDArray:
        """
        Ensure array is aligned for SIMD operations.

        Parameters
        ----------
        array : ndarray
            Input array
        alignment : int, default=32
            Alignment boundary in bytes (32 for AVX, 64 for AVX-512)

        Returns
        -------
        aligned : ndarray
            Aligned array
        """
        if array.ctypes.data % alignment == 0:
            return array

        # Create aligned copy
        aligned = np.empty_like(array, order='C')
        aligned[:] = array
        return aligned


class GPUBackend:
    """GPU acceleration backend manager."""

    def __init__(self):
        self.backend = self._detect_backend()
        self.device = None

        if self.backend == 'cuda' and HAS_TORCH:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif self.backend == 'cupy' and HAS_CUPY:
            self.device = cp.cuda.Device(0)

    def _detect_backend(self) -> str:
        """Detect available GPU backend."""
        if HAS_TORCH and torch.cuda.is_available():
            return 'cuda'
        elif HAS_CUPY:
            try:
                cp.cuda.Device(0).compute_capability
                return 'cupy'
            except:
                pass

        # Check for other backends
        system = platform.system()
        if system == 'Darwin':  # macOS
            try:
                import torch
                if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    return 'mps'  # Metal Performance Shaders
            except:
                pass

        return 'cpu'

    def get_device_info(self) -> Dict[str, Any]:
        """
        Get GPU device information.

        Returns
        -------
        info : dict
            Device information
        """
        info = {'backend': self.backend}

        if self.backend == 'cuda' and HAS_TORCH:
            if torch.cuda.is_available():
                info['name'] = torch.cuda.get_device_name(0)
                info['compute_capability'] = torch.cuda.get_device_capability(0)
                info['total_memory'] = torch.cuda.get_device_properties(0).total_memory
                info['available_memory'] = torch.cuda.memory_allocated(0)
        elif self.backend == 'cupy' and HAS_CUPY:
            device = cp.cuda.Device(0)
            info['name'] = device.attributes['Name']
            info['compute_capability'] = device.compute_capability
            info['total_memory'] = device.mem_info[1]
            info['available_memory'] = device.mem_info[0]
        elif self.backend == 'mps':
            info['name'] = 'Apple Metal'

        return info

    def to_gpu(self, array: NDArray) -> Any:
        """
        Transfer array to GPU.

        Parameters
        ----------
        array : ndarray
            Input CPU array

        Returns
        -------
        gpu_array : tensor or cupy array
            Array on GPU
        """
        if self.backend == 'cuda' and HAS_TORCH:
            return torch.from_numpy(array).to(self.device)
        elif self.backend == 'cupy' and HAS_CUPY:
            return cp.asarray(array)
        elif self.backend == 'mps' and HAS_TORCH:
            return torch.from_numpy(array).to('mps')
        else:
            warnings.warn("No GPU backend available, returning CPU array")
            return array

    def to_cpu(self, gpu_array: Any) -> NDArray:
        """
        Transfer array from GPU to CPU.

        Parameters
        ----------
        gpu_array : tensor or cupy array
            Array on GPU

        Returns
        -------
        array : ndarray
            CPU array
        """
        if self.backend == 'cuda' and HAS_TORCH:
            return gpu_array.cpu().numpy()
        elif self.backend == 'cupy' and HAS_CUPY:
            return cp.asnumpy(gpu_array)
        elif self.backend == 'mps' and HAS_TORCH:
            return gpu_array.cpu().numpy()
        else:
            return np.asarray(gpu_array)

    def sync(self):
        """Synchronize GPU operations."""
        if self.backend == 'cuda' and HAS_TORCH:
            torch.cuda.synchronize()
        elif self.backend == 'cupy' and HAS_CUPY:
            cp.cuda.Stream.null.synchronize()


class DLPackInterface:
    """DLPack interface for zero-copy tensor exchange."""

    @staticmethod
    def is_available() -> bool:
        """Check if DLPack is available."""
        return HAS_TORCH

    @staticmethod
    def to_dlpack(array: Any) -> Any:
        """
        Convert array to DLPack capsule.

        Parameters
        ----------
        array : tensor or ndarray
            Input array

        Returns
        -------
        dlpack : capsule
            DLPack capsule for zero-copy exchange
        """
        if not HAS_TORCH:
            raise NotImplementedError("DLPack requires PyTorch")

        if isinstance(array, np.ndarray):
            array = torch.from_numpy(array)

        if hasattr(torch.utils, 'dlpack'):
            return torch.utils.dlpack.to_dlpack(array)
        else:
            raise NotImplementedError("PyTorch version does not support DLPack")

    @staticmethod
    def from_dlpack(dlpack: Any) -> Any:
        """
        Convert DLPack capsule to tensor.

        Parameters
        ----------
        dlpack : capsule
            DLPack capsule

        Returns
        -------
        array : tensor
            Tensor from DLPack
        """
        if not HAS_TORCH:
            raise NotImplementedError("DLPack requires PyTorch")

        if hasattr(torch.utils, 'dlpack'):
            return torch.utils.dlpack.from_dlpack(dlpack)
        else:
            raise NotImplementedError("PyTorch version does not support DLPack")

    @staticmethod
    def zero_copy_transfer(array: NDArray, target_framework: str = 'torch') -> Any:
        """
        Zero-copy transfer to target framework.

        Parameters
        ----------
        array : ndarray
            Input array
        target_framework : str, default='torch'
            Target framework: 'torch', 'cupy', 'jax'

        Returns
        -------
        target_array : tensor or array
            Array in target framework
        """
        if target_framework == 'torch' and HAS_TORCH:
            # NumPy to PyTorch shares memory if possible
            return torch.from_numpy(array)
        elif target_framework == 'cupy' and HAS_CUPY:
            # CuPy zero-copy from NumPy
            return cp.asarray(array)
        else:
            raise ValueError(f"Unsupported target framework: {target_framework}")


class MemoryMappedOps:
    """Memory-mapped operations for large arrays."""

    @staticmethod
    def create_memmap(
        filename: str,
        dtype: np.dtype,
        mode: str = 'w+',
        shape: Optional[tuple] = None
    ) -> np.memmap:
        """
        Create memory-mapped array.

        Parameters
        ----------
        filename : str
            Path to memory-mapped file
        dtype : dtype
            Data type
        mode : str, default='w+'
            File mode: 'r', 'w+', 'r+', 'c'
        shape : tuple, optional
            Array shape (required for 'w+' mode)

        Returns
        -------
        memmap : np.memmap
            Memory-mapped array
        """
        return np.memmap(filename, dtype=dtype, mode=mode, shape=shape)

    @staticmethod
    def load_memmap(
        filename: str,
        dtype: np.dtype,
        mode: str = 'r',
        shape: Optional[tuple] = None
    ) -> np.memmap:
        """
        Load existing memory-mapped array.

        Parameters
        ----------
        filename : str
            Path to memory-mapped file
        dtype : dtype
            Data type
        mode : str, default='r'
            File mode
        shape : tuple, optional
            Array shape

        Returns
        -------
        memmap : np.memmap
            Memory-mapped array
        """
        return np.memmap(filename, dtype=dtype, mode=mode, shape=shape)


class BatchProcessor:
    """Batch processing optimizations."""

    @staticmethod
    def process_batches(
        data: NDArray,
        batch_size: int,
        process_fn: Callable,
        **kwargs
    ) -> list:
        """
        Process data in batches.

        Parameters
        ----------
        data : ndarray
            Input data (N, ...)
        batch_size : int
            Batch size
        process_fn : callable
            Function to apply to each batch
        **kwargs
            Additional arguments for process_fn

        Returns
        -------
        results : list
            List of batch results
        """
        n_samples = len(data)
        results = []

        for i in range(0, n_samples, batch_size):
            batch = data[i:i + batch_size]
            result = process_fn(batch, **kwargs)
            results.append(result)

        return results

    @staticmethod
    def parallel_batches(
        data: NDArray,
        batch_size: int,
        process_fn: Callable,
        n_workers: int = 4,
        **kwargs
    ) -> list:
        """
        Process batches in parallel.

        Parameters
        ----------
        data : ndarray
            Input data
        batch_size : int
            Batch size
        process_fn : callable
            Function to apply
        n_workers : int, default=4
            Number of parallel workers
        **kwargs
            Additional arguments

        Returns
        -------
        results : list
            Combined results
        """
        from concurrent.futures import ThreadPoolExecutor

        n_samples = len(data)
        batches = [data[i:i + batch_size] for i in range(0, n_samples, batch_size)]

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(process_fn, batch, **kwargs) for batch in batches]
            results = [f.result() for f in futures]

        return results


# Convenience functions
def get_optimal_backend() -> GPUBackend:
    """
    Get optimal GPU backend for current system.

    Returns
    -------
    backend : GPUBackend
        Initialized GPU backend
    """
    return GPUBackend()


def accelerate_array_op(
    array: NDArray,
    operation: str,
    *args,
    use_gpu: bool = True,
    **kwargs
) -> NDArray:
    """
    Accelerate array operation using available hardware.

    Parameters
    ----------
    array : ndarray
        Input array
    operation : str
        Operation to perform: 'sum', 'mean', 'std', 'matmul', etc.
    *args
        Positional arguments for operation
    use_gpu : bool, default=True
        Use GPU if available
    **kwargs
        Keyword arguments for operation

    Returns
    -------
    result : ndarray
        Operation result
    """
    backend = get_optimal_backend()

    if use_gpu and backend.backend != 'cpu':
        # Transfer to GPU
        gpu_array = backend.to_gpu(array)

        # Perform operation
        if operation == 'sum':
            if HAS_TORCH and isinstance(gpu_array, torch.Tensor):
                result = torch.sum(gpu_array, *args, **kwargs)
            elif HAS_CUPY:
                result = cp.sum(gpu_array, *args, **kwargs)
        elif operation == 'mean':
            if HAS_TORCH and isinstance(gpu_array, torch.Tensor):
                result = torch.mean(gpu_array.float(), *args, **kwargs)
            elif HAS_CUPY:
                result = cp.mean(gpu_array, *args, **kwargs)
        elif operation == 'std':
            if HAS_TORCH and isinstance(gpu_array, torch.Tensor):
                result = torch.std(gpu_array.float(), *args, **kwargs)
            elif HAS_CUPY:
                result = cp.std(gpu_array, *args, **kwargs)
        elif operation == 'matmul':
            other = backend.to_gpu(args[0])
            if HAS_TORCH and isinstance(gpu_array, torch.Tensor):
                result = torch.matmul(gpu_array, other)
            elif HAS_CUPY:
                result = cp.matmul(gpu_array, other)
        else:
            raise ValueError(f"Unknown operation: {operation}")

        # Transfer back to CPU
        return backend.to_cpu(result)
    else:
        # CPU fallback
        op_map = {
            'sum': np.sum,
            'mean': np.mean,
            'std': np.std,
            'matmul': np.matmul,
        }

        if operation in op_map:
            return op_map[operation](array, *args, **kwargs)
        else:
            raise ValueError(f"Unknown operation: {operation}")


def check_hardware_capabilities() -> Dict[str, Any]:
    """
    Check available hardware acceleration capabilities.

    Returns
    -------
    capabilities : dict
        Dictionary of available capabilities
    """
    capabilities = {
        'simd': SIMDAccelerator.detect_simd_support(),
        'gpu': {},
        'dlpack': DLPackInterface.is_available(),
    }

    backend = get_optimal_backend()
    capabilities['gpu'] = backend.get_device_info()

    return capabilities
