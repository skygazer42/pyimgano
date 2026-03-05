"""
Data Pipeline

Features:
- Dataset abstraction and loading
- Data prefetching and caching
- Parallel data loading
- Batching and collation
- Data augmentation pipelines
- Memory-efficient iterators
"""

from typing import Optional, Callable, List, Iterator, Any, Tuple, Union
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from concurrent.futures import ThreadPoolExecutor
import queue
import threading

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False


class Dataset:
    """Base dataset class."""

    def __init__(self):
        pass

    def __len__(self) -> int:
        """Return dataset size."""
        raise NotImplementedError

    def __getitem__(self, index: int) -> Any:
        """Get item by index."""
        raise NotImplementedError


class ImageDataset(Dataset):
    """Image dataset from directory or file list."""

    def __init__(
        self,
        image_paths: Union[List[Union[str, Path]], str, Path],
        transform: Optional[Callable] = None,
        load_to_memory: bool = False
    ):
        """
        Initialize image dataset.

        Parameters
        ----------
        image_paths : list or str or Path
            List of image paths, or directory path, or glob pattern
        transform : callable, optional
            Transformation function to apply to each image
        load_to_memory : bool, default=False
            Load all images to memory
        """
        super().__init__()

        # Process paths
        if isinstance(image_paths, (str, Path)):
            path = Path(image_paths)
            if path.is_dir():
                # Load all images from directory
                self.image_paths = sorted(path.glob('*.[jp][pn][g]'))
            else:
                # Glob pattern
                parent = path.parent
                self.image_paths = sorted(parent.glob(path.name))
        else:
            self.image_paths = [Path(p) for p in image_paths]

        self.transform = transform
        self.load_to_memory = load_to_memory
        self.cached_images = {}

        if load_to_memory:
            self._load_all_to_memory()

    def _load_all_to_memory(self):
        """Load all images to memory."""
        for i in range(len(self.image_paths)):
            self.cached_images[i] = self._load_image(i)

    def _load_image(self, index: int) -> NDArray:
        """Load single image."""
        path = self.image_paths[index]

        if HAS_OPENCV:
            img = cv2.imread(str(path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif HAS_PIL:
            img = np.array(Image.open(path))
        else:
            raise ImportError("Either OpenCV or PIL is required")

        return img

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Union[NDArray, Any]:
        if self.load_to_memory:
            img = self.cached_images[index]
        else:
            img = self._load_image(index)

        if self.transform:
            img = self.transform(img)

        return img


class DataLoader:
    """Data loader with prefetching and batching."""

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 0,
        prefetch_factor: int = 2,
        collate_fn: Optional[Callable] = None,
        drop_last: bool = False
    ):
        """
        Initialize data loader.

        Parameters
        ----------
        dataset : Dataset
            Dataset to load from
        batch_size : int, default=1
            Batch size
        shuffle : bool, default=False
            Shuffle data each epoch
        num_workers : int, default=0
            Number of worker threads for parallel loading
        prefetch_factor : int, default=2
            Number of batches to prefetch
        collate_fn : callable, optional
            Function to collate samples into batch
        drop_last : bool, default=False
            Drop last incomplete batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.collate_fn = collate_fn or self._default_collate
        self.drop_last = drop_last

        self.indices = None
        self._reset_indices()

    def _reset_indices(self):
        """Reset and optionally shuffle indices."""
        self.indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def _default_collate(self, batch: List[Any]) -> Any:
        """Default collation function."""
        if isinstance(batch[0], np.ndarray):
            return np.stack(batch)
        elif isinstance(batch[0], (int, float)):
            return np.array(batch)
        elif isinstance(batch[0], (list, tuple)):
            return [self._default_collate([item[i] for item in batch])
                    for i in range(len(batch[0]))]
        else:
            return batch

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator:
        self._reset_indices()

        if self.num_workers == 0:
            # Single-threaded
            return self._single_worker_iter()
        else:
            # Multi-threaded with prefetching
            return self._multi_worker_iter()

    def _single_worker_iter(self) -> Iterator:
        """Single-threaded iterator."""
        batch = []

        for idx in self.indices:
            item = self.dataset[idx]
            batch.append(item)

            if len(batch) == self.batch_size:
                yield self.collate_fn(batch)
                batch = []

        if batch and not self.drop_last:
            yield self.collate_fn(batch)

    def _multi_worker_iter(self) -> Iterator:
        """Multi-threaded iterator with prefetching."""
        prefetch_queue = queue.Queue(maxsize=self.prefetch_factor)
        stop_event = threading.Event()

        def worker():
            """Worker thread for loading data."""
            batch = []
            try:
                for idx in self.indices:
                    if stop_event.is_set():
                        break

                    item = self.dataset[idx]
                    batch.append(item)

                    if len(batch) == self.batch_size:
                        prefetch_queue.put(self.collate_fn(batch))
                        batch = []

                if batch and not self.drop_last:
                    prefetch_queue.put(self.collate_fn(batch))

            finally:
                prefetch_queue.put(None)  # Sentinel

        # Start worker thread
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        try:
            while True:
                batch = prefetch_queue.get()
                if batch is None:
                    break
                yield batch
        finally:
            stop_event.set()
            thread.join(timeout=1.0)


class DataCache:
    """LRU cache for dataset samples."""

    def __init__(self, max_size: int = 100):
        """
        Initialize cache.

        Parameters
        ----------
        max_size : int, default=100
            Maximum number of items to cache
        """
        self.max_size = max_size
        self.cache = {}
        self.access_order = []

    def get(self, key: Any) -> Optional[Any]:
        """Get item from cache."""
        if key in self.cache:
            # Update access order
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None

    def put(self, key: Any, value: Any):
        """Put item in cache."""
        if key in self.cache:
            # Update existing
            self.access_order.remove(key)
        elif len(self.cache) >= self.max_size:
            # Evict least recently used
            lru_key = self.access_order.pop(0)
            del self.cache[lru_key]

        self.cache[key] = value
        self.access_order.append(key)

    def clear(self):
        """Clear cache."""
        self.cache.clear()
        self.access_order.clear()


class CachedDataset(Dataset):
    """Dataset with LRU caching."""

    def __init__(
        self,
        base_dataset: Dataset,
        cache_size: int = 100
    ):
        """
        Initialize cached dataset.

        Parameters
        ----------
        base_dataset : Dataset
            Base dataset to wrap
        cache_size : int, default=100
            Cache size
        """
        super().__init__()
        self.base_dataset = base_dataset
        self.cache = DataCache(max_size=cache_size)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> Any:
        # Check cache
        item = self.cache.get(index)
        if item is not None:
            return item

        # Load from base dataset
        item = self.base_dataset[index]

        # Cache it
        self.cache.put(index, item)

        return item


class Transform:
    """Base transformation class."""

    def __call__(self, data: Any) -> Any:
        raise NotImplementedError


class Compose(Transform):
    """Compose multiple transformations."""

    def __init__(self, transforms: List[Transform]):
        self.transforms = transforms

    def __call__(self, data: Any) -> Any:
        for t in self.transforms:
            data = t(data)
        return data


class Normalize(Transform):
    """Normalize image."""

    def __init__(
        self,
        mean: Union[float, Tuple[float, ...]] = (0.485, 0.456, 0.406),
        std: Union[float, Tuple[float, ...]] = (0.229, 0.224, 0.225)
    ):
        self.mean = np.array(mean).reshape(1, 1, -1)
        self.std = np.array(std).reshape(1, 1, -1)

    def __call__(self, image: NDArray) -> NDArray:
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        return image


class Resize(Transform):
    """Resize image."""

    def __init__(self, size: Tuple[int, int]):
        self.size = size

    def __call__(self, image: NDArray) -> NDArray:
        if HAS_OPENCV:
            return cv2.resize(image, self.size)
        elif HAS_PIL:
            img = Image.fromarray(image)
            img = img.resize(self.size)
            return np.array(img)
        else:
            raise ImportError("Either OpenCV or PIL is required")


class RandomHorizontalFlip(Transform):
    """Random horizontal flip."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: NDArray) -> NDArray:
        if np.random.rand() < self.p:
            return np.fliplr(image)
        return image


class ToTensor(Transform):
    """Convert to tensor format (C, H, W)."""

    def __call__(self, image: NDArray) -> NDArray:
        # HWC -> CHW
        if len(image.shape) == 3:
            return np.transpose(image, (2, 0, 1))
        return image


class BatchProcessor:
    """Process data in batches with parallel workers."""

    def __init__(
        self,
        process_fn: Callable,
        batch_size: int = 32,
        num_workers: int = 4
    ):
        """
        Initialize batch processor.

        Parameters
        ----------
        process_fn : callable
            Function to apply to each batch
        batch_size : int, default=32
            Batch size
        num_workers : int, default=4
            Number of parallel workers
        """
        self.process_fn = process_fn
        self.batch_size = batch_size
        self.num_workers = num_workers

    def process(self, data: List[Any]) -> List[Any]:
        """
        Process data in parallel batches.

        Parameters
        ----------
        data : list
            Input data

        Returns
        -------
        results : list
            Processed results
        """
        # Create batches
        batches = [data[i:i + self.batch_size]
                   for i in range(0, len(data), self.batch_size)]

        # Process in parallel
        if self.num_workers == 1:
            results = [self.process_fn(batch) for batch in batches]
        else:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                results = list(executor.map(self.process_fn, batches))

        # Flatten results
        flat_results = []
        for batch_result in results:
            if isinstance(batch_result, list):
                flat_results.extend(batch_result)
            else:
                flat_results.append(batch_result)

        return flat_results


# Convenience functions
def create_image_dataloader(
    image_dir: Union[str, Path],
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    transform: Optional[Transform] = None
) -> DataLoader:
    """
    Create image data loader.

    Parameters
    ----------
    image_dir : str or Path
        Directory containing images
    batch_size : int, default=32
        Batch size
    shuffle : bool, default=True
        Shuffle data
    num_workers : int, default=4
        Number of worker threads
    transform : Transform, optional
        Transformations to apply

    Returns
    -------
    dataloader : DataLoader
        Configured data loader
    """
    dataset = ImageDataset(image_dir, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )


def load_images_parallel(
    image_paths: List[Union[str, Path]],
    num_workers: int = 4
) -> List[NDArray]:
    """
    Load images in parallel.

    Parameters
    ----------
    image_paths : list
        List of image paths
    num_workers : int, default=4
        Number of parallel workers

    Returns
    -------
    images : list
        Loaded images
    """
    def load_image(path):
        if HAS_OPENCV:
            img = cv2.imread(str(path))
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif HAS_PIL:
            return np.array(Image.open(path))
        else:
            raise ImportError("Either OpenCV or PIL is required")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        images = list(executor.map(load_image, image_paths))

    return images
