"""
Model management utilities.

Provides tools for:
- Saving and loading models
- Model conversion and export
- Model inspection and profiling
- Checkpointing

Example:
    >>> from pyimgano.utils.model_utils import save_model, load_model
    >>> save_model(detector, 'my_model.pkl')
    >>> detector = load_model('my_model.pkl')
"""

import json
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from numpy.typing import NDArray


def save_model(
    model: Any, path: str, metadata: Optional[Dict] = None, compression: bool = True
) -> None:
    """Save model to disk.

    Args:
        model: Model to save
        path: Save path
        metadata: Optional metadata dictionary
        compression: Whether to use compression

    Example:
        >>> save_model(detector, 'patchcore_bottle.pkl',
        ...            metadata={'category': 'bottle', 'auc': 0.98})
    """
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        "model": model,
        "metadata": metadata or {},
        "timestamp": time.time(),
        "pyimgano_version": "0.2.4",
    }

    protocol = pickle.HIGHEST_PROTOCOL if compression else pickle.DEFAULT_PROTOCOL

    with open(save_path, "wb") as f:
        pickle.dump(save_dict, f, protocol=protocol)

    print(f"Model saved to: {save_path}")


def load_model(path: str) -> Any:
    """Load model from disk.

    Args:
        path: Path to saved model

    Returns:
        Loaded model

    Example:
        >>> detector = load_model('patchcore_bottle.pkl')
    """
    with open(path, "rb") as f:
        save_dict = pickle.load(f)

    model = save_dict["model"]
    metadata = save_dict.get("metadata", {})

    print(f"Model loaded from: {path}")
    if metadata:
        print(f"Metadata: {metadata}")

    return model


def save_checkpoint(
    model: Any,
    save_dir: str,
    epoch: int,
    metrics: Optional[Dict[str, float]] = None,
    keep_last_n: int = 5,
) -> None:
    """Save model checkpoint.

    Args:
        model: Model to save
        save_dir: Directory to save checkpoints
        epoch: Current epoch number
        metrics: Training metrics
        keep_last_n: Number of recent checkpoints to keep

    Example:
        >>> save_checkpoint(detector, './checkpoints', epoch=10,
        ...                 metrics={'loss': 0.05, 'auc': 0.95})
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = save_dir / f"checkpoint_epoch_{epoch:04d}.pkl"

    save_dict = {"model": model, "epoch": epoch, "metrics": metrics or {}, "timestamp": time.time()}

    with open(checkpoint_path, "wb") as f:
        pickle.dump(save_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Clean up old checkpoints
    checkpoints = sorted(save_dir.glob("checkpoint_epoch_*.pkl"))
    if len(checkpoints) > keep_last_n:
        for old_checkpoint in checkpoints[:-keep_last_n]:
            old_checkpoint.unlink()

    print(f"Checkpoint saved: {checkpoint_path}")


def load_checkpoint(path: str) -> Dict[str, Any]:
    """Load model checkpoint.

    Args:
        path: Path to checkpoint

    Returns:
        Dictionary with model, epoch, and metrics

    Example:
        >>> checkpoint = load_checkpoint('./checkpoints/checkpoint_epoch_0010.pkl')
        >>> model = checkpoint['model']
        >>> epoch = checkpoint['epoch']
    """
    with open(path, "rb") as f:
        checkpoint = pickle.load(f)

    print(f"Checkpoint loaded from: {path}")
    print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"Metrics: {checkpoint.get('metrics', {})}")

    return checkpoint


def get_model_info(model: Any) -> Dict[str, Any]:
    """Get model information.

    Args:
        model: Model instance

    Returns:
        Dictionary with model information

    Example:
        >>> info = get_model_info(detector)
        >>> print(f"Model type: {info['type']}")
    """
    info = {
        "type": type(model).__name__,
        "module": type(model).__module__,
        "has_fit": hasattr(model, "fit"),
        "has_predict": hasattr(model, "predict"),
        "has_predict_proba": hasattr(model, "predict_proba"),
        "is_fitted": getattr(model, "fitted_", False),
    }

    # Try to get model parameters
    if hasattr(model, "__dict__"):
        params = {
            k: v for k, v in model.__dict__.items() if not k.startswith("_") and not callable(v)
        }
        info["parameters"] = params

    return info


def export_model_config(model: Any, path: str) -> None:
    """Export model configuration to JSON.

    Args:
        model: Model instance
        path: Save path for config file

    Example:
        >>> export_model_config(detector, 'model_config.json')
    """
    config = {}

    # Get init parameters if available
    if hasattr(model, "__dict__"):
        for key, value in model.__dict__.items():
            if not key.startswith("_") and not callable(value):
                # Convert to JSON-serializable types
                if isinstance(value, (int, float, str, bool, type(None))):
                    config[key] = value
                elif isinstance(value, (list, tuple)):
                    config[key] = list(value)
                elif isinstance(value, dict):
                    config[key] = dict(value)
                elif isinstance(value, np.ndarray):
                    config[key] = {"shape": value.shape, "dtype": str(value.dtype)}
                else:
                    config[key] = str(value)

    config["model_type"] = type(model).__name__

    with open(path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Model config exported to: {path}")


def profile_model(model: Any, test_data: NDArray, n_runs: int = 10) -> Dict[str, float]:
    """Profile model inference performance.

    Args:
        model: Trained model
        test_data: Test data [N, ...]
        n_runs: Number of profiling runs

    Returns:
        Performance metrics

    Example:
        >>> metrics = profile_model(detector, test_images, n_runs=10)
        >>> print(f"Avg inference time: {metrics['avg_time_ms']:.2f} ms")
    """
    import time

    times = []

    # Warmup
    _ = model.predict_proba(test_data[:1])

    # Profile
    for _ in range(n_runs):
        start = time.time()
        _ = model.predict_proba(test_data)
        end = time.time()
        times.append((end - start) * 1000)  # Convert to ms

    metrics = {
        "avg_time_ms": np.mean(times),
        "std_time_ms": np.std(times),
        "min_time_ms": np.min(times),
        "max_time_ms": np.max(times),
        "throughput_samples_per_sec": len(test_data) / (np.mean(times) / 1000),
        "latency_per_sample_ms": np.mean(times) / len(test_data),
    }

    return metrics


class ModelRegistry:
    """Registry for managing multiple models.

    Example:
        >>> registry = ModelRegistry('./models')
        >>> registry.register('patchcore_bottle', detector, metadata={'auc': 0.98})
        >>> model = registry.load('patchcore_bottle')
    """

    def __init__(self, base_dir: str = "./model_registry"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.base_dir / "index.json"

        # Load existing index
        if self.index_file.exists():
            with open(self.index_file, "r") as f:
                self.index = json.load(f)
        else:
            self.index = {}

    def register(
        self, name: str, model: Any, metadata: Optional[Dict] = None, overwrite: bool = False
    ) -> None:
        """Register a model.

        Args:
            name: Model name
            model: Model instance
            metadata: Optional metadata
            overwrite: Whether to overwrite existing model
        """
        if name in self.index and not overwrite:
            raise ValueError(f"Model '{name}' already exists. Set overwrite=True to replace.")

        # Save model
        model_path = self.base_dir / f"{name}.pkl"
        save_model(model, model_path, metadata=metadata)

        # Update index
        self.index[name] = {
            "path": str(model_path),
            "metadata": metadata or {},
            "registered_at": time.time(),
        }

        self._save_index()

        print(f"Model '{name}' registered successfully")

    def load(self, name: str) -> Any:
        """Load a registered model.

        Args:
            name: Model name

        Returns:
            Loaded model
        """
        if name not in self.index:
            raise ValueError(f"Model '{name}' not found in registry")

        model_path = self.index[name]["path"]
        return load_model(model_path)

    def list_models(self) -> Dict[str, Dict]:
        """List all registered models.

        Returns:
            Dictionary of model_name -> info
        """
        return self.index.copy()

    def remove(self, name: str) -> None:
        """Remove a model from registry.

        Args:
            name: Model name
        """
        if name not in self.index:
            raise ValueError(f"Model '{name}' not found in registry")

        # Remove model file
        model_path = Path(self.index[name]["path"])
        if model_path.exists():
            model_path.unlink()

        # Remove from index
        del self.index[name]
        self._save_index()

        print(f"Model '{name}' removed from registry")

    def _save_index(self) -> None:
        """Save index to disk."""
        with open(self.index_file, "w") as f:
            json.dump(self.index, f, indent=2)


def compare_models(
    models: Dict[str, Any], test_data: NDArray, test_labels: NDArray
) -> Dict[str, Dict[str, float]]:
    """Compare multiple models on the same test set.

    Args:
        models: Dictionary of model_name -> model
        test_data: Test data [N, ...]
        test_labels: Test labels [N]

    Returns:
        Dictionary of model_name -> metrics

    Example:
        >>> models = {'PatchCore': model1, 'SimpleNet': model2}
        >>> results = compare_models(models, test_data, test_labels)
    """
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

    results = {}

    for name, model in models.items():
        print(f"Evaluating {name}...")

        # Predict
        scores = model.predict_proba(test_data)
        predictions = model.predict(test_data)

        # Compute metrics
        auc_roc = roc_auc_score(test_labels, scores)
        ap = average_precision_score(test_labels, scores)
        f1 = f1_score(test_labels, predictions)

        # Profile
        profile = profile_model(model, test_data[:100], n_runs=5)

        results[name] = {
            "auc_roc": auc_roc,
            "average_precision": ap,
            "f1_score": f1,
            "avg_inference_time_ms": profile["avg_time_ms"],
            "throughput_fps": profile["throughput_samples_per_sec"],
        }

    return results
