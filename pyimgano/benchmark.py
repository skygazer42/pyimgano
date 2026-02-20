"""
Benchmark utilities for comparing anomaly detection algorithms.

This module provides tools to:
- Benchmark multiple algorithms on the same dataset
- Measure training and inference time
- Compare performance metrics
- Generate comparison reports
"""

import logging
import time
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from numpy.typing import NDArray

from .evaluation import evaluate_detector
from .models import create_model

logger = logging.getLogger(__name__)


class AlgorithmBenchmark:
    """
    Benchmark multiple anomaly detection algorithms.

    Examples
    --------
    >>> from pyimgano.benchmark import AlgorithmBenchmark
    >>>
    >>> # Define algorithms to benchmark
    >>> algorithms = {
    ...     'ECOD': {'model_name': 'vision_ecod', 'contamination': 0.1},
    ...     'PatchCore': {'model_name': 'vision_patchcore', 'device': 'cuda'},
    ...     'SimpleNet': {'model_name': 'vision_simplenet', 'epochs': 10},
    ... }
    >>>
    >>> # Run benchmark
    >>> benchmark = AlgorithmBenchmark(algorithms)
    >>> results = benchmark.run(
    ...     train_images=train_paths,
    ...     test_images=test_paths,
    ...     test_labels=test_labels
    ... )
    >>>
    >>> # Print results
    >>> benchmark.print_summary()
    """

    def __init__(self, algorithms: Dict[str, Dict[str, Any]]):
        """
        Initialize benchmark.

        Parameters
        ----------
        algorithms : dict
            Dictionary mapping algorithm names to their configuration.
            Each config should contain 'model_name' and optional parameters.

        Examples
        --------
        >>> algorithms = {
        ...     'ECOD': {'model_name': 'vision_ecod'},
        ...     'COPOD': {'model_name': 'vision_copod'},
        ... }
        """
        self.algorithms = algorithms
        self.results: Dict[str, Dict] = {}

    def run(
        self,
        train_images: Iterable[str],
        test_images: Iterable[str],
        test_labels: NDArray,
        verbose: bool = True,
    ) -> Dict[str, Dict]:
        """
        Run benchmark on all algorithms.

        Parameters
        ----------
        train_images : iterable of str
            Paths to training images (normal only)
        test_images : iterable of str
            Paths to test images
        test_labels : ndarray
            Binary labels for test images (0=normal, 1=anomaly)
        verbose : bool, default=True
            Whether to print progress

        Returns
        -------
        results : dict
            Dictionary containing results for each algorithm
        """
        train_images = list(train_images)
        test_images = list(test_images)

        logger.info(
            "Starting benchmark with %d algorithms on %d train / %d test images",
            len(self.algorithms), len(train_images), len(test_images)
        )

        for algo_name, algo_config in self.algorithms.items():
            if verbose:
                print(f"\n{'=' * 60}")
                print(f"Benchmarking: {algo_name}")
                print(f"{'=' * 60}")

            try:
                result = self._benchmark_single(
                    algo_name,
                    algo_config,
                    train_images,
                    test_images,
                    test_labels,
                    verbose
                )
                self.results[algo_name] = result

            except Exception as e:
                logger.error("Failed to benchmark %s: %s", algo_name, e)
                self.results[algo_name] = {
                    'error': str(e),
                    'success': False
                }

        if verbose:
            print(f"\n{'=' * 60}")
            print("Benchmark Complete")
            print(f"{'=' * 60}\n")

        return self.results

    def _benchmark_single(
        self,
        algo_name: str,
        algo_config: Dict,
        train_images: List[str],
        test_images: List[str],
        test_labels: NDArray,
        verbose: bool,
    ) -> Dict:
        """Benchmark a single algorithm."""
        # Create model
        model_name = algo_config.pop('model_name')
        if verbose:
            print(f"Creating model: {model_name}")

        detector = create_model(model_name, **algo_config)

        # Training phase
        if verbose:
            print(f"Training on {len(train_images)} images...")

        train_start = time.time()
        detector.fit(train_images)
        train_time = time.time() - train_start

        if verbose:
            print(f"Training completed in {train_time:.2f}s")

        # Inference phase
        if verbose:
            print(f"Running inference on {len(test_images)} images...")

        inference_start = time.time()
        # Use continuous anomaly scores for evaluation/metrics.
        # `predict()` may return binary labels for some detectors (PyOD semantics),
        # which would break AUROC/AP computations.
        test_scores = detector.decision_function(test_images)
        inference_time = time.time() - inference_start

        inference_per_image = inference_time / len(test_images)

        if verbose:
            print(f"Inference completed in {inference_time:.2f}s "
                  f"({inference_per_image * 1000:.1f}ms per image)")

        # Evaluation
        if verbose:
            print("Evaluating performance...")

        eval_results = evaluate_detector(test_labels, test_scores)

        # Compile results
        result = {
            'success': True,
            'train_time': train_time,
            'inference_time': inference_time,
            'inference_per_image': inference_per_image,
            'auroc': eval_results['auroc'],
            'average_precision': eval_results['average_precision'],
            'threshold': eval_results['threshold'],
            'metrics': eval_results['metrics'],
        }

        if verbose:
            print(f"\nResults:")
            print(f"  AUROC: {result['auroc']:.4f}")
            print(f"  AP: {result['average_precision']:.4f}")
            print(f"  F1: {result['metrics']['f1']:.4f}")

        return result

    def print_summary(self) -> None:
        """Print formatted summary of benchmark results."""
        if not self.results:
            print("No results available. Run benchmark first.")
            return

        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80 + "\n")

        # Performance metrics table
        print("Performance Metrics:")
        print("-" * 80)
        header = f"{'Algorithm':<20} {'AUROC':>8} {'AP':>8} {'F1':>8} {'Precision':>10} {'Recall':>8}"
        print(header)
        print("-" * 80)

        for algo_name, result in self.results.items():
            if not result.get('success', False):
                print(f"{algo_name:<20} FAILED: {result.get('error', 'Unknown error')}")
                continue

            metrics = result['metrics']
            print(
                f"{algo_name:<20} "
                f"{result['auroc']:>8.4f} "
                f"{result['average_precision']:>8.4f} "
                f"{metrics['f1']:>8.4f} "
                f"{metrics['precision']:>10.4f} "
                f"{metrics['recall']:>8.4f}"
            )

        # Timing table
        print("\n\nTiming Results:")
        print("-" * 80)
        header = f"{'Algorithm':<20} {'Train (s)':>12} {'Inference (s)':>14} {'Per Image (ms)':>15}"
        print(header)
        print("-" * 80)

        for algo_name, result in self.results.items():
            if not result.get('success', False):
                continue

            print(
                f"{algo_name:<20} "
                f"{result['train_time']:>12.2f} "
                f"{result['inference_time']:>14.2f} "
                f"{result['inference_per_image'] * 1000:>15.1f}"
            )

        # Best performers
        print("\n\nBest Performers:")
        print("-" * 80)

        successful = {
            name: res for name, res in self.results.items()
            if res.get('success', False)
        }

        if successful:
            # Best AUROC
            best_auroc = max(successful.items(), key=lambda x: x[1]['auroc'])
            print(f"Best AUROC:          {best_auroc[0]} ({best_auroc[1]['auroc']:.4f})")

            # Best F1
            best_f1 = max(successful.items(), key=lambda x: x[1]['metrics']['f1'])
            print(f"Best F1:             {best_f1[0]} ({best_f1[1]['metrics']['f1']:.4f})")

            # Fastest training
            fastest_train = min(successful.items(), key=lambda x: x[1]['train_time'])
            print(f"Fastest Training:    {fastest_train[0]} ({fastest_train[1]['train_time']:.2f}s)")

            # Fastest inference
            fastest_inf = min(successful.items(), key=lambda x: x[1]['inference_per_image'])
            print(f"Fastest Inference:   {fastest_inf[0]} "
                  f"({fastest_inf[1]['inference_per_image'] * 1000:.1f}ms/image)")

        print("\n" + "=" * 80 + "\n")

    def get_rankings(self, metric: str = 'auroc') -> List[tuple]:
        """
        Get algorithms ranked by a specific metric.

        Parameters
        ----------
        metric : str, default='auroc'
            Metric to rank by ('auroc', 'average_precision', 'f1',
            'train_time', 'inference_per_image')

        Returns
        -------
        rankings : list of tuple
            List of (algorithm_name, metric_value) sorted by metric
        """
        successful = {
            name: res for name, res in self.results.items()
            if res.get('success', False)
        }

        if metric in ['auroc', 'average_precision', 'train_time', 'inference_per_image']:
            rankings = [
                (name, res[metric])
                for name, res in successful.items()
            ]
        elif metric in ['f1', 'precision', 'recall', 'accuracy']:
            rankings = [
                (name, res['metrics'][metric])
                for name, res in successful.items()
            ]
        else:
            raise ValueError(f"Unknown metric: {metric}")

        # Sort (descending for performance metrics, ascending for time)
        reverse = metric not in ['train_time', 'inference_per_image']
        rankings.sort(key=lambda x: x[1], reverse=reverse)

        return rankings

    def save_results(self, filepath: str) -> None:
        """
        Save benchmark results to JSON file.

        Parameters
        ----------
        filepath : str
            Path to save results

        Examples
        --------
        >>> benchmark.save_results('benchmark_results.json')
        """
        import json

        # Convert numpy types to native Python types
        serializable_results = {}
        for algo_name, result in self.results.items():
            if not result.get('success', False):
                serializable_results[algo_name] = result
                continue

            serializable_result = {
                'success': result['success'],
                'train_time': float(result['train_time']),
                'inference_time': float(result['inference_time']),
                'inference_per_image': float(result['inference_per_image']),
                'auroc': float(result['auroc']),
                'average_precision': float(result['average_precision']),
                'threshold': float(result['threshold']),
                'metrics': {
                    k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                    for k, v in result['metrics'].items()
                }
            }
            serializable_results[algo_name] = serializable_result

        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        logger.info("Results saved to %s", filepath)


def quick_benchmark(
    train_images: Iterable[str],
    test_images: Iterable[str],
    test_labels: NDArray,
    algorithms: Optional[List[str]] = None,
) -> Dict:
    """
    Quick benchmark with default configurations.

    Parameters
    ----------
    train_images : iterable of str
        Paths to training images
    test_images : iterable of str
        Paths to test images
    test_labels : ndarray
        Test labels
    algorithms : list of str, optional
        List of algorithm names to benchmark. If None, uses common algorithms.

    Returns
    -------
    results : dict
        Benchmark results

    Examples
    --------
    >>> results = quick_benchmark(
    ...     train_images=train_paths,
    ...     test_images=test_paths,
    ...     test_labels=test_labels,
    ...     algorithms=['ECOD', 'COPOD', 'PatchCore']
    ... )
    """
    if algorithms is None:
        # Default fast algorithms
        algorithms = ['ECOD', 'COPOD', 'KNN']

    # Map algorithm names to configurations
    algo_configs = {}

    for algo in algorithms:
        algo_upper = algo.upper()

        if algo_upper == 'ECOD':
            algo_configs['ECOD'] = {'model_name': 'vision_ecod'}
        elif algo_upper == 'COPOD':
            algo_configs['COPOD'] = {'model_name': 'vision_copod'}
        elif algo_upper == 'KNN':
            algo_configs['KNN'] = {'model_name': 'vision_knn', 'n_neighbors': 5}
        elif algo_upper == 'PCA':
            algo_configs['PCA'] = {'model_name': 'vision_pca'}
        elif algo_upper == 'PATCHCORE':
            algo_configs['PatchCore'] = {
                'model_name': 'vision_patchcore',
                'coreset_sampling_ratio': 0.1
            }
        elif algo_upper == 'SIMPLENET':
            algo_configs['SimpleNet'] = {
                'model_name': 'vision_simplenet',
                'epochs': 10
            }
        else:
            logger.warning("Unknown algorithm: %s, skipping", algo)

    # Run benchmark
    benchmark = AlgorithmBenchmark(algo_configs)
    results = benchmark.run(train_images, test_images, test_labels)

    benchmark.print_summary()

    return results
