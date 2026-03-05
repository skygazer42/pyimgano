"""
Experiment tracking and management utilities.

Provides tools for:
- Logging experiments and hyperparameters
- Tracking metrics over time
- Generating experiment reports
- Comparing experiments

Example:
    >>> from pyimgano.utils.experiment_tracker import ExperimentTracker
    >>> tracker = ExperimentTracker('./experiments')
    >>> exp = tracker.create_experiment('patchcore_bottle')
    >>> exp.log_params({'backbone': 'resnet50', 'lr': 0.001})
    >>> exp.log_metric('auc', 0.98)
"""

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class ExperimentConfig:
    """Experiment configuration."""

    name: str
    created_at: str
    model_type: str
    dataset: str
    category: Optional[str] = None
    description: Optional[str] = None


class Experiment:
    """Single experiment tracker.

    Example:
        >>> exp = Experiment('./experiments/exp_001', 'patchcore_bottle')
        >>> exp.log_params({'backbone': 'resnet50'})
        >>> exp.log_metric('auc', 0.98)
        >>> exp.log_artifact('model.pkl', model_bytes)
    """

    def __init__(self, exp_dir: str, name: str):
        self.exp_dir = Path(exp_dir)
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        self.name = name
        self.params = {}
        self.metrics = {}
        self.tags = []
        self.artifacts = []

        # Create subdirectories
        self.params_file = self.exp_dir / "params.json"
        self.metrics_file = self.exp_dir / "metrics.json"
        self.artifacts_dir = self.exp_dir / "artifacts"
        self.artifacts_dir.mkdir(exist_ok=True)

        # Load existing if available
        if self.params_file.exists():
            with open(self.params_file, "r") as f:
                self.params = json.load(f)

        if self.metrics_file.exists():
            with open(self.metrics_file, "r") as f:
                self.metrics = json.load(f)

    def log_param(self, key: str, value: Any) -> None:
        """Log a single parameter.

        Args:
            key: Parameter name
            value: Parameter value
        """
        self.params[key] = value
        self._save_params()

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log multiple parameters.

        Args:
            params: Dictionary of parameters
        """
        self.params.update(params)
        self._save_params()

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Log a single metric.

        Args:
            key: Metric name
            value: Metric value
            step: Optional step/epoch number
        """
        if key not in self.metrics:
            self.metrics[key] = []

        entry = {
            "value": value,
            "timestamp": time.time(),
        }
        if step is not None:
            entry["step"] = step

        self.metrics[key].append(entry)
        self._save_metrics()

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log multiple metrics.

        Args:
            metrics: Dictionary of metrics
            step: Optional step/epoch number
        """
        for key, value in metrics.items():
            self.log_metric(key, value, step)

    def log_artifact(self, name: str, data: Any) -> None:
        """Log an artifact (file, image, model, etc.).

        Args:
            name: Artifact name
            data: Artifact data
        """
        import pickle

        artifact_path = self.artifacts_dir / name

        # Save based on extension
        if name.endswith(".json"):
            with open(artifact_path, "w") as f:
                json.dump(data, f, indent=2)
        elif name.endswith((".pkl", ".pickle")):
            with open(artifact_path, "wb") as f:
                pickle.dump(data, f)
        elif name.endswith(".npy"):
            np.save(artifact_path, data)
        elif name.endswith(".txt"):
            with open(artifact_path, "w") as f:
                f.write(str(data))
        else:
            # Binary data
            with open(artifact_path, "wb") as f:
                f.write(data)

        self.artifacts.append(str(artifact_path))

    def add_tag(self, tag: str) -> None:
        """Add a tag to the experiment.

        Args:
            tag: Tag string
        """
        if tag not in self.tags:
            self.tags.append(tag)
            self._save_params()

    def get_summary(self) -> Dict[str, Any]:
        """Get experiment summary.

        Returns:
            Summary dictionary
        """
        # Get latest metrics
        latest_metrics = {}
        for key, values in self.metrics.items():
            if values:
                latest_metrics[key] = values[-1]["value"]

        return {
            "name": self.name,
            "params": self.params,
            "latest_metrics": latest_metrics,
            "tags": self.tags,
            "num_artifacts": len(self.artifacts),
            "exp_dir": str(self.exp_dir),
        }

    def _save_params(self) -> None:
        """Save parameters to disk."""
        save_dict = {"params": self.params, "tags": self.tags}
        with open(self.params_file, "w") as f:
            json.dump(save_dict, f, indent=2)

    def _save_metrics(self) -> None:
        """Save metrics to disk."""
        with open(self.metrics_file, "w") as f:
            json.dump(self.metrics, f, indent=2)


class ExperimentTracker:
    """Experiment tracker for managing multiple experiments.

    Example:
        >>> tracker = ExperimentTracker('./experiments')
        >>> exp = tracker.create_experiment('patchcore_bottle')
        >>> exp.log_params({'backbone': 'resnet50'})
        >>> exp.log_metric('auc', 0.98)
        >>> tracker.list_experiments()
    """

    def __init__(self, base_dir: str = "./experiments"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.index_file = self.base_dir / "index.json"

        # Load existing index
        if self.index_file.exists():
            with open(self.index_file, "r") as f:
                self.index = json.load(f)
        else:
            self.index = {}

    def create_experiment(
        self,
        name: str,
        model_type: Optional[str] = None,
        dataset: Optional[str] = None,
        category: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Experiment:
        """Create a new experiment.

        Args:
            name: Experiment name
            model_type: Type of model
            dataset: Dataset name
            category: Dataset category
            description: Experiment description

        Returns:
            Experiment instance
        """
        # Generate unique ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_id = f"{name}_{timestamp}"

        exp_dir = self.base_dir / exp_id

        # Create experiment
        exp = Experiment(exp_dir, name)

        # Log config
        if model_type:
            exp.log_param("model_type", model_type)
        if dataset:
            exp.log_param("dataset", dataset)
        if category:
            exp.log_param("category", category)
        if description:
            exp.log_param("description", description)

        # Update index
        self.index[exp_id] = {"name": name, "created_at": timestamp, "exp_dir": str(exp_dir)}
        self._save_index()

        print(f"Experiment created: {exp_id}")

        return exp

    def get_experiment(self, exp_id: str) -> Experiment:
        """Get an existing experiment.

        Args:
            exp_id: Experiment ID

        Returns:
            Experiment instance
        """
        if exp_id not in self.index:
            raise ValueError(f"Experiment '{exp_id}' not found")

        exp_dir = self.index[exp_id]["exp_dir"]
        exp = Experiment(exp_dir, self.index[exp_id]["name"])

        return exp

    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all experiments.

        Returns:
            List of experiment summaries
        """
        experiments = []

        for exp_id, info in self.index.items():
            try:
                exp = self.get_experiment(exp_id)
                summary = exp.get_summary()
                summary["exp_id"] = exp_id
                summary["created_at"] = info["created_at"]
                experiments.append(summary)
            except Exception as e:
                print(f"Error loading experiment {exp_id}: {e}")

        return experiments

    def delete_experiment(self, exp_id: str) -> None:
        """Delete an experiment.

        Args:
            exp_id: Experiment ID
        """
        if exp_id not in self.index:
            raise ValueError(f"Experiment '{exp_id}' not found")

        # Remove directory
        exp_dir = Path(self.index[exp_id]["exp_dir"])
        if exp_dir.exists():
            import shutil

            shutil.rmtree(exp_dir)

        # Remove from index
        del self.index[exp_id]
        self._save_index()

        print(f"Experiment '{exp_id}' deleted")

    def compare_experiments(
        self, exp_ids: List[str], metrics: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """Compare multiple experiments.

        Args:
            exp_ids: List of experiment IDs
            metrics: List of metrics to compare (None for all)

        Returns:
            Dictionary of exp_id -> metrics
        """
        results = {}

        for exp_id in exp_ids:
            exp = self.get_experiment(exp_id)
            summary = exp.get_summary()

            if metrics is None:
                results[exp_id] = summary["latest_metrics"]
            else:
                results[exp_id] = {
                    k: v for k, v in summary["latest_metrics"].items() if k in metrics
                }

        return results

    def generate_report(self, exp_id: str, output_path: Optional[str] = None) -> str:
        """Generate markdown report for an experiment.

        Args:
            exp_id: Experiment ID
            output_path: Path to save report (optional)

        Returns:
            Report markdown string
        """
        exp = self.get_experiment(exp_id)
        summary = exp.get_summary()

        report = f"# Experiment Report: {summary['name']}\n\n"
        report += f"**Experiment ID:** {exp_id}\n"
        report += f"**Created:** {self.index[exp_id]['created_at']}\n\n"

        # Parameters
        report += "## Parameters\n\n"
        report += "| Parameter | Value |\n"
        report += "|-----------|-------|\n"
        for key, value in summary["params"].items():
            report += f"| {key} | {value} |\n"
        report += "\n"

        # Metrics
        report += "## Metrics\n\n"
        report += "| Metric | Value |\n"
        report += "|--------|-------|\n"
        for key, value in summary["latest_metrics"].items():
            report += f"| {key} | {value:.4f} |\n"
        report += "\n"

        # Tags
        if summary["tags"]:
            report += "## Tags\n\n"
            report += ", ".join(summary["tags"]) + "\n\n"

        # Artifacts
        if summary["num_artifacts"] > 0:
            report += f"## Artifacts\n\n"
            report += f"Number of artifacts: {summary['num_artifacts']}\n\n"

        if output_path:
            with open(output_path, "w") as f:
                f.write(report)
            print(f"Report saved to: {output_path}")

        return report

    def _save_index(self) -> None:
        """Save index to disk."""
        with open(self.index_file, "w") as f:
            json.dump(self.index, f, indent=2)


def track_experiment(
    name: str,
    model: Any,
    train_data: Any,
    test_data: Any,
    test_labels: Any,
    base_dir: str = "./experiments",
    **kwargs,
) -> Experiment:
    """Quick experiment tracking helper.

    Args:
        name: Experiment name
        model: Model to train and evaluate
        train_data: Training data
        test_data: Test data
        test_labels: Test labels
        base_dir: Base directory for experiments
        **kwargs: Additional parameters to log

    Returns:
        Experiment instance

    Example:
        >>> exp = track_experiment(
        ...     'patchcore_bottle',
        ...     model=detector,
        ...     train_data=train_imgs,
        ...     test_data=test_imgs,
        ...     test_labels=test_labels,
        ...     backbone='resnet50'
        ... )
    """
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

    # Create tracker and experiment
    tracker = ExperimentTracker(base_dir)
    exp = tracker.create_experiment(name, model_type=type(model).__name__)

    # Log parameters
    exp.log_params(kwargs)

    # Train
    print("Training model...")
    start_time = time.time()
    model.fit(train_data)
    train_time = time.time() - start_time
    exp.log_metric("train_time_sec", train_time)

    # Evaluate
    print("Evaluating model...")
    scores = model.predict_proba(test_data)
    predictions = model.predict(test_data)

    # Compute metrics
    auc_roc = roc_auc_score(test_labels, scores)
    ap = average_precision_score(test_labels, scores)
    f1 = f1_score(test_labels, predictions)

    exp.log_metric("auc_roc", auc_roc)
    exp.log_metric("average_precision", ap)
    exp.log_metric("f1_score", f1)

    print(f"AUC-ROC: {auc_roc:.4f}")
    print(f"AP: {ap:.4f}")
    print(f"F1: {f1:.4f}")

    return exp
