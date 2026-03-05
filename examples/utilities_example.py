"""
Comprehensive example demonstrating PyImgAno's utility functions.

This example shows how to use:
1. Dataset loading utilities
2. Advanced visualization
3. Model management
4. Experiment tracking

Run: python examples/utilities_example.py
"""

import numpy as np

from pyimgano.models import create_model
from pyimgano.utils import (  # Dataset utilities; Visualization utilities; Model utilities; Experiment tracking
    ExperimentTracker,
    ModelRegistry,
    MVTecDataset,
    compare_models,
    create_evaluation_report,
    load_dataset,
    load_model,
    plot_anomaly_heatmap,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_score_distribution,
    profile_model,
    save_model,
    track_experiment,
)


def example_1_dataset_loading():
    """Example 1: Dataset Loading Utilities"""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Dataset Loading Utilities")
    print("=" * 80)

    # Method 1: Using MVTecDataset directly
    print("\n1.1 Loading MVTec AD dataset...")
    dataset = MVTecDataset(
        root="./datasets/mvtec_ad", category="bottle", resize=(256, 256), load_masks=True
    )

    # Get training data
    train_data = dataset.get_train_data()
    print(f"Training images: {train_data.shape}")

    # Get test data with labels and masks
    test_data, test_labels, test_masks = dataset.get_test_data()
    print(f"Test images: {test_data.shape}")
    print(f"Test labels: {test_labels.shape} - {test_labels.sum()} anomalies")
    if test_masks is not None:
        print(f"Test masks: {test_masks.shape}")

    # Get dataset info
    info = dataset.get_info()
    print(f"\nDataset: {info.name}")
    print(f"Category: {dataset.category}")
    print(f"Num train: {info.num_train}")
    print(f"Num test: {info.num_test}")

    # Method 2: Using factory function
    print("\n1.2 Using load_dataset factory...")
    dataset2 = load_dataset("mvtec", "./datasets/mvtec_ad", category="bottle")
    train_data2 = dataset2.get_train_data()
    print(f"Loaded {len(train_data2)} training images")

    return train_data, test_data, test_labels, test_masks


def example_2_advanced_visualization(test_data, test_labels, scores, predictions):
    """Example 2: Advanced Visualization"""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Advanced Visualization")
    print("=" * 80)

    # 2.1 ROC Curve
    print("\n2.1 Plotting ROC curve...")
    auc_score, _ = plot_roc_curve(
        test_labels,
        scores,
        title="Model ROC Curve",
        save_path="./outputs/roc_curve.png",
        show=False,
    )
    print(f"AUC-ROC: {auc_score:.4f}")

    # 2.2 Confusion Matrix
    print("\n2.2 Plotting confusion matrix...")
    plot_confusion_matrix(
        test_labels,
        predictions,
        labels=["Normal", "Anomaly"],
        title="Confusion Matrix",
        save_path="./outputs/confusion_matrix.png",
        show=False,
    )

    # 2.3 Score Distribution
    print("\n2.3 Plotting score distribution...")
    normal_scores = scores[test_labels == 0]
    anomaly_scores = scores[test_labels == 1]
    plot_score_distribution(
        normal_scores, anomaly_scores, save_path="./outputs/score_distribution.png", show=False
    )

    # 2.4 Anomaly Heatmap (for first anomaly)
    print("\n2.4 Plotting anomaly heatmap...")
    first_anomaly_idx = np.where(test_labels == 1)[0][0]
    # Assuming model has predict_anomaly_map method
    # anomaly_map = model.predict_anomaly_map(test_data[[first_anomaly_idx]])[0]
    # plot_anomaly_heatmap(
    #     test_data[first_anomaly_idx],
    #     anomaly_map,
    #     save_path='./outputs/anomaly_heatmap.png',
    #     show=False
    # )

    # 2.5 Create full evaluation report
    print("\n2.5 Creating comprehensive evaluation report...")
    figures = create_evaluation_report(
        test_labels,
        scores,
        predictions,
        model_name="PatchCore",
        save_dir="./outputs/evaluation_report",
    )
    print(f"Generated {len(figures)} evaluation plots")


def example_3_model_management(model):
    """Example 3: Model Management"""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Model Management")
    print("=" * 80)

    # 3.1 Save and load model
    print("\n3.1 Saving model...")
    save_model(
        model,
        "./models/patchcore_bottle.pkl",
        metadata={"category": "bottle", "auc": 0.98, "trained_at": "2025-01-01"},
    )

    print("\n3.2 Loading model...")
    loaded_model = load_model("./models/patchcore_bottle.pkl")

    # 3.3 Model Registry
    print("\n3.3 Using Model Registry...")
    registry = ModelRegistry("./model_registry")

    registry.register("patchcore_bottle_v1", model, metadata={"version": "v1", "auc": 0.98})

    registry.register(
        "patchcore_bottle_v2", model, metadata={"version": "v2", "auc": 0.99}, overwrite=True
    )

    # List models
    models = registry.list_models()
    print(f"Registered models: {len(models)}")
    for name, info in models.items():
        print(f"  - {name}: {info['metadata']}")

    # 3.4 Model profiling
    print("\n3.4 Profiling model performance...")
    # Create dummy test data for profiling
    dummy_data = np.random.rand(100, 256, 256, 3).astype(np.uint8)
    metrics = profile_model(model, dummy_data, n_runs=5)

    print(f"Average inference time: {metrics['avg_time_ms']:.2f} ms")
    print(f"Throughput: {metrics['throughput_samples_per_sec']:.2f} samples/sec")
    print(f"Latency per sample: {metrics['latency_per_sample_ms']:.2f} ms")


def example_4_experiment_tracking(model, train_data, test_data, test_labels):
    """Example 4: Experiment Tracking"""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Experiment Tracking")
    print("=" * 80)

    # 4.1 Create experiment tracker
    print("\n4.1 Creating experiment tracker...")
    tracker = ExperimentTracker("./experiments")

    # 4.2 Create and log experiment
    print("\n4.2 Creating experiment...")
    exp = tracker.create_experiment(
        "patchcore_bottle_exp",
        model_type="PatchCore",
        dataset="MVTec AD",
        category="bottle",
        description="Testing PatchCore on bottle category",
    )

    # Log parameters
    exp.log_params(
        {
            "backbone": "wide_resnet50",
            "coreset_ratio": 0.1,
            "image_size": (256, 256),
            "batch_size": 8,
        }
    )

    # Log metrics
    exp.log_metric("auc_roc", 0.98)
    exp.log_metric("average_precision", 0.96)
    exp.log_metric("f1_score", 0.92)
    exp.log_metric("train_time_sec", 120.5)

    # Add tags
    exp.add_tag("production")
    exp.add_tag("mvtec")

    # Log artifacts
    exp.log_artifact("config.json", {"backbone": "wide_resnet50"})

    # 4.3 List all experiments
    print("\n4.3 Listing all experiments...")
    experiments = tracker.list_experiments()
    for exp_info in experiments:
        print(f"  - {exp_info['name']}: AUC={exp_info['latest_metrics'].get('auc_roc', 'N/A')}")

    # 4.4 Generate report
    print("\n4.4 Generating experiment report...")
    if experiments:
        report = tracker.generate_report(
            experiments[0]["exp_id"], output_path="./outputs/experiment_report.md"
        )
        print(f"Report saved: ./outputs/experiment_report.md")

    # 4.5 Quick experiment tracking
    print("\n4.5 Using quick experiment tracking...")
    exp2 = track_experiment(
        "simplenet_bottle",
        model=model,
        train_data=train_data,
        test_data=test_data,
        test_labels=test_labels,
        backbone="resnet18",
        epochs=10,
    )


def example_5_model_comparison(models_dict, test_data, test_labels):
    """Example 5: Model Comparison"""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Model Comparison")
    print("=" * 80)

    print("\n5.1 Comparing multiple models...")
    results = compare_models(models_dict, test_data, test_labels)

    print("\nComparison Results:")
    print("-" * 80)
    print(f"{'Model':<20} {'AUC-ROC':<10} {'AP':<10} {'F1':<10} {'Time (ms)':<15}")
    print("-" * 80)
    for name, metrics in results.items():
        print(
            f"{name:<20} "
            f"{metrics['auc_roc']:<10.4f} "
            f"{metrics['average_precision']:<10.4f} "
            f"{metrics['f1_score']:<10.4f} "
            f"{metrics['avg_inference_time_ms']:<15.2f}"
        )


def main():
    """Main function running all examples."""
    print("\n" + "=" * 80)
    print("PyImgAno Utilities Comprehensive Example")
    print("=" * 80)

    # Create output directories
    import os

    os.makedirs("./outputs", exist_ok=True)
    os.makedirs("./outputs/evaluation_report", exist_ok=True)
    os.makedirs("./models", exist_ok=True)

    # Example 1: Dataset Loading
    # Note: This requires the MVTec AD dataset to be downloaded
    # Uncomment if you have the dataset
    # train_data, test_data, test_labels, test_masks = example_1_dataset_loading()

    # For demonstration, create dummy data
    print("\nCreating dummy data for demonstration...")
    train_data = np.random.rand(50, 256, 256, 3).astype(np.uint8)
    test_data = np.random.rand(30, 256, 256, 3).astype(np.uint8)
    test_labels = np.array([0] * 20 + [1] * 10)  # 20 normal, 10 anomalies

    # Train a simple model
    print("\nTraining a simple model for demonstration...")
    model = create_model("vision_ecod", contamination=0.1)
    # Note: For real usage, use appropriate feature extraction
    # Here we just flatten for demonstration
    train_features = train_data.reshape(len(train_data), -1)
    test_features = test_data.reshape(len(test_data), -1)

    model.fit(train_features)
    scores = model.decision_function(test_features)
    predictions = model.predict(test_features)

    # Example 2: Advanced Visualization
    example_2_advanced_visualization(test_data, test_labels, scores, predictions)

    # Example 3: Model Management
    example_3_model_management(model)

    # Example 4: Experiment Tracking
    # example_4_experiment_tracking(model, train_data, test_data, test_labels)

    # Example 5: Model Comparison
    # models_dict = {
    #     'ECOD': model,
    #     'COPOD': create_model('vision_copod', contamination=0.1)
    # }
    # example_5_model_comparison(models_dict, test_features, test_labels)

    print("\n" + "=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - ./outputs/roc_curve.png")
    print("  - ./outputs/confusion_matrix.png")
    print("  - ./outputs/score_distribution.png")
    print("  - ./outputs/evaluation_report/")
    print("  - ./models/patchcore_bottle.pkl")
    print("  - ./model_registry/")
    print("  - ./experiments/")


if __name__ == "__main__":
    main()
