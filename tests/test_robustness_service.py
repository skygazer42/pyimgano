from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pyimgano.services.robustness_service import RobustnessRunRequest, run_robustness_request


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_summarize_robustness_report_includes_drop_metrics() -> None:
    from pyimgano.services.robustness_service import _summarize_robustness_report

    summary = _summarize_robustness_report(
        {
            "clean": {"results": {"auroc": 0.95}, "latency_ms_per_image": 1.0},
            "corruptions": {
                "lighting": {
                    "severity_1": {"results": {"auroc": 0.90}, "latency_ms_per_image": 1.2},
                    "severity_2": {"results": {"auroc": 0.80}, "latency_ms_per_image": 1.3},
                }
            },
        }
    )

    assert summary["clean_auroc"] == 0.95
    assert summary["mean_corruption_auroc"] == 0.85
    assert summary["worst_corruption_auroc"] == 0.8
    assert summary["mean_corruption_drop_auroc"] == 0.1
    assert summary["worst_corruption_drop_auroc"] == 0.15
    assert summary["clean_latency_ms_per_image"] == 1.0
    assert summary["mean_corruption_latency_ms_per_image"] == 1.25
    assert summary["worst_corruption_latency_ms_per_image"] == 1.3
    assert summary["mean_corruption_latency_ratio"] == 1.25
    assert summary["worst_corruption_latency_ratio"] == 1.3


def test_summarize_robustness_protocol_exposes_comparability_metadata() -> None:
    from pyimgano.reporting.robustness_summary import summarize_robustness_protocol

    summary = summarize_robustness_protocol(
        {
            "corruption_mode": "full",
            "clean": {"results": {"auroc": 0.95}},
            "corruptions": {
                "lighting": {
                    "severity_1": {"results": {"auroc": 0.90}},
                    "severity_2": {"results": {"auroc": 0.80}},
                },
                "jpeg": {
                    "severity_1": {"results": {"auroc": 0.88}},
                },
            },
        }
    )

    assert summary["corruption_mode"] == "full"
    assert summary["has_clean_baseline"] is True
    assert summary["condition_count"] == 4
    assert summary["corruption_count"] == 2
    assert summary["severity_count"] == 2
    assert summary["conditions"] == ["clean", "jpeg", "lighting"]
    assert summary["severities"] == [1, 2]
    assert summary["comparability_hints"] == {
        "recommends_same_environment": True,
        "requires_same_category": True,
        "requires_same_corruption_protocol": True,
        "requires_same_dataset": True,
        "requires_same_input_mode": True,
        "requires_same_resize": True,
        "requires_same_severities": True,
        "requires_same_split": True,
    }


def test_run_robustness_request_delegates_to_benchmark(monkeypatch) -> None:
    import pyimgano.services.robustness_service as robustness_service
    import pyimgano.services.dataset_split_service as dataset_split_service

    calls: list[dict[str, object]] = []

    class _Split:
        train_paths = ["train_0.png"]
        test_paths = ["test_0.png"]
        test_labels = [0]
        test_masks = None

    monkeypatch.setattr(
        dataset_split_service,
        "load_benchmark_style_split",
        lambda **_kwargs: dataset_split_service.LoadedBenchmarkSplit(
            split=_Split(),
            pixel_skip_reason=None,
        ),
    )
    monkeypatch.setattr(robustness_service, "create_model", lambda *_a, **_k: object())

    monkeypatch.setattr(
        robustness_service,
        "_run_robustness_benchmark",
        lambda *args, **kwargs: calls.append(kwargs) or {"clean": {}, "corruptions": {}},
    )

    payload = run_robustness_request(
        RobustnessRunRequest(
            dataset="mvtec",
            root="/tmp/root",
            category="bottle",
            model="vision_ecod",
            input_mode="paths",
            pixel_segf1=False,
        )
    )

    assert "robustness" in payload
    assert payload["robustness_protocol"]["corruption_mode"] == "clean_only"
    assert payload["robustness_protocol"]["condition_count"] == 1
    assert payload["robustness_protocol"]["comparability_hints"]["requires_same_corruption_protocol"] is True
    assert payload["robustness_trust"]["status"] == "partial"
    assert "clean_only_mode" in payload["robustness_trust"]["degraded_by"]
    assert "missing_corruption_conditions" in payload["robustness_trust"]["degraded_by"]
    assert payload["model"] == "vision_ecod"
    assert isinstance(calls, list)


def test_run_robustness_request_uses_request_checkpoint_path(monkeypatch) -> None:
    import pyimgano.services.robustness_service as robustness_service
    import pyimgano.services.dataset_split_service as dataset_split_service

    class _Split:
        train_paths = ["train_0.png"]
        test_paths = ["test_0.png"]
        test_labels = [0]
        test_masks = None

    captured: dict[str, object] = {}

    monkeypatch.setattr(
        dataset_split_service,
        "load_benchmark_style_split",
        lambda **_kwargs: dataset_split_service.LoadedBenchmarkSplit(
            split=_Split(),
            pixel_skip_reason=None,
        ),
    )
    monkeypatch.setattr(
        robustness_service,
        "create_model",
        lambda name, **kwargs: captured.update(name=str(name), kwargs=dict(kwargs)) or object(),
    )
    monkeypatch.setattr(
        robustness_service,
        "_run_robustness_benchmark",
        lambda *args, **kwargs: {"clean": {}, "corruptions": {}},
    )

    payload = robustness_service.run_robustness_request(
        robustness_service.RobustnessRunRequest(
            dataset="mvtec",
            root="/tmp/root",
            category="bottle",
            model="vision_patchcore_anomalib",
            checkpoint_path="/tmp/checkpoint.ckpt",
            input_mode="paths",
            pixel_segf1=False,
        )
    )

    assert payload["model"] == "vision_patchcore_anomalib"
    assert captured["name"] == "vision_patchcore_anomalib"
    assert captured["kwargs"]["checkpoint_path"] == "/tmp/checkpoint.ckpt"


def test_run_robustness_request_accepts_model_preset_alias(monkeypatch) -> None:
    import pyimgano.services.robustness_service as robustness_service
    import pyimgano.services.dataset_split_service as dataset_split_service

    class _Split:
        train_paths = ["train_0.png"]
        test_paths = ["test_0.png"]
        test_labels = [0]
        test_masks = None

    captured: dict[str, object] = {}

    monkeypatch.setattr(
        dataset_split_service,
        "load_benchmark_style_split",
        lambda **_kwargs: dataset_split_service.LoadedBenchmarkSplit(
            split=_Split(),
            pixel_skip_reason=None,
        ),
    )
    monkeypatch.setattr(
        robustness_service,
        "create_model",
        lambda name, **kwargs: captured.update(name=str(name), kwargs=dict(kwargs)) or object(),
    )
    monkeypatch.setattr(
        robustness_service,
        "_run_robustness_benchmark",
        lambda *args, **kwargs: {"clean": {}, "corruptions": {}},
    )

    payload = robustness_service.run_robustness_request(
        robustness_service.RobustnessRunRequest(
            dataset="mvtec",
            root="/tmp/root",
            category="bottle",
            model="industrial-structural-ecod",
            input_mode="paths",
            pixel_segf1=False,
        )
    )

    assert payload["model"] == "industrial-structural-ecod"
    assert captured["name"] == "vision_feature_pipeline"


def test_run_robustness_request_delegates_split_loading_through_service(monkeypatch) -> None:
    import pyimgano.services.robustness_service as robustness_service
    import pyimgano.services.dataset_split_service as dataset_split_service

    calls: list[dict[str, object]] = []

    class _Split:
        train_paths = ["train_0.png"]
        test_paths = ["test_0.png"]
        test_labels = [0]
        test_masks = None

    monkeypatch.setattr(
        dataset_split_service,
        "load_benchmark_style_split",
        lambda **kwargs: calls.append(dict(kwargs))
        or dataset_split_service.LoadedBenchmarkSplit(
            split=_Split(),
            pixel_skip_reason=None,
        ),
    )
    monkeypatch.setattr(robustness_service, "create_model", lambda *_a, **_k: object())
    monkeypatch.setattr(
        robustness_service,
        "_run_robustness_benchmark",
        lambda *args, **kwargs: {"clean": {}, "corruptions": {}},
    )

    robustness_service.run_robustness_request(
        RobustnessRunRequest(
            dataset="mvtec",
            root="/tmp/root",
            category="bottle",
            model="vision_ecod",
            resize=(32, 32),
            input_mode="paths",
            pixel_segf1=False,
        )
    )

    assert len(calls) == 1
    assert calls[0]["dataset"] == "mvtec"
    assert calls[0]["root"] == "/tmp/root"
    assert calls[0]["category"] == "bottle"
    assert calls[0]["resize"] == (32, 32)
    assert calls[0]["load_masks"] is True


def test_run_robustness_request_can_persist_run_artifacts(monkeypatch, tmp_path: Path) -> None:
    import pyimgano.services.robustness_service as robustness_service
    import pyimgano.services.dataset_split_service as dataset_split_service

    class _Split:
        train_paths = ["train_0.png"]
        test_paths = ["test_0.png"]
        test_labels = [0]
        test_masks = None

    monkeypatch.setattr(
        dataset_split_service,
        "load_benchmark_style_split",
        lambda **_kwargs: dataset_split_service.LoadedBenchmarkSplit(
            split=_Split(),
            pixel_skip_reason=None,
        ),
    )
    monkeypatch.setattr(robustness_service, "create_model", lambda *_a, **_k: object())
    monkeypatch.setattr(
        robustness_service,
        "_run_robustness_benchmark",
        lambda *args, **kwargs: {
            "clean": {"results": {"auroc": 0.95}, "latency_ms_per_image": 1.0},
            "corruptions": {
                "lighting": {
                    "severity_1": {"results": {"auroc": 0.85}, "latency_ms_per_image": 1.2}
                }
            },
        },
    )

    payload = robustness_service.run_robustness_request(
        robustness_service.RobustnessRunRequest(
            dataset="mvtec",
            root="/tmp/root",
            category="bottle",
            model="vision_ecod",
            input_mode="paths",
            pixel_segf1=False,
            output_dir=str(tmp_path / "robust_run"),
        )
    )

    run_dir = Path(str(payload["run_dir"]))
    assert run_dir.exists()
    assert (run_dir / "report.json").exists()
    assert (run_dir / "config.json").exists()
    assert (run_dir / "environment.json").exists()
    assert (run_dir / "artifacts" / "robustness_conditions.csv").exists()
    assert (run_dir / "artifacts" / "robustness_summary.json").exists()

    saved_report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    assert saved_report["robustness_summary"]["clean_auroc"] == 0.95
    assert saved_report["robustness_summary"]["worst_corruption_auroc"] == 0.85
    assert saved_report["robustness_protocol"]["corruption_mode"] == "clean_only"
    assert saved_report["robustness_protocol"]["condition_count"] == 2
    assert saved_report["robustness_protocol"]["conditions"] == ["clean", "lighting"]
    assert saved_report["robustness_trust"]["status"] == "partial"
    assert saved_report["robustness_trust"]["trust_signals"]["has_audit_refs"] is True
    assert saved_report["robustness_trust"]["trust_signals"]["has_audit_digests"] is True
    assert (
        saved_report["robustness_trust"]["audit_refs"]["robustness_conditions_csv"]
        == "artifacts/robustness_conditions.csv"
    )
    assert (
        saved_report["robustness_trust"]["audit_refs"]["robustness_summary_json"]
        == "artifacts/robustness_summary.json"
    )
    assert saved_report["robustness_trust"]["audit_digests"]["robustness_conditions_csv"] == _sha256(
        run_dir / "artifacts" / "robustness_conditions.csv"
    )
    assert saved_report["robustness_trust"]["audit_digests"]["robustness_summary_json"] == _sha256(
        run_dir / "artifacts" / "robustness_summary.json"
    )
