from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from pyimgano.train_progress import TrainProgressReporter


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _format_float(value: Any, *, digits: int = 6) -> str | None:
    parsed = _safe_float(value)
    if parsed is None:
        return None
    return f"{parsed:.{digits}f}"


def _format_bool(value: Any) -> str:
    return "True" if bool(value) else "False"


def _format_int(value: Any) -> str | None:
    try:
        return str(int(value))
    except Exception:
        return None


def _format_resize(value: Any) -> str | None:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) == 2:
            try:
                return f"{int(value[0])}x{int(value[1])}"
            except Exception:
                return "x".join(str(item) for item in value)
    if value is None:
        return None
    return str(value)


def _comma_join(values: Sequence[str]) -> str:
    items = [str(item) for item in values if str(item)]
    return ",".join(items) if items else "none"


def _format_column_row(*values: object, width: int = 12) -> str:
    return "".join(f"{str(value):>{width}}" for value in values).rstrip()


def _format_duration(value: Any) -> str | None:
    parsed = _safe_float(value)
    if parsed is None:
        return None
    if parsed < 60.0:
        return f"{parsed:.1f}s"
    minutes = int(parsed // 60.0)
    seconds = int(round(parsed % 60.0))
    if seconds == 60:
        minutes += 1
        seconds = 0
    return f"{minutes}m{seconds:02d}s"


def _format_rate(value: Any) -> str | None:
    parsed = _safe_float(value)
    if parsed is None:
        return None
    return f"{parsed:.1f}"


def _format_progress_bar(epoch: int, total_epochs: int | None, *, width: int = 10) -> str:
    if total_epochs is None or total_epochs <= 0:
        return "-" * width
    bounded_epoch = max(0, min(int(epoch), int(total_epochs)))
    filled = int(round((bounded_epoch / int(total_epochs)) * width))
    filled = max(0, min(width, filled))
    return ("█" * filled) + ("-" * (width - filled))


def _format_badge(label: str, category: str | None = None) -> str:
    if category:
        return f"[{label} {category}]"
    return f"[{label}]"


def _append_kv(bits: list[str], key: str, value: Any) -> None:
    if value is None:
        return
    bits.append(f"{key}={value}")


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _format_data_ref(*, dataset: Any, category: Any) -> str | None:
    if dataset is None:
        return None
    dataset_name = str(dataset)
    if not dataset_name:
        return None
    category_name = str(category) if category is not None else ""
    return f"{dataset_name}/{category_name}" if category_name else dataset_name


def _format_sequence_count(value: Any) -> str | None:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return str(len(value))
    return None


def _format_table_line(
    *values: object,
    widths: Sequence[int] | None = None,
    alignments: str = ">",
) -> str:
    cells = [str(value) for value in values]
    if widths is None:
        widths = tuple(12 for _ in cells)
    parts: list[str] = []
    for idx, cell in enumerate(cells):
        width = int(widths[idx]) if idx < len(widths) else int(widths[-1])
        align = alignments[idx] if idx < len(alignments) else alignments[-1]
        parts.append(f"{cell:{align}{width}}")
    return " ".join(parts).rstrip()


def _stringify_cell(value: Any, *, default: str = "-") -> str:
    if value is None:
        return default
    return str(value)


def _mapping_total(value: Mapping[str, Any] | None) -> str:
    if not value:
        return "-"
    total = 0
    seen = False
    for key in ("train", "val", "test"):
        raw = value.get(key)
        if raw is None:
            continue
        total += int(raw)
        seen = True
    return str(total) if seen else "-"


def _format_metric_cell(value: Any, *, digits: int = 6) -> str:
    formatted = _format_float(value, digits=digits)
    return formatted if formatted is not None else "-"


def _format_live_token(label: str, value: str, *, width: int) -> str:
    return f"{label} {str(value):>{width}}"


def _training_epoch_live_line(
    *,
    category: str,
    epoch: int,
    total_epochs: int | None,
    loss: str,
    lr: str,
    train_count: str,
    eta_s: str,
    items_per_s: str,
) -> str:
    total_epochs_int = int(total_epochs) if total_epochs is not None else None
    percent = (
        int(round((int(epoch) / total_epochs_int) * 100))
        if total_epochs_int not in (None, 0)
        else 0
    )
    progress_bar = _format_progress_bar(int(epoch), total_epochs_int, width=10)
    return (
        f"[TRAIN {category}] "
        f"{percent:>3}%|{progress_bar}| "
        f"{int(epoch)}/{total_epochs_int if total_epochs_int is not None else '?'} "
        f"{_format_live_token('loss', loss, width=6)} "
        f"{_format_live_token('lr', lr, width=6)} "
        f"{_format_live_token('n', train_count, width=3)} "
        f"{_format_live_token('eta', eta_s, width=4)} "
        f"{_format_live_token('ips', items_per_s, width=4)}"
    )


def _training_end_bits(report_map: Mapping[str, Any]) -> list[str]:
    timing = dict(report_map.get("timing", {}))
    detector_state = dict(report_map.get("detector_training_state", {}))
    line_bits = ["stage=training_complete"]
    epochs_completed = detector_state.get("epochs_completed", None)
    if epochs_completed is not None:
        line_bits.append(f"epochs={int(epochs_completed)}")
    steps_completed = detector_state.get("steps_completed", None)
    if steps_completed is not None:
        line_bits.append(f"steps={int(steps_completed)}")
    fit_s_raw = _safe_float(timing.get("fit_s"))
    fit_s = None if fit_s_raw is None else f"{fit_s_raw:.3f}"
    if fit_s is not None:
        line_bits.append(f"fit_s={fit_s}")
    if fit_s_raw is not None and epochs_completed not in (None, 0):
        line_bits.append(f"epoch_s={fit_s_raw / int(epochs_completed):.3f}")
    best_loss = _format_float(detector_state.get("best_loss"), digits=4)
    if best_loss is not None:
        line_bits.append(f"best_loss={best_loss}")
    last_lr = _format_float(detector_state.get("last_lr"), digits=4)
    if last_lr is not None:
        line_bits.append(f"last_lr={last_lr}")
    stop_reason = detector_state.get("stop_reason", None)
    if stop_reason is not None:
        line_bits.append(f"stop_reason={stop_reason}")
    return line_bits


def _run_end_metric_bits(report: Mapping[str, Any]) -> list[str]:
    bits: list[str] = []
    mean_metrics = report.get("mean_metrics", None)
    if isinstance(mean_metrics, Mapping):
        for key in ("auroc", "average_precision"):
            value = _format_float(mean_metrics.get(key))
            if value is not None:
                bits.append(f"mean_{key}={value}")
        return bits

    results = report.get("results", None)
    if not isinstance(results, Mapping):
        return bits
    for key in ("auroc", "average_precision"):
        value = _format_float(results.get(key))
        if value is not None:
            bits.append(f"{key}={value}")
    return bits


def _run_end_artifact_lines(
    *,
    run_dir: str | None,
    per_image_jsonl: bool,
    categories: Sequence[str],
    artifacts: Mapping[str, str],
) -> list[tuple[str, str]]:
    if run_dir is None:
        return []

    artifact_lines: list[tuple[str, str]] = []
    for kind, rel_path in (
        ("report", "report.json"),
        ("config", "config.json"),
        ("environment", "environment.json"),
    ):
        candidate = Path(run_dir) / rel_path
        if candidate.exists():
            artifact_lines.append((kind, str(candidate)))
    if per_image_jsonl and len(categories) == 1:
        candidate = Path(run_dir) / "categories" / categories[0] / "per_image.jsonl"
        if candidate.exists():
            artifact_lines.append(("per_image", str(candidate)))
    for kind, path in artifacts.items():
        artifact_lines.append((kind, str(path)))
    return artifact_lines


def _resolve_run_relative_path(path: Any, *, run_dir: str | None) -> str:
    raw = str(path)
    if not raw:
        return raw
    candidate = Path(raw)
    if candidate.is_absolute() or run_dir is None:
        return raw
    return str(Path(run_dir) / candidate)


def emit_dry_run_summary(payload: Mapping[str, Any]) -> None:
    config = dict(_as_mapping(payload.get("config", {})))
    dataset = dict(_as_mapping(config.get("dataset", {})))
    model = dict(_as_mapping(config.get("model", {})))
    output = dict(_as_mapping(config.get("output", {})))
    training = dict(_as_mapping(config.get("training", {})))

    run_bits: list[str] = ["engine=dry-run"]
    _append_kv(run_bits, "recipe", config.get("recipe"))
    _append_kv(run_bits, "model", model.get("name"))
    _append_kv(
        run_bits,
        "data",
        _format_data_ref(dataset=dataset.get("name"), category=dataset.get("category")),
    )

    cfg_bits: list[str] = []
    _append_kv(cfg_bits, "imgsz", _format_resize(dataset.get("resize")))
    _append_kv(cfg_bits, "device", model.get("device"))
    _append_kv(cfg_bits, "epochs", training.get("epochs"))
    _append_kv(cfg_bits, "batch", training.get("batch_size"))
    _append_kv(cfg_bits, "workers", training.get("num_workers"))

    opt_bits: list[str] = []
    _append_kv(opt_bits, "optimizer", training.get("optimizer_name"))
    _append_kv(opt_bits, "scheduler", training.get("scheduler_name"))
    _append_kv(opt_bits, "criterion", training.get("criterion_name"))

    tracker = training.get("tracker_backend")
    if tracker:
        _append_kv(opt_bits, "tracker", tracker)

    callbacks = training.get("callbacks")
    if isinstance(callbacks, Sequence) and not isinstance(callbacks, (str, bytes)) and callbacks:
        _append_kv(opt_bits, "callbacks", _comma_join([str(item) for item in callbacks]))

    out_bits: list[str] = []
    _append_kv(out_bits, "save_run", _format_bool(output.get("save_run")))
    _append_kv(out_bits, "output_dir", output.get("output_dir"))
    _append_kv(out_bits, "input", dataset.get("input_mode"))
    _append_kv(out_bits, "manifest", dataset.get("manifest_path"))

    print("Dry Run Summary")
    print(f"{_format_badge('RUN')} " + " ".join(run_bits))
    print(f"{_format_badge('CFG')} " + " ".join(cfg_bits))
    if opt_bits:
        print(f"{_format_badge('OPT')} " + " ".join(opt_bits))
    print(f"{_format_badge('OUT')} " + " ".join(out_bits))


def emit_preflight_summary(payload: Mapping[str, Any]) -> None:
    preflight = dict(_as_mapping(payload.get("preflight", {})))
    summary = dict(_as_mapping(preflight.get("summary", {})))
    issues = [dict(item) for item in preflight.get("issues", []) if isinstance(item, Mapping)]
    dataset_readiness = dict(_as_mapping(preflight.get("dataset_readiness", {})))
    error_count = sum(1 for item in issues if str(item.get("severity")) == "error")
    warning_count = sum(1 for item in issues if str(item.get("severity")) == "warning")
    info_count = sum(1 for item in issues if str(item.get("severity")) == "info")

    run_bits: list[str] = ["engine=preflight"]
    _append_kv(run_bits, "dataset", preflight.get("dataset"))
    _append_kv(run_bits, "category", preflight.get("category"))

    cfg_bits: list[str] = []
    split_policy = _as_mapping(summary.get("split_policy"))
    split_mode = split_policy.get("mode")
    split_scope = split_policy.get("scope")
    if split_mode or split_scope:
        if split_mode and split_scope:
            _append_kv(cfg_bits, "split", f"{split_mode}/{split_scope}")
        else:
            _append_kv(cfg_bits, "split", split_mode or split_scope)
    _append_kv(cfg_bits, "seed", split_policy.get("seed"))
    _append_kv(
        cfg_bits,
        "test_normal_fraction",
        _format_float(split_policy.get("test_normal_fraction"), digits=3),
    )
    manifest_state = _as_mapping(summary.get("manifest"))
    if "ok" in manifest_state:
        _append_kv(cfg_bits, "manifest_ok", _format_bool(manifest_state.get("ok")))
    elif "ok" in summary:
        _append_kv(cfg_bits, "dataset_ok", _format_bool(summary.get("ok")))

    data_bits: list[str] = []
    _append_kv(data_bits, "manifest", summary.get("manifest_path"))
    _append_kv(data_bits, "root", summary.get("root_fallback") or summary.get("dataset_root"))
    _append_kv(data_bits, "categories", _format_sequence_count(summary.get("categories")))

    counts = _as_mapping(summary.get("counts"))
    assigned_counts = _as_mapping(summary.get("assigned_counts"))
    explicit_by_split = _as_mapping(counts.get("explicit_by_split"))
    explicit_test_labels = _as_mapping(counts.get("explicit_test_labels"))
    mask_coverage = _as_mapping(summary.get("mask_coverage"))

    print("Preflight Summary")
    print(f"{_format_badge('PREFLIGHT')} " + " ".join(run_bits))
    if cfg_bits:
        print(f"{_format_badge('CFG')} " + " ".join(cfg_bits))
    if data_bits:
        print(
            f"{_format_badge('DATA', str(preflight.get('category') or ''))} " + " ".join(data_bits)
        )
    if counts or assigned_counts:
        print(
            _format_table_line(
                "Scope",
                "Total",
                "Train",
                "Val",
                "Test",
                "Cal",
                "Normal",
                "Anomaly",
                widths=(10, 7, 7, 5, 6, 5, 8, 9),
            )
        )
        print(
            _format_table_line(
                "Explicit",
                _stringify_cell(counts.get("total")),
                _stringify_cell(explicit_by_split.get("train")),
                _stringify_cell(explicit_by_split.get("val")),
                _stringify_cell(explicit_by_split.get("test")),
                "-",
                _stringify_cell(explicit_test_labels.get("normal")),
                _stringify_cell(explicit_test_labels.get("anomaly")),
                widths=(10, 7, 7, 5, 6, 5, 8, 9),
            )
        )
        print(
            _format_table_line(
                "Assigned",
                _mapping_total(assigned_counts),
                _stringify_cell(assigned_counts.get("train")),
                _stringify_cell(assigned_counts.get("val")),
                _stringify_cell(assigned_counts.get("test")),
                _stringify_cell(assigned_counts.get("calibration")),
                "-",
                "-",
                widths=(10, 7, 7, 5, 6, 5, 8, 9),
            )
        )
        if mask_coverage:
            anomaly_total = mask_coverage.get("anomaly_test_total")
            mask_exists = mask_coverage.get("anomaly_test_mask_exists")
            print(
                " ".join(
                    [
                        "Pixel",
                        f"enabled={_format_bool(_as_mapping(summary.get('pixel_metrics')).get('enabled'))}",
                        f"masks={_stringify_cell(mask_exists, default='0')}/{_stringify_cell(anomaly_total, default='0')}",
                    ]
                )
            )
    print(
        f"{_format_badge('CHECK')} "
        + " ".join(
            [
                f"errors={error_count}",
                f"warnings={warning_count}",
                f"infos={info_count}",
            ]
        )
    )
    if dataset_readiness:
        readiness_bits = [f"dataset_readiness={dataset_readiness.get('status')}"]
        issue_codes = [str(item) for item in dataset_readiness.get("issue_codes", []) if str(item)]
        if issue_codes:
            readiness_bits.append(f"dataset_issue_codes={','.join(issue_codes)}")
        print(f"{_format_badge('READY')} " + " ".join(readiness_bits))
    if issues:
        print(f"{'Severity':<10} {'Code':<28} Message")
    for issue in issues:
        severity = str(issue.get("severity") or "info")
        code = str(issue.get("code") or "UNKNOWN")
        message = str(issue.get("message") or "")
        print(f"{severity:<10} {code:<28} {message}")


class TrainConsoleReporter(TrainProgressReporter):
    def __init__(self) -> None:
        self._current_category: str | None = None
        self._run_dir: str | None = None
        self._seen_epochs: set[tuple[str | None, int]] = set()
        self._artifacts: dict[str, str] = {}
        self._device: str = "-"
        self._train_counts: dict[str, int] = {}
        self._resize: str = "-"
        self._input_mode: str = "-"
        self._per_image_jsonl: bool = False
        self._categories: list[str] = []
        self._multi_category: bool = False
        self._live_line_active: bool = False
        self._live_line_width: int = 0

    def _finish_live_line(self) -> None:
        if self._live_line_active:
            print()
            self._live_line_active = False
            self._live_line_width = 0

    def _emit_line(self, line: str) -> None:
        self._finish_live_line()
        print(line)

    def _emit_live_line(self, line: str) -> None:
        padding = ""
        line_width = len(line)
        if line_width < self._live_line_width:
            padding = " " * (self._live_line_width - line_width)
        print(f"\r{line}{padding}", end="", flush=True)
        self._live_line_active = True
        self._live_line_width = max(self._live_line_width, line_width)

    def _emit_metric_table(
        self,
        *,
        badge: str,
        category: str,
        headers: Sequence[str],
        values: Sequence[str],
        widths: Sequence[int],
        value_alignments: str | None = None,
    ) -> None:
        self._emit_line(f"{_format_badge(badge, str(category))}")
        self._emit_line(
            _format_table_line(
                *headers,
                widths=widths,
                alignments="<" * len(headers),
            )
        )
        self._emit_line(
            _format_table_line(
                *values,
                widths=widths,
                alignments=value_alignments or (">" * len(values)),
            )
        )

    def on_run_start(self, *, config: Any, request: Any) -> None:
        dataset_cfg = getattr(config, "dataset", None)
        model_cfg = getattr(config, "model", None)
        output_cfg = getattr(config, "output", None)
        training_cfg = getattr(config, "training", None)
        dataset_name = getattr(dataset_cfg, "name", None)
        dataset_category = getattr(dataset_cfg, "category", None)
        self._device = str(getattr(model_cfg, "device", None) or "-")
        self._resize = _format_resize(getattr(dataset_cfg, "resize", None)) or "-"
        self._input_mode = str(getattr(dataset_cfg, "input_mode", None) or "-")
        self._per_image_jsonl = bool(getattr(output_cfg, "per_image_jsonl", False))
        self._emit_line("Train Run Summary")
        run_bits: list[str] = [
            "engine=train",
            f"recipe={getattr(config, 'recipe', None)}",
            f"model={getattr(model_cfg, 'name', None)}",
            f"data={dataset_name}/{dataset_category}",
        ]
        cfg_bits: list[str] = [f"imgsz={self._resize}", f"device={self._device}"]
        _append_kv(cfg_bits, "epochs", getattr(training_cfg, "epochs", None))
        _append_kv(cfg_bits, "batch", getattr(training_cfg, "batch_size", None))
        _append_kv(cfg_bits, "workers", getattr(training_cfg, "num_workers", None))

        opt_bits: list[str] = []
        _append_kv(opt_bits, "optimizer", getattr(training_cfg, "optimizer_name", None))
        _append_kv(opt_bits, "scheduler", getattr(training_cfg, "scheduler_name", None))
        _append_kv(opt_bits, "criterion", getattr(training_cfg, "criterion_name", None))

        out_bits: list[str] = [
            f"config={getattr(request, 'config_path', None)}",
            f"save_run={_format_bool(getattr(output_cfg, 'save_run', False))}",
            f"input={self._input_mode}",
            f"per_image_jsonl={_format_bool(self._per_image_jsonl)}",
            f"export_infer_config={_format_bool(getattr(request, 'export_infer_config', False))}",
            f"export_deploy_bundle={_format_bool(getattr(request, 'export_deploy_bundle', False))}",
        ]

        self._emit_line(f"{_format_badge('RUN')} " + " ".join(run_bits))
        self._emit_line(f"{_format_badge('CFG')} " + " ".join(cfg_bits))
        if opt_bits:
            self._emit_line(f"{_format_badge('OPT')} " + " ".join(opt_bits))
        self._emit_line(f"{_format_badge('OUT')} " + " ".join(out_bits))

    def on_run_context(self, *, run_dir: str | None) -> None:
        self._run_dir = run_dir
        if run_dir is not None:
            self._emit_line(f"{_format_badge('OUT')} save_dir={run_dir}")

    def on_category_start(
        self,
        *,
        category: str,
        index: int | None = None,
        total: int | None = None,
    ) -> None:
        self._current_category = str(category)
        if self._current_category not in self._categories:
            self._categories.append(self._current_category)
        self._multi_category = bool(total is not None and total > 1)
        if total is not None and total > 1:
            self._emit_line(f"{_format_badge('DATA', str(category))} category={index}/{total}")

    def on_dataset_loaded(
        self,
        *,
        category: str,
        train_count: int,
        calibration_count: int,
        test_count: int,
        anomaly_count: int,
        pixel_metrics_enabled: bool | None,
        pixel_metrics_reason: str | None = None,
    ) -> None:
        self._train_counts[str(category)] = int(train_count)
        show_pixel = bool(pixel_metrics_reason) or pixel_metrics_enabled is not True
        headers = ["Train", "Cal", "Test", "Anom", "imgsz", "Input"]
        values = [
            str(int(train_count)),
            str(int(calibration_count)),
            str(int(test_count)),
            str(int(anomaly_count)),
            str(self._resize),
            str(self._input_mode),
        ]
        widths = [8, 6, 6, 6, 8, 10]
        alignments = ">>>>><"
        if show_pixel:
            headers.append("Pixel")
            if pixel_metrics_enabled is None:
                pixel_value = "-"
            elif pixel_metrics_enabled:
                pixel_value = "on"
            else:
                pixel_value = "off"
            values.append(pixel_value)
            widths.append(6)
            alignments += "<"
        self._emit_metric_table(
            badge="DATA",
            category=str(category),
            headers=tuple(headers),
            values=tuple(values),
            widths=tuple(widths),
            value_alignments=alignments,
        )
        if pixel_metrics_reason:
            self._emit_line(f"{_format_badge('DATA', str(category))} note={pixel_metrics_reason}")

    def on_training_start(
        self,
        *,
        category: str,
        enabled: bool,
        fit_kwargs: Mapping[str, Any] | None = None,
        tracker_backend: str | None = None,
        callback_names: list[str] | None = None,
    ) -> None:
        fit_map = dict(fit_kwargs) if isinstance(fit_kwargs, Mapping) else {}
        epochs = fit_map.get("epochs", None)
        summary_bits = [
            f"status={'enabled' if enabled else 'disabled'}",
            f"epochs={epochs}",
        ]
        if tracker_backend:
            summary_bits.append(f"tracker={tracker_backend}")
        if callback_names:
            summary_bits.append(f"callbacks={_comma_join(callback_names)}")
        self._emit_line(f"{_format_badge('TRAIN', str(category))} " + " ".join(summary_bits))
        if enabled:
            detail_bits: list[str] = []
            for key, label in (
                ("batch_size", "batch"),
                ("num_workers", "workers"),
                ("optimizer_name", "optimizer"),
                ("scheduler_name", "scheduler"),
                ("criterion_name", "criterion"),
                ("max_steps", "max_steps"),
                ("early_stopping_patience", "early_stop"),
            ):
                value = fit_map.get(key, None)
                if value is None:
                    continue
                detail_bits.append(f"{label}={value}")
            if detail_bits:
                self._emit_line(f"{_format_badge('TRAIN', str(category))} " + " ".join(detail_bits))
        if enabled:
            self._emit_line(
                _format_column_row(
                    "Epoch",
                    "Device",
                    "loss",
                    "lr",
                    "Train",
                    "Category",
                    "Time",
                    "ETA",
                    "items/s",
                )
            )

    def on_training_epoch(
        self,
        *,
        epoch: int,
        total_epochs: int | None,
        metrics: Mapping[str, Any],
        live: bool = False,
    ) -> None:
        key = (self._current_category, int(epoch))
        if key in self._seen_epochs:
            return
        self._seen_epochs.add(key)
        category = self._current_category or "-"
        loss = _format_float(metrics.get("loss"), digits=4) or "-"
        lr = _format_float(metrics.get("lr"), digits=4) or "-"
        train_count = (
            _format_int(metrics.get("train_items"))
            or _format_int(self._train_counts.get(category))
            or "-"
        )
        epoch_s = _format_duration(metrics.get("epoch_s")) or "-"
        eta_s = _format_duration(metrics.get("eta_s")) or "-"
        items_per_s = _format_rate(metrics.get("items_per_s")) or "-"
        if live:
            self._emit_live_line(
                _training_epoch_live_line(
                    category=category,
                    epoch=int(epoch),
                    total_epochs=total_epochs,
                    loss=loss,
                    lr=lr,
                    train_count=train_count,
                    eta_s=eta_s,
                    items_per_s=items_per_s,
                )
            )
            return

        self._emit_line(
            _format_column_row(
                f"{int(epoch)}/{int(total_epochs) if total_epochs is not None else '?'}",
                self._device,
                loss,
                lr,
                train_count,
                category,
                epoch_s,
                eta_s,
                items_per_s,
            )
        )

    def on_training_end(
        self,
        *,
        category: str,
        report: Mapping[str, Any] | None,
        checkpoint_meta: Mapping[str, Any] | None = None,
    ) -> None:
        report_map = dict(report) if isinstance(report, Mapping) else {}
        line_bits = _training_end_bits(report_map)
        self._emit_line(f"{_format_badge('DONE', str(category))} " + " ".join(line_bits))
        if isinstance(checkpoint_meta, Mapping) and checkpoint_meta.get("path"):
            checkpoint_path = _resolve_run_relative_path(
                checkpoint_meta.get("path"),
                run_dir=self._run_dir,
            )
            checkpoint_key = f"checkpoint.{category}" if self._multi_category else "checkpoint"
            self._artifacts[checkpoint_key] = checkpoint_path
        for key in ("callback_warnings", "tracker_warnings"):
            for item in report_map.get(key, []) or []:
                self._emit_line(f"warning={item}")

    def on_calibration_end(
        self,
        *,
        category: str,
        threshold: float,
        quantile: float,
        source: str,
        score_summary: Mapping[str, Any] | None = None,
    ) -> None:
        score_mean = None
        if isinstance(score_summary, Mapping):
            score_mean = score_summary.get("mean")
        self._emit_metric_table(
            badge="CAL",
            category=str(category),
            headers=("Threshold", "Quantile", "ScoreMean", "Source"),
            values=(
                _format_metric_cell(threshold),
                _format_metric_cell(quantile),
                _format_metric_cell(score_mean),
                str(source),
            ),
            widths=(10, 10, 10, 14),
            value_alignments=">>><",
        )

    def on_evaluation_end(
        self,
        *,
        category: str,
        results: Mapping[str, Any],
        dataset_summary: Mapping[str, Any] | None = None,
    ) -> None:
        pixel_metrics = results.get("pixel_metrics", None)
        pixel_map = pixel_metrics if isinstance(pixel_metrics, Mapping) else {}
        self._emit_metric_table(
            badge="VAL",
            category=str(category),
            headers=("AUROC", "AP", "pAUROC", "pAP", "AUPRO", "SegF1"),
            values=(
                _format_metric_cell(results.get("auroc")),
                _format_metric_cell(results.get("average_precision")),
                _format_metric_cell(pixel_map.get("pixel_auroc")),
                _format_metric_cell(pixel_map.get("pixel_average_precision")),
                _format_metric_cell(pixel_map.get("aupro")),
                _format_metric_cell(pixel_map.get("pixel_segf1")),
            ),
            widths=(10, 10, 10, 10, 10, 10),
        )
        if isinstance(dataset_summary, Mapping):
            pixel_summary = dataset_summary.get("pixel_metrics", None)
            if isinstance(pixel_summary, Mapping) and pixel_summary.get("enabled") is False:
                self._emit_line(
                    f"{_format_badge('VAL', str(category))} pixel_metrics_reason={pixel_summary.get('reason')}"
                )

    def on_artifact_written(self, *, kind: str, path: str) -> None:
        resolved_path = _resolve_run_relative_path(path, run_dir=self._run_dir)
        self._artifacts[str(kind)] = resolved_path
        self._emit_line(f"{_format_badge('SAVE')} {kind}={resolved_path}")

    def on_run_end(self, *, report: Mapping[str, Any]) -> None:
        bits = ["status=done"]
        if self._run_dir is not None:
            bits.append(f"save_dir={self._run_dir}")
        bits.extend(_run_end_metric_bits(report))
        self._emit_line(f"{_format_badge('DONE')} " + " ".join(bits))
        if self._run_dir is not None:
            artifact_lines = _run_end_artifact_lines(
                run_dir=self._run_dir,
                per_image_jsonl=self._per_image_jsonl,
                categories=self._categories,
                artifacts=self._artifacts,
            )
            if artifact_lines:
                seen: set[tuple[str, str]] = set()
                for kind, path in artifact_lines:
                    item = (kind, path)
                    if item in seen:
                        continue
                    seen.add(item)
                    self._emit_line(f"{_format_badge('SAVE')} {kind}={path}")
            self._emit_line(
                f"{_format_badge('DONE')} next=pyimgano-runs quality {self._run_dir} --json"
            )


__all__ = [
    "TrainConsoleReporter",
    "emit_dry_run_summary",
    "emit_preflight_summary",
]
