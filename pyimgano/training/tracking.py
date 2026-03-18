from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping, Protocol


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> str:
    return str(value)


def _normalize_tracking_uri(raw: str | Path | None) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    if "://" in text:
        return text
    return Path(text).resolve().as_uri()


class TrainingTracker(Protocol):
    def log_params(self, params: Mapping[str, Any]) -> None:
        ...

    def log_metrics(self, metrics: Mapping[str, float], *, step: int | None = None) -> None:
        ...

    def log_artifact(self, name: str, artifact: Any) -> None:
        ...

    def close(self) -> None:
        ...


class NullTracker:
    def log_params(self, params: Mapping[str, Any]) -> None:  # noqa: ARG002 - interface parity
        return

    def log_metrics(  # noqa: ARG002 - interface parity
        self,
        metrics: Mapping[str, float],
        *,
        step: int | None = None,
    ) -> None:
        return

    def log_artifact(self, name: str, artifact: Any) -> None:  # noqa: ARG002 - interface parity
        return

    def close(self) -> None:
        return


class JsonlTracker:
    def __init__(self, log_dir: str | Path) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.params_path = self.log_dir / "params.json"
        self.metrics_path = self.log_dir / "metrics.jsonl"
        self.artifacts_dir = self.log_dir / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._params: dict[str, Any] = {}

    def log_params(self, params: Mapping[str, Any]) -> None:
        self._params.update(dict(params))
        self.params_path.write_text(
            json.dumps(self._params, indent=2, sort_keys=True, default=_json_default),
            encoding="utf-8",
        )

    def log_metrics(self, metrics: Mapping[str, float], *, step: int | None = None) -> None:
        payload: dict[str, Any] = {
            "timestamp": _utc_now_iso(),
            "metrics": dict(metrics),
        }
        if step is not None:
            payload["step"] = int(step)
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=_json_default))
            handle.write("\n")

    def log_artifact(self, name: str, artifact: Any) -> None:
        artifact_name = str(name).strip()
        if not artifact_name:
            raise ValueError("artifact name must be non-empty")
        target = self.artifacts_dir / artifact_name
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(artifact, (bytes, bytearray)):
            target.write_bytes(bytes(artifact))
            return
        if isinstance(artifact, str):
            target.write_text(artifact, encoding="utf-8")
            return
        target.write_text(json.dumps(artifact, indent=2, default=_json_default), encoding="utf-8")

    def close(self) -> None:
        return


class TensorBoardTracker:
    def __init__(self, *, log_dir: str | Path, run_name: str | None = None) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except Exception as exc:  # noqa: BLE001 - optional dependency boundary
            raise RuntimeError(
                "TensorBoard tracker requires torch with tensorboard support"
            ) from exc

        path = Path(log_dir)
        if run_name:
            path = path / str(run_name)
        path.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(log_dir=path.as_posix())

    def log_params(self, params: Mapping[str, Any]) -> None:
        self._writer.add_text("run/params", json.dumps(dict(params), default=_json_default))
        self._writer.flush()

    def log_metrics(self, metrics: Mapping[str, float], *, step: int | None = None) -> None:
        for key, value in dict(metrics).items():
            try:
                scalar = float(value)
            except Exception:  # noqa: BLE001 - defensive conversion
                continue
            self._writer.add_scalar(str(key), scalar, global_step=step)
        self._writer.flush()

    def log_artifact(self, name: str, artifact: Any) -> None:
        self._writer.add_text(f"artifact/{name}", json.dumps(artifact, default=_json_default))
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()


class WandbTracker:
    def __init__(
        self,
        *,
        project: str,
        run_name: str | None = None,
        mode: str | None = None,
    ) -> None:
        try:
            import wandb
        except Exception as exc:  # noqa: BLE001 - optional dependency boundary
            raise RuntimeError("W&B tracker requires the 'wandb' package") from exc

        init_kwargs: dict[str, Any] = {"project": str(project)}
        if run_name is not None:
            init_kwargs["name"] = str(run_name)
        if mode is not None:
            init_kwargs["mode"] = str(mode)
        self._wandb = wandb
        self._run = wandb.init(**init_kwargs)
        if self._run is None:
            raise RuntimeError("wandb.init(...) returned None")

    def log_params(self, params: Mapping[str, Any]) -> None:
        self._run.config.update(dict(params), allow_val_change=True)

    def log_metrics(self, metrics: Mapping[str, float], *, step: int | None = None) -> None:
        payload = dict(metrics)
        if step is not None:
            self._wandb.log(payload, step=int(step))
            return
        self._wandb.log(payload)

    def log_artifact(self, name: str, artifact: Any) -> None:
        target = Path(name).name
        if not target:
            raise ValueError("artifact name must be non-empty")
        artifact_dir = Path(".wandb-artifacts")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / target
        if isinstance(artifact, (bytes, bytearray)):
            artifact_path.write_bytes(bytes(artifact))
        elif isinstance(artifact, str):
            artifact_path.write_text(artifact, encoding="utf-8")
        else:
            artifact_path.write_text(
                json.dumps(artifact, indent=2, default=_json_default),
                encoding="utf-8",
            )
        logged = self._wandb.Artifact(target, type="pyimgano-artifact")
        logged.add_file(artifact_path.as_posix())
        self._run.log_artifact(logged)

    def close(self) -> None:
        self._run.finish()


class MlflowTracker:
    def __init__(
        self,
        *,
        tracking_uri: str | None = None,
        experiment_name: str | None = None,
        run_name: str | None = None,
    ) -> None:
        try:
            import mlflow
        except Exception as exc:  # noqa: BLE001 - optional dependency boundary
            raise RuntimeError("MLflow tracker requires the 'mlflow' package") from exc

        self._mlflow = mlflow
        if tracking_uri is not None:
            mlflow.set_tracking_uri(str(tracking_uri))
        if experiment_name is not None and str(experiment_name).strip():
            mlflow.set_experiment(str(experiment_name))
        self._run = mlflow.start_run(run_name=(str(run_name) if run_name is not None else None))

    def log_params(self, params: Mapping[str, Any]) -> None:
        payload: dict[str, str] = {}
        for key, value in dict(params).items():
            if value is None:
                continue
            payload[str(key)] = str(value)
        if payload:
            self._mlflow.log_params(payload)

    def log_metrics(self, metrics: Mapping[str, float], *, step: int | None = None) -> None:
        payload: dict[str, float] = {}
        for key, value in dict(metrics).items():
            try:
                payload[str(key)] = float(value)
            except Exception:  # noqa: BLE001 - defensive conversion
                continue
        if not payload:
            return
        if step is not None:
            self._mlflow.log_metrics(payload, step=int(step))
            return
        self._mlflow.log_metrics(payload)

    def log_artifact(self, name: str, artifact: Any) -> None:
        artifact_name = Path(str(name).strip()).name
        if not artifact_name:
            raise ValueError("artifact name must be non-empty")

        with TemporaryDirectory(prefix="pyimgano-mlflow-artifact-") as temp_dir:
            artifact_path = Path(temp_dir) / artifact_name
            if isinstance(artifact, (bytes, bytearray)):
                artifact_path.write_bytes(bytes(artifact))
            elif isinstance(artifact, str):
                artifact_path.write_text(artifact, encoding="utf-8")
            else:
                artifact_path.write_text(
                    json.dumps(artifact, indent=2, default=_json_default),
                    encoding="utf-8",
                )
            self._mlflow.log_artifact(artifact_path.as_posix())

    def close(self) -> None:
        self._mlflow.end_run()


def create_training_tracker(
    backend: str | None,
    *,
    log_dir: str | Path | None = None,
    project: str | None = None,
    run_name: str | None = None,
    mode: str | None = None,
) -> TrainingTracker:
    normalized = "none" if backend is None else str(backend).strip().lower()

    if normalized in {"", "none", "null", "off", "disabled"}:
        return NullTracker()
    if normalized == "jsonl":
        out_dir = Path("./experiments") if log_dir is None else Path(log_dir)
        return JsonlTracker(out_dir)
    if normalized == "tensorboard":
        out_dir = Path("./runs") if log_dir is None else Path(log_dir)
        return TensorBoardTracker(log_dir=out_dir, run_name=run_name)
    if normalized == "wandb":
        resolved_project = str(project).strip() if project is not None else "pyimgano"
        return WandbTracker(project=resolved_project, run_name=run_name, mode=mode)
    if normalized == "mlflow":
        tracking_uri = _normalize_tracking_uri(log_dir)
        experiment_name = str(project).strip() if project is not None else "pyimgano"
        return MlflowTracker(
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
            run_name=run_name,
        )
    raise ValueError(
        "Unsupported training tracker backend: "
        f"{backend!r}. Choose from: none, jsonl, tensorboard, wandb, mlflow."
    )


__all__ = [
    "TrainingTracker",
    "NullTracker",
    "JsonlTracker",
    "TensorBoardTracker",
    "WandbTracker",
    "MlflowTracker",
    "create_training_tracker",
]
