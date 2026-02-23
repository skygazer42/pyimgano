from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


_SAFE_CHARS_RE = re.compile(r"[^a-zA-Z0-9._-]+")


def _sanitize_component(text: str) -> str:
    text = str(text).strip()
    text = _SAFE_CHARS_RE.sub("_", text)
    return text.strip("._-") or "run"


def build_run_dir_name(*, dataset: str, model: str, category: str | None = None) -> str:
    """Build a stable run directory name.

    Format: YYYYMMDD_HHMMSS_<dataset>_<model>[_<category>]
    """

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    parts = [ts, _sanitize_component(dataset), _sanitize_component(model)]
    if category is not None:
        parts.append(_sanitize_component(category))
    return "_".join(parts)


def ensure_run_dir(*, output_dir: str | Path | None, name: str) -> Path:
    """Create and return the run directory under `runs/` unless overridden."""

    if output_dir is None:
        base = Path("runs")
        out = base / name
        if out.exists():
            # Avoid silently mixing artifacts from multiple runs that happen to
            # share the same timestamp-derived directory name.
            for i in range(1, 1000):
                candidate = base / f"{name}_{i:03d}"
                if not candidate.exists():
                    out = candidate
                    break
    else:
        out = Path(output_dir)

    out.mkdir(parents=True, exist_ok=True)
    return out


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    report_json: Path
    config_json: Path
    categories_dir: Path


def build_run_paths(run_dir: Path) -> RunPaths:
    return RunPaths(
        run_dir=run_dir,
        report_json=run_dir / "report.json",
        config_json=run_dir / "config.json",
        categories_dir=run_dir / "categories",
    )
