"""Project logging helpers.

Keep the public API tiny: many modules want a consistent way to map a simple
`verbose` integer to a standard `logging` level without configuring global
handlers or formats.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import IO, Any


def verbosity_to_level(verbose: int | bool | None) -> int:
    """Map a user-facing verbosity value to a `logging` level.

    Conventions:
    - 0 / False: warnings only
    - 1 / True / None: info
    - >=2: debug
    """

    if verbose is None:
        return logging.INFO
    if isinstance(verbose, bool):
        return logging.INFO if verbose else logging.WARNING
    v = int(verbose)
    if v <= 0:
        return logging.WARNING
    if v == 1:
        return logging.INFO
    return logging.DEBUG


def get_logger(name: str, *, verbose: int | bool | None = None) -> logging.Logger:
    """Return a logger configured with the desired level.

    This function does *not* add handlers; it only sets the logger level.
    """

    logger = logging.getLogger(name)
    logger.setLevel(verbosity_to_level(verbose))
    return logger


class _JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=True)


def _coerce_level(
    *,
    level: int | str | None,
    verbose: int | bool | None,
) -> int:
    if level is None:
        return verbosity_to_level(verbose)
    if isinstance(level, str):
        candidate = getattr(logging, str(level).strip().upper(), None)
        if isinstance(candidate, int):
            return candidate
        raise ValueError(
            f"Unsupported log level {level!r}; expected one of DEBUG/INFO/WARNING/ERROR/CRITICAL"
        )
    return int(level)


def configure_logging(
    *,
    logger_name: str | None = None,
    level: int | str | None = None,
    verbose: int | bool | None = None,
    json_output: bool = False,
    stream: IO[str] | None = None,
    file_path: str | Path | None = None,
    rotate_bytes: int = 0,
    backup_count: int = 3,
    force: bool = False,
) -> logging.Logger:
    """Initialize a logger with consistent formatting.

    By default, this configures one stream handler on stderr. You can provide
    `file_path` to add a file handler as well.
    """

    logger = logging.getLogger(logger_name) if logger_name else logging.getLogger()
    logger.setLevel(_coerce_level(level=level, verbose=verbose))

    if force:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)

    handlers: list[logging.Handler] = []
    if file_path is not None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if int(rotate_bytes) > 0:
            handlers.append(
                RotatingFileHandler(
                    path.as_posix(),
                    maxBytes=int(rotate_bytes),
                    backupCount=max(1, int(backup_count)),
                    encoding="utf-8",
                )
            )
        else:
            handlers.append(logging.FileHandler(path.as_posix(), encoding="utf-8"))

    # Keep stderr logging by default even when file output is enabled.
    if stream is not None or not handlers:
        handlers.append(logging.StreamHandler(stream))

    formatter: logging.Formatter
    if json_output:
        formatter = _JsonLogFormatter()
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    for handler in handlers:
        handler.setLevel(logger.level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


__all__ = ["verbosity_to_level", "get_logger", "configure_logging"]
