"""Project logging helpers.

Keep the public API tiny: many modules want a consistent way to map a simple
`verbose` integer to a standard `logging` level without configuring global
handlers or formats.
"""

from __future__ import annotations

import logging


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

