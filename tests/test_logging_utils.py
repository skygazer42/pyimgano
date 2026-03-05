from __future__ import annotations

import logging


def test_verbosity_to_level_mapping() -> None:
    from pyimgano.utils.logging import verbosity_to_level

    assert verbosity_to_level(None) == logging.INFO
    assert verbosity_to_level(False) == logging.WARNING
    assert verbosity_to_level(True) == logging.INFO
    assert verbosity_to_level(0) == logging.WARNING
    assert verbosity_to_level(1) == logging.INFO
    assert verbosity_to_level(2) == logging.DEBUG
    assert verbosity_to_level(3) == logging.DEBUG


def test_get_logger_sets_level() -> None:
    from pyimgano.utils.logging import get_logger

    logger = get_logger("pyimgano.tests", verbose=2)
    assert logger.level == logging.DEBUG
