from __future__ import annotations

import io
import json
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


def test_configure_logging_writes_json_line() -> None:
    from pyimgano.utils.logging import configure_logging

    stream = io.StringIO()
    logger = configure_logging(
        logger_name="pyimgano.tests.structured",
        verbose=2,
        json_output=True,
        stream=stream,
        force=True,
    )
    logger.info("structured hello")

    payload = json.loads(stream.getvalue().splitlines()[-1])
    assert payload["level"] == "INFO"
    assert payload["logger"] == "pyimgano.tests.structured"
    assert payload["message"] == "structured hello"
