"""Small, dependency-free application logging setup."""

from __future__ import annotations

import logging

LOGGER_NAME = "nfl_coaching_sim"
_HANDLER_MARKER = "_nfl_coach_handler"


def configure_logging(level: str = "INFO") -> None:
    """Configure concise console logs without changing third-party loggers."""

    logger = logging.getLogger(LOGGER_NAME)
    normalized_level = level.strip().upper()
    logger.setLevel(normalized_level)
    logger.propagate = False
    if any(getattr(handler, _HANDLER_MARKER, False) for handler in logger.handlers):
        return

    handler = logging.StreamHandler()
    setattr(handler, _HANDLER_MARKER, True)
    handler.setFormatter(
        logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] %(name)s => %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    logger.addHandler(handler)

