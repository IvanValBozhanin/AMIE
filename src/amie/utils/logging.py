"""Structured logging configuration utilities."""

from __future__ import annotations

import logging
import os
from uuid import uuid4

from typing import cast

import structlog
from structlog.contextvars import bind_contextvars, merge_contextvars
from structlog.processors import JSONRenderer, TimeStamper
from structlog.stdlib import BoundLogger, LoggerFactory
from structlog.stdlib import add_log_level, filter_by_level

__all__ = ["configure_logging", "get_logger", "logger"]


def _parse_log_level(level_name: str) -> int:
    """Translate environment-provided level into logging module constant."""
    try:
        return getattr(logging, level_name.upper())
    except AttributeError:
        return logging.INFO


def configure_logging(*, run_id: str | None = None, level: str | None = None) -> BoundLogger:
    """Configure structlog and return a bound logger enriched with a run identifier."""
    log_level_name = cast(str, level or os.getenv("LOG_LEVEL", "INFO"))
    log_level = _parse_log_level(log_level_name)

    logging.basicConfig(
        level=log_level,
        format="%(message)s",
    )

    structlog.configure(
        processors=[
            merge_contextvars,
            filter_by_level,
            TimeStamper(fmt="iso"),
            add_log_level,
            structlog.processors.EventRenamer("event"),
            structlog.processors.dict_tracebacks,
            JSONRenderer(),
        ],
        context_class=dict,
        wrapper_class=BoundLogger,
        logger_factory=LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    bind_contextvars(run_id=run_id or os.getenv("RUN_ID", uuid4().hex))
    return structlog.get_logger()


def get_logger() -> BoundLogger:
    """Return the root structlog logger with current context bindings."""
    return structlog.get_logger()


# Configure module-level logger on import so users can log immediately.
logger = configure_logging()
