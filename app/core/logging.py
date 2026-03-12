from __future__ import annotations

import logging
from logging.config import dictConfig

from app.core.settings import settings


class _DefaultContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "environment"):
            record.environment = settings.environment
        if not hasattr(record, "service"):
            record.service = settings.project_name
        return True


def setup_logging() -> None:
    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "structured": {
                    "format": (
                        "%(asctime)s | %(levelname)s | %(name)s | "
                        "env=%(environment)s | service=%(service)s | %(message)s"
                    )
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": settings.log_level,
                    "formatter": "structured",
                    "filters": ["default_context"],
                }
            },
            "filters": {"default_context": {"()": _DefaultContextFilter}},
            "root": {"handlers": ["console"], "level": settings.log_level},
        }
    )


class ContextAdapter(logging.LoggerAdapter):
    def process(self, msg: str, kwargs: dict) -> tuple[str, dict]:
        extra = kwargs.setdefault("extra", {})
        extra.setdefault("environment", settings.environment)
        extra.setdefault("service", settings.project_name)
        return msg, kwargs


def get_logger(name: str) -> logging.LoggerAdapter:
    return ContextAdapter(logging.getLogger(name), {})