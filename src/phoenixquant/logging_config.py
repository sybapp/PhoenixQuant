"""Logging helpers for PhoenixQuant."""
from __future__ import annotations

import json
import logging
from typing import Iterable


class JsonFormatter(logging.Formatter):
    """Render log records as JSON strings."""

    def format(self, record: logging.LogRecord) -> str:  # pragma: no cover - formatting logic
        standard_attrs = {
            "name",
            "msg",
            "args",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "exc_info",
            "exc_text",
            "stack_info",
            "lineno",
            "funcName",
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "processName",
            "process",
            "message",
        }
        data = {
            "time": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in standard_attrs
        }
        if extras:
            data.update(extras)
        if record.exc_info:
            data["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(data, ensure_ascii=False)


def setup_logging(level: int = logging.INFO, handlers: Iterable[logging.Handler] | None = None) -> None:
    """Configure the root logger with JSON formatting."""

    root = logging.getLogger()
    root.setLevel(level)
    if handlers:
        for handler in list(root.handlers):
            root.removeHandler(handler)
        for handler in handlers:
            handler.setFormatter(JsonFormatter())
            root.addHandler(handler)
    elif not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        root.addHandler(handler)

