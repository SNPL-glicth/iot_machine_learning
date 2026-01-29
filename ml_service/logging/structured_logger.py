"""Structured logging for ML service.

Provides JSON-formatted logs with context for observability.
ISO 27001 compliant - no sensitive data in logs.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add extra fields if present
        if hasattr(record, "extra_data"):
            log_data["data"] = record.extra_data
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, ensure_ascii=False, default=str)


class StructuredLogger(logging.Logger):
    """Logger with structured logging support."""
    
    def _log_with_data(
        self,
        level: int,
        msg: str,
        data: Optional[dict] = None,
        *args,
        **kwargs,
    ) -> None:
        if data:
            kwargs.setdefault("extra", {})["extra_data"] = data
        super().log(level, msg, *args, **kwargs)
    
    def info_with_data(self, msg: str, data: Optional[dict] = None, *args, **kwargs):
        self._log_with_data(logging.INFO, msg, data, *args, **kwargs)
    
    def warning_with_data(self, msg: str, data: Optional[dict] = None, *args, **kwargs):
        self._log_with_data(logging.WARNING, msg, data, *args, **kwargs)
    
    def error_with_data(self, msg: str, data: Optional[dict] = None, *args, **kwargs):
        self._log_with_data(logging.ERROR, msg, data, *args, **kwargs)


@dataclass
class LogContext:
    """Context manager for adding context to logs."""
    
    operation: str
    sensor_id: Optional[int] = None
    device_id: Optional[int] = None
    extra: dict = field(default_factory=dict)
    _start_time: float = field(default=0.0, init=False)
    
    def __enter__(self) -> "LogContext":
        self._start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        duration_ms = (time.time() - self._start_time) * 1000
        self.extra["duration_ms"] = round(duration_ms, 2)
    
    def to_dict(self) -> dict:
        result = {
            "operation": self.operation,
            **self.extra,
        }
        if self.sensor_id is not None:
            result["sensor_id"] = self.sensor_id
        if self.device_id is not None:
            result["device_id"] = self.device_id
        return result


def configure_logging(
    level: str = "INFO",
    json_format: bool = True,
) -> None:
    """Configure logging for the ML service.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        json_format: If True, use JSON format. If False, use standard format.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Set custom logger class
    logging.setLoggerClass(StructuredLogger)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)
    
    if json_format:
        handler.setFormatter(StructuredFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
        ))
    
    root_logger.addHandler(handler)


def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger instance.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        StructuredLogger instance
    """
    return logging.getLogger(name)


@contextmanager
def log_operation(
    logger: logging.Logger,
    operation: str,
    sensor_id: Optional[int] = None,
    **extra,
):
    """Context manager for logging operations with timing.
    
    Usage:
        with log_operation(logger, "predict", sensor_id=123) as ctx:
            # do work
            ctx.extra["result"] = "success"
    """
    ctx = LogContext(operation=operation, sensor_id=sensor_id, extra=extra)
    
    logger.info(
        "[%s] Starting operation",
        operation,
        extra={"extra_data": ctx.to_dict()},
    )
    
    try:
        with ctx:
            yield ctx
        
        logger.info(
            "[%s] Completed in %.2fms",
            operation,
            ctx.extra.get("duration_ms", 0),
            extra={"extra_data": ctx.to_dict()},
        )
    except Exception as e:
        ctx.extra["error"] = str(e)
        logger.error(
            "[%s] Failed: %s",
            operation,
            str(e),
            extra={"extra_data": ctx.to_dict()},
            exc_info=True,
        )
        raise
