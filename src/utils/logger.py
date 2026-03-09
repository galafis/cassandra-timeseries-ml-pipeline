"""
Structured logging configuration for the timeseries pipeline.

Provides JSON-formatted logging with contextual fields for
tracing pipeline execution across distributed components.
"""

import logging
import logging.handlers
import json
import sys
import os
from datetime import datetime, timezone
from typing import Optional


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging output."""

    def __init__(self, service_name: str = "timeseries-pipeline"):
        super().__init__()
        self.service_name = service_name

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "service": self.service_name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
        }

        if hasattr(record, "series_id"):
            log_entry["series_id"] = record.series_id

        if hasattr(record, "operation"):
            log_entry["operation"] = record.operation

        if hasattr(record, "duration_ms"):
            log_entry["duration_ms"] = record.duration_ms

        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
            }

        return json.dumps(log_entry, default=str)


class PipelineLogger:
    """
    Centralized logger for the timeseries ML pipeline.

    Provides structured JSON logging with support for contextual
    fields such as series_id and operation tracking.
    """

    _loggers: dict = {}

    @classmethod
    def get_logger(
        cls,
        name: str,
        level: Optional[str] = None,
        log_file: Optional[str] = None,
    ) -> logging.Logger:
        """
        Get or create a logger instance.

        Args:
            name: Logger name, typically the module path.
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
            log_file: Optional file path for log output.

        Returns:
            Configured logging.Logger instance.
        """
        if name in cls._loggers:
            return cls._loggers[name]

        log_level = getattr(
            logging,
            (level or os.getenv("LOG_LEVEL", "INFO")).upper(),
        )

        logger = logging.getLogger(name)
        logger.setLevel(log_level)
        logger.propagate = False

        if not logger.handlers:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(log_level)
            console_handler.setFormatter(JSONFormatter())
            logger.addHandler(console_handler)

            if log_file:
                file_handler = logging.handlers.RotatingFileHandler(
                    log_file,
                    maxBytes=50 * 1024 * 1024,
                    backupCount=5,
                )
                file_handler.setLevel(log_level)
                file_handler.setFormatter(JSONFormatter())
                logger.addHandler(file_handler)

        cls._loggers[name] = logger
        return logger

    @staticmethod
    def with_context(
        logger: logging.Logger,
        message: str,
        level: str = "info",
        **context,
    ) -> None:
        """
        Log a message with additional context fields.

        Args:
            logger: The logger instance.
            message: Log message text.
            level: Log level string.
            **context: Additional context key-value pairs attached to the log.
        """
        extra = {k: v for k, v in context.items()}
        log_func = getattr(logger, level.lower(), logger.info)
        record_factory = logging.getLogRecordFactory()

        original_factory = logging.getLogRecordFactory()

        def custom_factory(*args, **kwargs):
            record = original_factory(*args, **kwargs)
            for k, v in extra.items():
                setattr(record, k, v)
            return record

        logging.setLogRecordFactory(custom_factory)
        log_func(message)
        logging.setLogRecordFactory(record_factory)


def get_logger(name: str, **kwargs) -> logging.Logger:
    """Convenience function to get a pipeline logger."""
    return PipelineLogger.get_logger(name, **kwargs)
