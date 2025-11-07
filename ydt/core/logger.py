"""
Logging configuration for YDT

Provides consistent logging across all modules.
"""

import logging
import sys
from typing import Optional


def setup_logger(
    name: str = "ydt",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Setup logger with consistent formatting

    Args:
        name: Logger name
        level: Logging level (default: INFO)
        log_file: Optional log file path
        format_string: Optional custom format string

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers.clear()

    # Default format
    if format_string is None:
        format_string = "%(asctime)s-%(name)s-%(levelname)s-%(lineno)d- %(message)s"

    formatter = logging.Formatter(format_string)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "ydt") -> logging.Logger:
    """
    Get or create logger

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger
