"""
Global Logger Setup Module

This module provides centralized logger configuration with support for:
- File and console output
- Colored console output via ColoredFormatter
- UTC+8 timestamp formatting
- Rotating file handlers

Dependencies:
    - logging: Standard logging library
    - sys: For stderr handling
    - datetime: For timestamp generation
    - logging.handlers: For RotatingFileHandler
    - pathlib: For path operations
    - robocoin_dataset.utils.log_config: For ColoredFormatter and timestamp utilities

Usage:
    Standard usage:
        logger = setup_logger(
            name="my_script",
            log_dir="./logs",
            console_output=True,
        )
    
    With UTC+8 timestamp:
        logger = setup_logger_utc8(
            name="my_script",
            log_dir="./logs",
        )
"""

import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

from robocoin_dataset.utils.log_config import ColoredFormatter, get_utc8_timestamp


def setup_logger(
    name: str,
    log_dir: Path,
    level=logging.INFO,  # noqa: ANN001
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    fmt: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
    console_output: bool = True,
) -> logging.Logger:
    """
    Setup logger with file (with timestamp) and optional console output.

    Input:
        name (str): Logger name
        log_dir (Path): Directory path where log files will be stored
        level (int): Logging level (default: logging.INFO)
        max_bytes (int): Max bytes per rotating log file (default: 10MB)
        backup_count (int): Number of backup log files (default: 5)
        fmt (str): Log message format string
        datefmt (str): Date format string
        console_output (bool): Whether to output logs to console (default: True)

    Output:
        logging.Logger: Configured logger instance

    Logic:
        1. Check if logger already has handlers, return if configured
        2. Set logger level and disable propagation
        3. Create log directory if needed, fallback to console-only if fails
        4. Create file handler with rotating capability
        5. Optionally add console handler
        6. Log initialization message

    Usage:
        Called by main scripts to create loggers with automatic rotating files.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        # Already configured, avoid duplication
        return logger

    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    # Create log directory
    log_dir = Path(log_dir)
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Unable to mkdir log dir {log_dir}: {e}", file=sys.stderr)
        # Fallback: console-only
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        return logger

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filepath = log_dir / f"{name}_{timestamp}.log"

    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_filepath, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    # Console handler (optional)
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.info(f"The log system has been started, log file: {log_filepath.resolve()}")

    return logger


def setup_logger_utc8(
    name: str,
    log_dir: Path,
    level: int = logging.INFO,
    console_output: bool = True,
    colored_console: bool = True,
) -> logging.Logger:
    """
    Setup logger with UTC+8 timestamp and optional colored console output.

    Input:
        name (str): Logger name (used for both logger and log file naming)
        log_dir (Path): Directory path where log files will be stored
        level (int): Logging level (default: logging.INFO)
        console_output (bool): Whether to output logs to console (default: True)
        colored_console (bool): Whether to use colored console formatter (default: True)

    Output:
        logging.Logger: Configured logger instance

    Logic:
        1. Check if logger already has handlers, return if configured
        2. Set logger level and disable propagation
        3. Create log directory if needed, fallback to console-only if fails
        4. Generate UTC+8 timestamp and create log file path
        5. Create file handler with standard formatter
        6. Optionally add console handler with colored formatter
        7. Log initialization message with log file path

    Usage:
        Called by metadata/logging.py and readme/logging.py to setup
        stage-specific loggers with UTC+8 timestamps and colored output.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        # Already configured, avoid duplication
        return logger

    logger.setLevel(level)
    logger.propagate = False

    log_dir = Path(log_dir)
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Unable to create log directory {log_dir}: {e}", file=sys.stderr)
        # Fallback: console-only
        console_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | [%(name)s] | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        return logger

    # Standard formatter for file output
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | [%(name)s] | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Create log file path with UTC+8 timestamp
    timestamp = get_utc8_timestamp()
    log_filepath = log_dir / f"{name}_{timestamp}.log"

    # File handler
    file_handler = logging.FileHandler(log_filepath, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler (optional)
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)

        if colored_console:
            console_handler.setFormatter(
                ColoredFormatter(formatter._fmt, formatter.datefmt)
            )
        else:
            console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)

    logger.info(f"[SETUP] Log system initialized, log file: {log_filepath.resolve()}")
    return logger
