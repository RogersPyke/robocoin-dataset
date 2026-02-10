"""
Log Configuration Module

This module provides logging utilities including ANSI color codes and helper functions
for colored log output. It is designed as a global utility module for use across the
entire codebase.

Dependencies:
    - logging: For logging functionality
    - os: For environment variable detection
    - pathlib: For path operations
    - dataclasses: For LogConfig dataclass

Usage:
    Basic usage:
        from robocoin_dataset.utils.log_config import log_success, log_error, ANSI_GREEN
        
        logger = logging.getLogger(__name__)
        log_success(logger, "Operation completed successfully")
        log_error(logger, "Operation failed")
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

# Calculate default log path relative to project root
# This assumes utils is at: src/robocoin_dataset/utils/
# Project root would be: src/robocoin_dataset/../../
DEFAULT_OUTPUT_LOG_PATH = Path(__file__).resolve().parents[2] / "logs"

# ANSI color codes for terminal output
# Red for WARNING and ERR
ANSI_RED = "\033[91m"
# Green for SUCCESS
ANSI_GREEN = "\033[92m"
# Blue for URLs and arguments
ANSI_BLUE = "\033[94m"
# Reset color
ANSI_RESET = "\033[0m"


def colorize(text: str, color: str, use_color: bool = True) -> str:
    """
    Apply ANSI color code to text if terminal supports colors.

    Input:
        text (str): Text to colorize
        color (str): ANSI color code (e.g., ANSI_RED, ANSI_GREEN, ANSI_BLUE)
        use_color (bool): Whether to apply color (default: True, auto-detected if None)

    Output:
        str: Colorized text string

    Logic:
        1. If use_color is False, return text as-is
        2. If use_color is True, check terminal support via environment variables
        3. If terminal supports colors, wrap text with color code and reset code
        4. Return colorized or original text

    Usage:
        Used by log helper functions to add color to log messages.
    """
    if use_color is False:
        return text
    # Auto-detect if use_color is None
    if use_color is True:
        # Check if terminal supports colors
        use_color = os.getenv("TERM") not in (None, "dumb") and os.getenv("NO_COLOR") is None
    if use_color:
        return f"{color}{text}{ANSI_RESET}"
    return text


def log_success(logger: logging.Logger, message: str) -> None:
    """
    Log success message with green color.

    Input:
        logger (logging.Logger): Logger instance
        message (str): Success message to log

    Output:
        None

    Logic:
        1. Colorize message with ANSI_GREEN
        2. Log at INFO level

    Usage:
        Call this function to log success messages with green color.
    """
    logger.info(colorize(message, ANSI_GREEN))


def log_error(logger: logging.Logger, message: str) -> None:
    """
    Log error message with red color.

    Input:
        logger (logging.Logger): Logger instance
        message (str): Error message to log

    Output:
        None

    Logic:
        1. Colorize message with ANSI_RED
        2. Log at ERROR level

    Usage:
        Call this function to log error messages with red color.
    """
    logger.error(colorize(message, ANSI_RED))


def log_warning(logger: logging.Logger, message: str) -> None:
    """
    Log warning message with red color.

    Input:
        logger (logging.Logger): Logger instance
        message (str): Warning message to log

    Output:
        None

    Logic:
        1. Colorize message with ANSI_RED
        2. Log at WARNING level

    Usage:
        Call this function to log warning messages with red color.
    """
    logger.warning(colorize(message, ANSI_RED))


def log_url(logger: logging.Logger, message: str, level: int = logging.INFO) -> None:
    """
    Log URL or argument with blue color.

    Input:
        logger (logging.Logger): Logger instance
        message (str): URL or argument message to log
        level (int): Logging level (default: logging.INFO)

    Output:
        None

    Logic:
        1. Colorize message with ANSI_BLUE
        2. Log at specified level

    Usage:
        Call this function to log URLs or arguments with blue color.
    """
    logger.log(level, colorize(message, ANSI_BLUE))


@dataclass()
class LogConfig:
    """
    Configuration class for logging settings.

    This class defines the configuration parameters for logging in the application,
    including log directory, console output options, and log level.

    Attributes:
        log_dir (str): Directory path where log files will be stored. Defaults to DEFAULT_OUTPUT_LOG_PATH.
        log_to_console (bool): Flag indicating whether logs should be output to console. Defaults to True.
        log_level (str): Logging level (e.g., "DEBUG", "INFO", "WARNING", "ERROR"). Defaults to "INFO".

    Usage:
        Used in configuration files to specify logging settings.
    """

    log_dir: str = str(DEFAULT_OUTPUT_LOG_PATH)
    log_to_console: bool = True
    log_level: str = "INFO"

