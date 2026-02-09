import logging
import os
from dataclasses import dataclass
from pathlib import Path

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

    Args:
        text: Text to colorize
        color: ANSI color code (e.g., ANSI_RED, ANSI_GREEN, ANSI_BLUE)
        use_color: Whether to apply color (default: True, auto-detected if None)

    Returns:
        Colorized text string
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

    Args:
        logger: Logger instance
        message: Success message to log
    """
    logger.info(colorize(message, ANSI_GREEN))


def log_error(logger: logging.Logger, message: str) -> None:
    """
    Log error message with red color.

    Args:
        logger: Logger instance
        message: Error message to log
    """
    logger.error(colorize(message, ANSI_RED))


def log_warning(logger: logging.Logger, message: str) -> None:
    """
    Log warning message with red color.

    Args:
        logger: Logger instance
        message: Warning message to log
    """
    logger.warning(colorize(message, ANSI_RED))


def log_url(logger: logging.Logger, message: str, level: int = logging.INFO) -> None:
    """
    Log URL or argument with blue color.

    Args:
        logger: Logger instance
        message: URL or argument message to log
        level: Logging level (default: INFO)
    """
    logger.log(level, colorize(message, ANSI_BLUE))


@dataclass()
class LogConfig:
    """
    Configuration class for logging settings.

    This class defines the configuration parameters for logging in the application,
    including log directory, console output options, and log level.

    Attributes:
        log_dir (str): Directory path where log files will be stored. Defaults to empty string.
        log_to_console (bool): Flag indicating whether logs should be output to console. Defaults to True.
        log_level (str): Logging level (e.g., "DEBUG", "INFO", "WARNING", "ERROR"). Defaults to "INFO".
    """

    log_dir: str = str(DEFAULT_OUTPUT_LOG_PATH)
    log_to_console: bool = True
    log_level: str = "INFO"
