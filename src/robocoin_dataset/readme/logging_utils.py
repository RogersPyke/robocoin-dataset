"""
README logging utility functions.

Purpose:
    Provide logger setup and directory validation functions for README generation.

Dependencies:
    - logging, sys
    - datetime, timezone, timedelta
    - pathlib.Path
    - robocoin_dataset.utils.log_config
"""

import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from robocoin_dataset.utils.log_config import ANSI_BLUE, ANSI_GREEN, ANSI_RED, colorize


def get_utc8_timestamp() -> str:
    """
    Generate timestamp string in YYYYMMDDHHMMSS format (UTC+8).

    Input:
        None.

    Output:
        str: Timestamp string in format YYYYMMDDHHMMSS.
    """
    utc8 = timezone(timedelta(hours=8))
    now = datetime.now(utc8)
    return now.strftime("%Y%m%d%H%M%S")


def setup_readme_logger(
    log_dir: Path,
    script_name: str = "gen_readme",
    level: int = logging.INFO,
    console_output: bool = True,
) -> logging.Logger:
    """
    Setup logger with UTC+8 timestamp-based file naming.

    Input:
        log_dir (Path): Directory path where log files will be stored.
        script_name (str): Name of the script for log file naming.
        level (int): Logging level.
        console_output (bool): Whether to output logs to console.

    Output:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(script_name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    log_dir = Path(log_dir)
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Unable to create log directory {log_dir}: {e}", file=sys.stderr)
        console_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | [%(name)s] | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        return logger

    timestamp = get_utc8_timestamp()
    log_filepath = log_dir / f"{script_name}_{timestamp}.log"

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | [%(name)s] | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_filepath, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)

        class ColoredFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                msg = super().format(record)
                if record.levelno >= logging.ERROR:
                    return colorize(msg, ANSI_RED)
                if record.levelno >= logging.WARNING:
                    return colorize(msg, ANSI_RED)
                if record.levelno == logging.INFO and "SUCCESS" in msg:
                    return colorize(msg, ANSI_GREEN)
                if "URL" in msg or "path" in msg.lower() or "argument" in msg.lower():
                    return colorize(msg, ANSI_BLUE)
                return msg

        console_handler.setFormatter(ColoredFormatter(formatter._fmt, formatter.datefmt))
        logger.addHandler(console_handler)

    logger.info(f"[SETUP] Log system initialized, log file: {log_filepath.resolve()}")
    return logger


def validate_hardlink_directory(
    hardlink_dir: Path,
    logger: logging.Logger | None = None,
) -> None:
    """
    Validate that the provided directory appears to be a hardlink directory.

    Input:
        hardlink_dir (Path): Directory path to validate.
        logger (logging.Logger | None): Logger instance for warning messages.

    Output:
        None.
    """
    if logger is None:
        logger = logging.getLogger("validate_hardlink")

    hardlink_dir = Path(hardlink_dir)
    dir_name = hardlink_dir.name.lower()

    if not dir_name.endswith("hardlink"):
        warning_msg = (
            f"[HARDLINK_VALIDATE] WARNING: Directory name does not end with 'hardlink': "
            f"{hardlink_dir}\n"
            f"  Expected pattern: *hardlink (e.g., 'dataset_name_qced_hardlink')\n"
            f"  The input directory may not be the correct hardlink folder.\n"
            f"  Please verify you have provided the correct dataset directory.\n"
            f"  Continuing with generation anyway..."
        )
        logger.warning(colorize(warning_msg, ANSI_RED))
    else:
        logger.info(
            f"[HARDLINK_VALIDATE] Directory name validates as hardlink: {hardlink_dir.name}"
        )
