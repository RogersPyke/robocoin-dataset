"""
Logging utilities for the Collect stage (metadata collection).

Purpose:
    Provide logger setup for independent metadata collection operations
    by leveraging global logging utilities from robocoin_dataset.utils.

Dependencies:
    - logging
    - pathlib.Path
    - robocoin_dataset.utils.logger (setup_logger_utc8)
    - robocoin_dataset.utils.log_config (ANSI colors, log_warning)

Usage:
    logger = setup_collect_logger(
        log_dir="/path/to/logs",
        script_name="collect",
    )
    logger.info("[STAGE] Processing started")
    
    validate_hardlink_directory(hardlink_dir, logger)
"""

import logging
from pathlib import Path

from robocoin_dataset.utils.log_config import ANSI_RED, colorize, log_warning
from robocoin_dataset.utils.logger import setup_logger_utc8


def setup_collect_logger(
    log_dir: Path,
    script_name: str = "collect",
    level: int = logging.INFO,
    console_output: bool = True,
) -> logging.Logger:
    """
    Setup logger for metadata collection with UTC+8 timestamp-based file naming.

    Input:
        log_dir (Path): Directory path where log files will be stored.
        script_name (str): Name of the script for log file naming (e.g., "collect").
        level (int): Logging level (default: logging.INFO).
        console_output (bool): Whether to output logs to console (default: True).

    Output:
        logging.Logger: Configured logger instance.

    Logic:
        Delegates to setup_logger_utc8 from global utilities, which handles:
        1. Logger creation and configuration
        2. UTC+8 timestamp generation
        3. File and console output setup
        4. Colored console formatting

    Usage:
        Called by InfoCollector and other metadata collection classes
        to setup centralized logging for audit trails.

    Note:
        This is a metadata-level logger. README stage should use
        readme/logging.py setup_readme_logger instead.
    """
    return setup_logger_utc8(
        name=script_name,
        log_dir=log_dir,
        level=level,
        console_output=console_output,
        colored_console=True,
    )


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

    Logic:
        1. Get logger instance (create default if None)
        2. Normalize directory path and extract name
        3. Check if directory name ends with 'hardlink'
        4. Log validation result with appropriate color

    Usage:
        Called during InfoCollector initialization to validate input directory.
        Ensures user is operating on correct dataset structure.
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
        log_warning(logger, warning_msg)
    else:
        logger.info(
            f"[HARDLINK_VALIDATE] Directory name validates as hardlink: {hardlink_dir.name}"
        )
