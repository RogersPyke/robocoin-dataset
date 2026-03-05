"""
Logging utilities for the Render stage (README generation).

Purpose:
    Provide logger setup for README generation operations
    by leveraging global logging utilities from robocoin_dataset.utils.

Dependencies:
    - logging
    - pathlib.Path
    - robocoin_dataset.utils.logger (setup_logger_utc8)

Usage:
    logger = setup_readme_logger(
        log_dir="/path/to/logs",
        script_name="gen_readme",
    )
    logger.info("[STAGE] Rendering started")
"""

import logging
from pathlib import Path

from robocoin_dataset.utils.logger import setup_logger_utc8


def setup_readme_logger(
    log_dir: Path,
    script_name: str = "gen_readme",
    level: int = logging.INFO,
    console_output: bool = True,
) -> logging.Logger:
    """
    Setup logger for README generation with UTC+8 timestamp-based file naming.

    Input:
        log_dir (Path): Directory path where log files will be stored.
        script_name (str): Name of the script for log file naming (e.g., "gen_readme").
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
        Called by ReadmeGenerator to setup centralized logging for audit trails
        during the Render stage (template loading, rendering, file writing).

    Note:
        This is a README-level logger. Metadata collection uses
        metadata/logging.py setup_collect_logger instead.
    """
    return setup_logger_utc8(
        name=script_name,
        log_dir=log_dir,
        level=level,
        console_output=console_output,
        colored_console=True,
    )
