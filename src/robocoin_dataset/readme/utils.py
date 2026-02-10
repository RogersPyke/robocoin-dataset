"""
README Generator Utility Functions

This module contains all underlying implementation functions for README generation,
including YAML loading, template rendering, file writing, and logging setup.
These functions are used by the ReadmeGenerator class for actual work.

Dependencies:
    - yaml: For parsing YAML configuration files
    - jinja2: For template rendering
    - logging: For audit logging
    - pathlib: For file path operations
    - datetime: For timestamp generation (UTC+8)

Usage:
    These functions are typically called by ReadmeGenerator class, not directly.
"""

import logging
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict

import yaml
from jinja2 import Environment, FileSystemLoader, TemplateNotFound

from robocoin_dataset.utils.log_config import (
    ANSI_BLUE,
    ANSI_GREEN,
    ANSI_RED,
    ANSI_RESET,
    colorize,
    log_error,
    log_success,
)


# ============================================================================
# Logging Configuration
# ============================================================================


def get_utc8_timestamp() -> str:
    """
    Generate timestamp string in YYYYMMDDHHMMSS format (UTC+8).

    Returns:
        str: Timestamp string in format YYYYMMDDHHMMSS (e.g., "20250115143022")

    Usage:
        Used for log file naming to ensure sortable and unambiguous file names.
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
        log_dir (Path): Directory path where log files will be stored
        script_name (str): Name of the script for log file naming (default: "gen_readme")
        level (int): Logging level (default: logging.INFO)
        console_output (bool): Whether to output logs to console (default: True)

    Output:
        logging.Logger: Configured logger instance

    Logic:
        1. Create log directory if it doesn't exist
        2. Generate timestamp in YYYYMMDDHHMMSS format (UTC+8)
        3. Create log file path: <script_name>_<timestamp>.log
        4. Configure file handler and optional console handler
        5. Apply ANSI color formatting for console output

    Usage:
        Called during ReadmeGenerator initialization to set up audit logging.
    """
    logger = logging.getLogger(script_name)
    if logger.handlers:
        # Already configured, avoid duplicate handlers
        return logger

    logger.setLevel(level)
    logger.propagate = False

    # Create log directory
    log_dir = Path(log_dir)
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Unable to create log directory {log_dir}: {e}", file=sys.stderr)
        # Fallback: console output only
        console_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | [%(name)s] | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        return logger

    # Generate timestamp in YYYYMMDDHHMMSS format (UTC+8)
    timestamp = get_utc8_timestamp()
    log_filepath = log_dir / f"{script_name}_{timestamp}.log"

    # Configure formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | [%(name)s] | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler
    file_handler = logging.FileHandler(log_filepath, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler (optional, with color support)
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)

        # Custom formatter for console with color support
        class ColoredFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                msg = super().format(record)
                if record.levelno >= logging.ERROR:
                    return colorize(msg, ANSI_RED)
                elif record.levelno >= logging.WARNING:
                    return colorize(msg, ANSI_RED)
                elif record.levelno == logging.INFO and "SUCCESS" in msg:
                    return colorize(msg, ANSI_GREEN)
                elif "URL" in msg or "path" in msg.lower() or "argument" in msg.lower():
                    return colorize(msg, ANSI_BLUE)
                return msg

        console_handler.setFormatter(ColoredFormatter(formatter._fmt, formatter.datefmt))
        logger.addHandler(console_handler)

    logger.info(f"[SETUP] Log system initialized, log file: {log_filepath.resolve()}")
    return logger


# ============================================================================
# YAML Loading Functions
# ============================================================================


def load_yaml_file(yaml_path: Path, logger: logging.Logger) -> Dict[str, Any]:
    """
    Load and parse YAML file into dictionary.

    Input:
        yaml_path (Path): Path to the YAML file to load
        logger (logging.Logger): Logger instance for error reporting

    Output:
        Dict[str, Any]: Parsed YAML content as dictionary

    Logic:
        1. Check if file exists
        2. Read file content
        3. Parse YAML content
        4. Extract 'value' fields from nested structure (if present)
        5. Return flattened dictionary

    Usage:
        Called by ReadmeGenerator to load dataset metadata from info.yaml.

    Raises:
        FileNotFoundError: If YAML file does not exist
        yaml.YAMLError: If YAML parsing fails
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        error_msg = f"[YAML_LOAD] YAML file not found: {yaml_path}"
        log_error(logger, error_msg)
        raise FileNotFoundError(error_msg)

    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)

        if raw_data is None:
            logger.warning(f"[YAML_LOAD] YAML file is empty: {yaml_path}")
            return {}

        # Extract 'value' fields from nested structure
        # Format: {key: {value: actual_value, required: ..., source: ...}}
        flattened_data = {}
        for key, value in raw_data.items():
            if isinstance(value, dict) and "value" in value:
                flattened_data[key] = value["value"]
            else:
                flattened_data[key] = value

        logger.info(f"[YAML_LOAD] Successfully loaded YAML file: {yaml_path}")
        return flattened_data

    except yaml.YAMLError as e:
        error_msg = f"[YAML_LOAD] Failed to parse YAML file {yaml_path}: {e}"
        log_error(logger, error_msg)
        raise
    except Exception as e:
        error_msg = f"[YAML_LOAD] Unexpected error loading YAML file {yaml_path}: {e}"
        log_error(logger, error_msg)
        raise


# ============================================================================
# Template Rendering Functions
# ============================================================================


def load_jinja2_template(template_path: Path, logger: logging.Logger) -> Any:
    """
    Load Jinja2 template from file.

    Input:
        template_path (Path): Path to the Jinja2 template file (.j2)
        logger (logging.Logger): Logger instance for error reporting

    Output:
        jinja2.Template: Loaded Jinja2 template object

    Logic:
        1. Check if template file exists
        2. Create Jinja2 Environment with FileSystemLoader
        3. Load template from file
        4. Return template object

    Usage:
        Called by ReadmeGenerator to load the readme.j2 template for rendering.

    Raises:
        FileNotFoundError: If template file does not exist
        TemplateNotFound: If template cannot be loaded by Jinja2
    """
    template_path = Path(template_path)
    if not template_path.exists():
        error_msg = f"[TEMPLATE_LOAD] Template file not found: {template_path}"
        log_error(logger, error_msg)
        raise FileNotFoundError(error_msg)

    try:
        template_dir = template_path.parent
        template_name = template_path.name

        env = Environment(loader=FileSystemLoader(str(template_dir)))

        # Add a safe YAML dump filter for rendering structured objects (e.g., features)
        # Notes:
        # - allow_unicode=False: prefer ASCII output (per repo constraints)
        # - sort_keys=False: preserve insertion order for readability
        def to_yaml(obj: Any) -> str:
            return yaml.safe_dump(
                obj,
                sort_keys=False,
                default_flow_style=False,
                allow_unicode=False,
                width=120,
            )

        env.filters["to_yaml"] = to_yaml
        template = env.get_template(template_name)

        logger.info(f"[TEMPLATE_LOAD] Successfully loaded template: {template_path}")
        return template

    except TemplateNotFound as e:
        error_msg = f"[TEMPLATE_LOAD] Template not found: {template_path} - {e}"
        log_error(logger, error_msg)
        raise
    except Exception as e:
        error_msg = f"[TEMPLATE_LOAD] Unexpected error loading template {template_path}: {e}"
        log_error(logger, error_msg)
        raise


def render_template(
    template: Any, data: Dict[str, Any], logger: logging.Logger
) -> str:
    """
    Render Jinja2 template with provided data.

    Input:
        template (jinja2.Template): Jinja2 template object to render
        data (Dict[str, Any]): Dictionary of variables to pass to template
        logger (logging.Logger): Logger instance for error reporting

    Output:
        str: Rendered template content as string

    Logic:
        1. Call template.render() with data dictionary
        2. Return rendered string

    Usage:
        Called by ReadmeGenerator to generate README content from template and data.

    Raises:
        jinja2.TemplateError: If template rendering fails
    """
    try:
        rendered_content = template.render(**data)
        logger.info("[TEMPLATE_RENDER] Successfully rendered template")
        return rendered_content

    except Exception as e:
        error_msg = f"[TEMPLATE_RENDER] Failed to render template: {e}"
        log_error(logger, error_msg)
        raise


# ============================================================================
# File Writing Functions
# ============================================================================


def write_readme_file(
    content: str, output_path: Path, logger: logging.Logger
) -> None:
    """
    Write rendered README content to file.

    Input:
        content (str): Rendered README content to write
        output_path (Path): Path where README.md file should be written
        logger (logging.Logger): Logger instance for error reporting

    Output:
        None

    Logic:
        1. Create parent directories if they don't exist
        2. Write content to file with UTF-8 encoding
        3. Log success message

    Usage:
        Called by ReadmeGenerator to save the generated README.md file.

    Raises:
        IOError: If file writing fails
    """
    output_path = Path(output_path)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        log_success(logger, f"[FILE_WRITE] Successfully wrote README to: {output_path}")
        logger.info(f"[FILE_WRITE] README file size: {len(content)} bytes")

    except Exception as e:
        error_msg = f"[FILE_WRITE] Failed to write README file {output_path}: {e}"
        log_error(logger, error_msg)
        raise

