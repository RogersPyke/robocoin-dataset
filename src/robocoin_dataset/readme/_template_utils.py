"""
README template utility functions.

Purpose:
    Provide template loading, rendering, and README file writing helpers.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict

import yaml
from jinja2 import Environment, FileSystemLoader, TemplateNotFound

from robocoin_dataset.utils.log_config import log_error, log_success


def load_jinja2_template(template_path: Path, logger: logging.Logger) -> Any:
    """
    Load Jinja2 template from file.

    Input:
        template_path (Path): Path to the Jinja2 template file (.j2).
        logger (logging.Logger): Logger instance for error reporting.

    Output:
        Any: Loaded Jinja2 template object.
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

        def to_yaml(obj: Any) -> str:
            return yaml.safe_dump(
                obj,
                sort_keys=False,
                default_flow_style=False,
                allow_unicode=False,
                width=120,
            )

        def match_test(value: Any, pattern: str) -> bool:
            if value is None:
                return False
            return re.search(pattern, str(value)) is not None

        env.filters["to_yaml"] = to_yaml
        env.tests["match"] = match_test
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
        template (Any): Jinja2 template object to render.
        data (Dict[str, Any]): Dictionary of variables to pass to template.
        logger (logging.Logger): Logger instance for error reporting.

    Output:
        str: Rendered template content as string.
    """
    try:
        rendered_content = template.render(**data)
        logger.info("[TEMPLATE_RENDER] Successfully rendered template")
        return rendered_content
    except Exception as e:
        error_msg = f"[TEMPLATE_RENDER] Failed to render template: {e}"
        log_error(logger, error_msg)
        raise


def write_readme_file(
    content: str, output_path: Path, logger: logging.Logger
) -> None:
    """
    Write rendered README content to file.

    Input:
        content (str): Rendered README content to write.
        output_path (Path): Path where README.md file should be written.
        logger (logging.Logger): Logger instance for error reporting.

    Output:
        None.
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
