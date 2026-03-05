"""
README stage utility functions.

This module provides the minimal helpers needed by the README Render stage
(Phase 2).  It intentionally has no dependency on the Collect stage
(metadata/) to keep the two stages cleanly separated.

Public API:
    load_collected_info_yaml  — read the flat info.yaml written by InfoCollector
                                and return it as a plain dict for template rendering
"""

import logging
from pathlib import Path
from typing import Any, Dict

import yaml

from robocoin_dataset.utils.log_config import log_error


def load_collected_info_yaml(
    info_yaml_path: Path,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """
    Load a pre-collected flat info.yaml into a context dict for template rendering.

    This function is the Phase 2 counterpart to InfoCollector.collect(): it reads
    the flat YAML file that InfoCollector wrote and returns it as a plain dict.

    Unlike load_schema_yaml (which reads the schema template with nested field
    specs), this function expects a simple flat key->value mapping with no schema
    metadata.

    Input:
        info_yaml_path (Path): Absolute path to the info.yaml written by InfoCollector.
            Expected format: flat YAML mapping (str -> Any), e.g.::

                dataset_name: my_robot_task
                task_categories: [robotics]
                statistics:
                    total_episodes: 100

        logger (logging.Logger): Logger instance for error reporting.

    Output:
        Dict[str, Any]: Flat context dict; keys are field names, values are resolved.
            Returns an empty dict if the file is empty.

    Raises:
        FileNotFoundError: If info_yaml_path does not exist.
        ValueError: If YAML top-level type is not a mapping.
        yaml.YAMLError: If YAML parsing fails.
    """
    info_yaml_path = Path(info_yaml_path)
    if not info_yaml_path.exists():
        error_msg = f"[INFO_LOAD] Collected info.yaml not found: {info_yaml_path}"
        log_error(logger, error_msg)
        raise FileNotFoundError(error_msg)

    try:
        with open(info_yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        error_msg = f"[INFO_LOAD] Failed to parse info.yaml at {info_yaml_path}: {e}"
        log_error(logger, error_msg)
        raise

    if data is None:
        logger.warning(f"[INFO_LOAD] Collected info.yaml is empty: {info_yaml_path}")
        return {}

    if not isinstance(data, dict):
        error_msg = (
            f"[INFO_LOAD] Unexpected top-level type in {info_yaml_path}: "
            f"{type(data).__name__}. Expected mapping."
        )
        log_error(logger, error_msg)
        raise ValueError(error_msg)

    logger.info(
        f"[INFO_LOAD] Loaded {len(data)} fields from collected info.yaml: {info_yaml_path}"
    )
    return data
