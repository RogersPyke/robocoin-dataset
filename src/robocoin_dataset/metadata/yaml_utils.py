"""
YAML file utilities for the Collect stage (metadata information collection).

Purpose:
    Provide YAML file locating and loading helpers specifically for metadata
    collection. This module is self-contained and does not depend on readme/ modules.

Dependencies:
    - logging, pathlib.Path, yaml
    - robocoin_dataset.utils.log_config (log_error)

Usage example:
    yaml_path = locate_yaml_file(
        hardlink_dir="/path/to/dataset_qced_hardlink",
        yaml_filename="local_dataset_info.yaml",
        logger=logger,
    )
    data = load_yaml_file(yaml_path, logger)
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

import yaml

from robocoin_dataset.utils.log_config import log_error


def locate_yaml_file(
    hardlink_dir: Path,
    yaml_filename: str = "local_dataset_info.yaml",
    custom_yaml_path: Path | None = None,
    logger: logging.Logger | None = None,
) -> Path:
    """
    Locate YAML file with priority-based search strategy.

    Input:
        hardlink_dir (Path): The hardlink directory (dataset input directory).
        yaml_filename (str): Name of YAML file to search for in hardlink_dir.
        custom_yaml_path (Path | None): Explicit YAML file path provided by user.
        logger (logging.Logger | None): Logger instance for error reporting.

    Output:
        Path: Validated path to the located YAML file.

    Usage:
        Locates the input metadata file for the Collect stage. Supports both
        automatic discovery (in hardlink_dir) and explicit user-provided paths.

    Raises:
        FileNotFoundError: If file not found.
        ValueError: If path is not a valid file.
    """
    if logger is None:
        logger = logging.getLogger("locate_yaml")

    hardlink_dir = Path(hardlink_dir)
    if not hardlink_dir.exists():
        error_msg = f"[YAML_LOCATE] Hardlink directory does not exist: {hardlink_dir}"
        log_error(logger, error_msg)
        raise FileNotFoundError(error_msg)
    if not hardlink_dir.is_dir():
        error_msg = f"[YAML_LOCATE] Hardlink path is not a directory: {hardlink_dir}"
        log_error(logger, error_msg)
        raise ValueError(error_msg)

    if custom_yaml_path is not None:
        custom_yaml_path = Path(custom_yaml_path)
        if not custom_yaml_path.exists():
            error_msg = (
                f"[YAML_LOCATE] Custom YAML file path not found or not accessible: "
                f"{custom_yaml_path}"
            )
            log_error(logger, error_msg)
            raise FileNotFoundError(error_msg)
        if not custom_yaml_path.is_file():
            error_msg = f"[YAML_LOCATE] Custom YAML path is not a file: {custom_yaml_path}"
            log_error(logger, error_msg)
            raise ValueError(error_msg)
        logger.info(f"[YAML_LOCATE] Using custom YAML file path: {custom_yaml_path}")
        return custom_yaml_path

    default_yaml_path = hardlink_dir / yaml_filename
    if default_yaml_path.exists() and default_yaml_path.is_file():
        logger.info(
            f"[YAML_LOCATE] Found YAML file at first level of hardlink directory: "
            f"{default_yaml_path}"
        )
        return default_yaml_path

    try:
        first_level_files = list(hardlink_dir.iterdir())
        first_level_names = [f.name for f in first_level_files if f.is_file()]
        available_files_str = ", ".join(first_level_names) if first_level_names else "(none)"
    except Exception as e:
        available_files_str = f"(unable to list: {e})"

    error_msg = (
        f"[YAML_LOCATE] YAML file '{yaml_filename}' not found in hardlink directory "
        f"first level: {hardlink_dir}\n"
        f"  Available files at first level: {available_files_str}\n"
        f"  To fix: (1) Ensure '{yaml_filename}' is placed in {hardlink_dir}, or "
        f"(2) Provide explicit YAML file path via parameter"
    )
    log_error(logger, error_msg)
    raise FileNotFoundError(error_msg)


def _normalize_aliases(alias_value: Any) -> List[str]:
    """
    Normalize alias configuration into list of strings.

    Input:
        alias_value (Any): Alias config value from schema.

    Output:
        List[str]: Normalized alias names.

    Usage:
        Internal helper for load_yaml_file. Converts various alias formats
        (string, list, None) into a consistent list representation.
    """
    if alias_value is None:
        return []
    if isinstance(alias_value, str):
        return [alias_value]
    if isinstance(alias_value, list):
        return [item for item in alias_value if isinstance(item, str)]
    return []


def load_yaml_file(yaml_path: Path, logger: logging.Logger) -> Dict[str, Any]:
    """
    Load and parse YAML file into dictionary with alias support.

    Input:
        yaml_path (Path): Path to the YAML file to load.
        logger (logging.Logger): Logger instance for error reporting.

    Output:
        Dict[str, Any]: Parsed YAML content as dictionary.
                        For schema YAML: raw nested structure with field specs.
                        For data YAML: flattened key->value mapping.

    Usage:
        Loads YAML files during metadata collection. Handles:
        - Missing/empty files (returns empty dict or logs warning)
        - Schema files with nested field specifications
        - User data with canonical keys and deprecated aliases
        - Conflict detection (canonical + alias both set)

    Raises:
        FileNotFoundError: If yaml_path does not exist.
        ValueError: If YAML root is not a mapping.
        yaml.YAMLError: If YAML parsing fails.
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
        if not isinstance(raw_data, dict):
            logger.warning(
                f"[YAML_LOAD] YAML root is not mapping for {yaml_path}: {type(raw_data)}"
            )
            return {}

        flattened_data: Dict[str, Any] = {}
        consumed_alias_keys: set[str] = set()

        for key, value in raw_data.items():
            if isinstance(value, dict) and (
                "value" in value or "default" in value or "source" in value
            ):
                aliases = _normalize_aliases(value.get("alias"))
                canonical_value = value.get("value")
                canonical_is_set = canonical_value not in (None, "")
                alias_hits = [
                    alias_key
                    for alias_key in aliases
                    if alias_key in raw_data and raw_data[alias_key] not in (None, "")
                ]

                if canonical_is_set and alias_hits:
                    error_msg = (
                        f"[YAML_LOAD] Conflict for field '{key}': canonical key and alias "
                        f"keys are both set ({alias_hits}). Keep only canonical key."
                    )
                    log_error(logger, error_msg)
                    raise ValueError(error_msg)
                if len(alias_hits) > 1:
                    error_msg = (
                        f"[YAML_LOAD] Conflict for field '{key}': multiple alias keys are set "
                        f"({alias_hits}). Keep only one alias key."
                    )
                    log_error(logger, error_msg)
                    raise ValueError(error_msg)

                if canonical_is_set:
                    flattened_data[key] = canonical_value
                elif alias_hits:
                    alias_key = alias_hits[0]
                    flattened_data[key] = raw_data[alias_key]
                    consumed_alias_keys.add(alias_key)
                    logger.warning(
                        f"[YAML_LOAD] Deprecated alias '{alias_key}' used for field '{key}'. "
                        "Please migrate to canonical field name."
                    )
                elif "default" in value:
                    flattened_data[key] = value.get("default")
                    if value.get("source") != "fixed":
                        logger.warning(
                            f"[YAML_LOAD] Field '{key}' is missing. "
                            "Using default value from schema."
                        )
                else:
                    flattened_data[key] = None
                    if value.get("source") != "fixed":
                        logger.warning(
                            f"[YAML_LOAD] Field '{key}' is missing and has no default."
                        )
            else:
                if key not in consumed_alias_keys:
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
