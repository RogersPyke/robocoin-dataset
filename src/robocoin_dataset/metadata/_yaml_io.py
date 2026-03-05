"""
YAML / JSON / JSONL file loading helpers for the Collect stage.

These are low-level I/O primitives shared across the other collect_utils sub-modules.
No business logic lives here.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import yaml

from robocoin_dataset.utils.log_config import log_error


def load_yaml_file(yaml_path: Path, logger: logging.Logger) -> Dict[str, Any]:
    """
    Load and parse a YAML file into a dictionary.

    Input:
        yaml_path (Path): Path to the YAML file.
        logger (logging.Logger): Logger instance for error reporting.

    Output:
        Dict[str, Any]: Parsed YAML content.
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

        logger.info(f"[YAML_LOAD] Successfully loaded YAML file: {yaml_path}")
        return raw_data

    except yaml.YAMLError as e:
        error_msg = f"[YAML_LOAD] Failed to parse YAML file {yaml_path}: {e}"
        log_error(logger, error_msg)
        raise
    except Exception as e:
        error_msg = f"[YAML_LOAD] Unexpected error loading YAML file {yaml_path}: {e}"
        log_error(logger, error_msg)
        raise


def load_schema_yaml(schema_yaml_path: Path, logger: logging.Logger) -> Dict[str, Any]:
    """
    Load the assets/info.yaml schema as a raw dictionary.

    Input:
        schema_yaml_path (Path): Absolute path to the schema YAML (assets/info.yaml).
        logger (logging.Logger): Logger instance for error reporting.

    Output:
        Dict[str, Any]: Raw schema dictionary keyed by canonical field names.
    """
    schema_yaml_path = Path(schema_yaml_path)
    if not schema_yaml_path.exists():
        error_msg = f"[SCHEMA_LOAD] Schema YAML file not found: {schema_yaml_path}"
        log_error(logger, error_msg)
        raise FileNotFoundError(error_msg)

    with open(schema_yaml_path, "r", encoding="utf-8") as f:
        schema = yaml.safe_load(f)

    if not isinstance(schema, dict):
        error_msg = (
            f"[SCHEMA_LOAD] Invalid schema format in {schema_yaml_path}. "
            "Expected top-level mapping."
        )
        log_error(logger, error_msg)
        raise ValueError(error_msg)

    logger.info(
        f"[SCHEMA_LOAD] Loaded schema from {schema_yaml_path} with {len(schema)} fields"
    )
    return schema


def load_json_file(json_path: Path, logger: logging.Logger) -> Dict[str, Any]:
    """
    Load a JSON file into a dictionary.

    Input:
        json_path (Path): Path to the JSON file.
        logger (logging.Logger): Logger for warnings.

    Output:
        Dict[str, Any]: Parsed JSON content (empty dict if root is not a mapping).
    """
    if not json_path.exists():
        raise FileNotFoundError(f"[SOURCE_LOAD] File not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        logger.warning(
            f"[SOURCE_LOAD] JSON root is not mapping for {json_path}, got {type(data)}"
        )
        return {}
    return data


def load_jsonl_file(jsonl_path: Path) -> List[Dict[str, Any]]:
    """
    Load a JSONL file into a list of record dicts.

    Input:
        jsonl_path (Path): Path to the JSONL file.

    Output:
        List[Dict[str, Any]]: Parsed records (non-dict lines are skipped).
    """
    records: List[Dict[str, Any]] = []
    if not jsonl_path.exists():
        raise FileNotFoundError(f"[SOURCE_LOAD] File not found: {jsonl_path}")
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                records.append(obj)
    return records
