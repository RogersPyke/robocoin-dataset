"""
Artifact validation for hub upload.

This module validates required artifacts before upload:
- README.md must exist
- info.yaml must exist and be a mapping
- dataset_name must exist in info.yaml

Finds README.md in a case-insensitive way (e.g. README.md, readme.md) and uses
the resolved folder path so symlinks and relative paths match the on-disk layout.

Dependencies:
    - pathlib.Path: Path handling.

Usage:
    from robocoin_dataset.hub_upload.check.val_readme import validate_readme
    validate_readme(Path("/path/to/dataset_folder"))
    # Raises ValueError if README.md is missing.
"""

from pathlib import Path
from typing import Any

import yaml

# Error message when README is missing (required by caller)
# User-facing error when README.md is missing
MISSING_README_MSG = "Missing README, please generate first."
MISSING_INFO_YAML_MSG = "Missing info.yaml, please run metadata collection first."
INVALID_INFO_YAML_MSG = "Invalid info.yaml, expected YAML mapping with dataset_name."

# Canonical filename we look for (case-insensitive)
README_FILENAME = "README.md"


def get_readme_path(folder_path: Path) -> Path | None:
    """
    Return the path to README.md in the folder, using case-insensitive matching.

    Expands ~ and resolves folder_path so symlinks and relative paths match the
    real directory. Accepts any file whose name equals "readme.md" when lowercased
    (e.g. README.md, readme.md). Returns None if the folder does not exist, is not
    a directory, or contains no such file.

    Input:
        folder_path: Path to the directory to be uploaded (may contain ~).
    Output:
        Path to the README file, or None if not found.
    """
    try:
        expanded = folder_path.expanduser()
        resolved = expanded.resolve()
    except (OSError, RuntimeError):
        return None
    if not resolved.is_dir():
        return None
    for p in resolved.iterdir():
        if p.is_file() and p.name.lower() == README_FILENAME.lower():
            return p
    return None


def validate_readme(folder_path: Path) -> None:
    """
    Require that the folder contains a README.md file.

    Uses case-insensitive matching and the resolved folder path so that paths
    from the database (e.g. with symlinks or different casing) still find the
    file when it exists as readme.md or README.md.

    Input:
        folder_path: Path to the directory to be uploaded. Must exist and be a directory.
    Output:
        None. Returns normally if README.md exists and is a file.
    Raises:
        ValueError: If README.md is missing or not a regular file (message: MISSING_README_MSG).
    Usage:
        Call before uploading a dataset folder so upload fails fast when README is missing.
    """
    readme_path = get_readme_path(folder_path)
    if readme_path is None:
        try:
            resolved = folder_path.expanduser().resolve()
            path_checked = str(resolved)
        except Exception:
            path_checked = str(folder_path)
        raise ValueError(f"{MISSING_README_MSG} Path checked: {path_checked}")


def get_info_yaml_path(folder_path: Path) -> Path | None:
    """
    Return `<folder_path>/info.yaml` if it exists and is a file.
    """
    try:
        resolved = folder_path.expanduser().resolve()
    except (OSError, RuntimeError):
        return None
    info_yaml_path = resolved / "info.yaml"
    if info_yaml_path.exists() and info_yaml_path.is_file():
        return info_yaml_path
    return None


def load_dataset_name_from_info_yaml(info_yaml_path: Path) -> str:
    """
    Load dataset_name from info.yaml.
    """
    with open(info_yaml_path, "r", encoding="utf-8") as f:
        data: Any = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{INVALID_INFO_YAML_MSG} Path checked: {info_yaml_path}")
    dataset_name = (data.get("dataset_name") or "").strip()
    if not dataset_name:
        raise ValueError(f"{INVALID_INFO_YAML_MSG} dataset_name is missing. Path checked: {info_yaml_path}")
    return dataset_name


def validate_upload_artifacts(folder_path: Path) -> str:
    """
    Validate required upload artifacts and return canonical dataset_name.
    """
    validate_readme(folder_path)
    info_yaml_path = get_info_yaml_path(folder_path)
    if info_yaml_path is None:
        try:
            resolved = folder_path.expanduser().resolve()
            path_checked = str(resolved / "info.yaml")
        except Exception:
            path_checked = str(folder_path / "info.yaml")
        raise ValueError(f"{MISSING_INFO_YAML_MSG} Path checked: {path_checked}")
    return load_dataset_name_from_info_yaml(info_yaml_path)
