"""
README presence check for hub upload.

This module provides a single validation: ensure the folder to be uploaded
contains a README.md file. Used before upload to avoid pushing datasets without
documentation.

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

# Error message when README is missing (required by caller)
# User-facing error when README.md is missing
MISSING_README_MSG = "Missing README, please generate first."

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
