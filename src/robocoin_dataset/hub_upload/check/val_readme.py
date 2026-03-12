"""
README presence check for hub upload.

This module provides a single validation: ensure the folder to be uploaded
contains a README.md file. Used before upload to avoid pushing datasets without
documentation.

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


def validate_readme(folder_path: Path) -> None:
    """
    Require that the folder contains a README.md file.

    Input:
        folder_path: Path to the directory to be uploaded. Must exist and be a directory.
    Output:
        None. Returns normally if README.md exists and is a file.
    Raises:
        ValueError: If README.md is missing or not a regular file (message: MISSING_README_MSG).
    Usage:
        Call before uploading a dataset folder so upload fails fast when README is missing.
    """
    readme_path = folder_path / "README.md"
    if not readme_path.exists() or not readme_path.is_file():
        raise ValueError(MISSING_README_MSG)
