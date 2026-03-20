"""
Temporary upload hook: skip any directory named "backup".

This module is intentionally detachable. To disable it, remove the single call
site in scripts/hub_upload/upload2hub.py or delete this file.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any


def _copy_tree_without_backup(src_root: Path, dst_root: Path) -> None:
    """Copy a directory tree while excluding folders named exactly 'backup'."""
    if not src_root.exists():
        raise FileNotFoundError(f"Source path does not exist: {src_root}")

    dst_root.mkdir(parents=True, exist_ok=True)

    for entry in src_root.iterdir():
        if entry.name == "backup":
            continue

        target = dst_root / entry.name
        if entry.is_dir():
            shutil.copytree(
                entry,
                target,
                ignore=shutil.ignore_patterns("backup"),
                copy_function=shutil.copy2,
                dirs_exist_ok=True,
            )
        else:
            shutil.copy2(entry, target)


def install_skip_backup_upload_hook(uploader: Any, logger: Any | None = None) -> None:
    """
    Monkey-patch uploader.hub.upload_repo() to stage a copy without `backup` dirs.

    This is a temporary plug-and-play hook and should not be moved into core code.
    """
    if getattr(uploader.hub, "_skip_backup_hook_installed", False):
        return

    original_upload_repo = uploader.hub.upload_repo

    def wrapped_upload_repo(*args: Any, **kwargs: Any) -> Any:
        folder_path = kwargs.get("folder_path")
        if folder_path is None:
            return original_upload_repo(*args, **kwargs)

        src_root = Path(folder_path)
        with tempfile.TemporaryDirectory(prefix="temp-skip-backup-upload-") as tmpdir:
            staged_root = Path(tmpdir) / src_root.name
            _copy_tree_without_backup(src_root=src_root, dst_root=staged_root)
            kwargs["folder_path"] = staged_root

            if logger:
                logger.info(
                    "[TEMP_HOOK] Skip-backup hook active: staging upload without 'backup' dirs"
                )
            return original_upload_repo(*args, **kwargs)

    uploader.hub.upload_repo = wrapped_upload_repo
    uploader.hub._skip_backup_hook_installed = True
