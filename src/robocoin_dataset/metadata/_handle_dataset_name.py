#!/usr/bin/env python3
"""
_handle_dataset_name.py

This script does exactly one thing:
- Ensure `dataset_name` starts with the robot name (derived from `robot_name`
  or legacy `device_model` in local_dataset_info.yaml).

If `dataset_name` does not start with the robot name, this script prefixes
and normalizes it into the canonical form:
`<robot_name>_<operation_task_name>`.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Optional

import yaml


SUPPORTED_YAML_NAMES = {"local_dataset_info.yaml", "local_dataset_info.yml"}


def _coerce_str(value: Any) -> Optional[str]:
    """
    Normalize a value into a non-empty string.
    - `str` -> stripped string
    - `list[str]` -> first non-empty element
    - otherwise -> None
    """
    if isinstance(value, str):
        v = value.strip()
        return v if v else None
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str) and item.strip():
                return item.strip()
    return None


def _as_robot_name(data: dict[str, Any]) -> Optional[str]:
    """
    Derive the robot name from either:
    - `robot_name` (preferred), or
    - legacy `device_model` (may be string or list[str]).
    """
    robot_name = _coerce_str(data.get("robot_name"))
    if robot_name:
        return robot_name

    device_model = data.get("device_model")
    if isinstance(device_model, list) and device_model:
        # legacy examples usually use ["Galaxea_R1_Lite"]
        for item in device_model:
            if isinstance(item, str) and item.strip():
                return item.strip()
    if isinstance(device_model, str) and device_model.strip():
        return device_model.strip()

    return None


def _as_dataset_name(data: dict[str, Any]) -> Optional[str]:
    return _coerce_str(data.get("dataset_name"))


def _normalize_dataset_name(robot_name: str, dataset_name: str) -> str:
    """
    Canonicalize dataset_name into: `<robot_name>_<operation_task_name>`

    Rules:
    - If dataset_name already starts with robot_name, normalize the delimiter
      right after robot_name into '_'.
    - If dataset_name doesn't start with robot_name, prefix it with '_'.
    """
    robot_name = robot_name.strip()
    dataset_name = dataset_name.strip()

    if dataset_name.startswith(robot_name):
        rest = dataset_name[len(robot_name) :]
        # Remove leading separators/spaces; keep the remaining operation task.
        rest = rest.lstrip("_ ").lstrip("-")
        if not rest:
            return robot_name
        return f"{robot_name}_{rest}"

    return f"{robot_name}_{dataset_name}"


def patch_dataset_name_in_context(context_data: dict[str, Any]) -> None:
    """
    In-place patch for the Collect stage output context.

    We normalize the final output `dataset_name` so that README/info.yaml
    always use `<robot_name>_<operation_task_name>`, even when the original
    local_dataset_info.yaml uses legacy or shortened forms.
    """
    dataset_name = _coerce_str(context_data.get("dataset_name"))
    robot_name = _coerce_str(context_data.get("robot_name"))
    if not dataset_name or not robot_name:
        return

    context_data["dataset_name"] = _normalize_dataset_name(robot_name, dataset_name)


def _patch_one_yaml(yaml_path: Path, dry_run: bool) -> bool:
    try:
        with yaml_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return False

    robot_name = _as_robot_name(data)
    dataset_name = _as_dataset_name(data)
    if not robot_name or not dataset_name:
        return False

    patched = _normalize_dataset_name(robot_name, dataset_name)
    if patched == dataset_name:
        return False
    data["dataset_name"] = patched

    if dry_run:
        return True

    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False, indent=2)
    return True


def _iter_local_dataset_info_paths(scan_root: Path) -> list[Path]:
    found: list[Path] = []
    for current, dirnames, files in os.walk(scan_root):
        if SUPPORTED_YAML_NAMES & set(files):
            filename = next(iter(SUPPORTED_YAML_NAMES & set(files)))
            found.append(Path(current) / filename)
            # do not recurse into subdirectories of a dataset folder
            dirnames.clear()
    return found


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize dataset_name into `<robot_name>_<operation_task_name>` (legacy compat)."
    )
    parser.add_argument("scan_root", type=str, help="Root folder to scan for local_dataset_info.yaml/.yml")
    parser.add_argument("--dry-run", action="store_true", help="Do not modify files; just report.")
    args = parser.parse_args()

    scan_root = Path(args.scan_root).expanduser().resolve()
    if not scan_root.exists() or not scan_root.is_dir():
        raise SystemExit(f"scan_root is not a directory: {scan_root}")

    paths = _iter_local_dataset_info_paths(scan_root)
    changed = 0
    for p in paths:
        if _patch_one_yaml(p, dry_run=args.dry_run):
            changed += 1
            print(f"{'[dry-run] ' if args.dry_run else ''}Patched: {p}")

    print(f"Total matched: {len(paths)}, total patched: {changed}")


if __name__ == "__main__":
    main()

