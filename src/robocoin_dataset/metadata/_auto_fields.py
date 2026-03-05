"""
Auto-field computation and statistics building for the Collect stage.

Responsibilities:
- Compute dataset_size, data_structure tree, data_path / video_path patterns.
- Build the ``statistics`` sub-dict from episode_stats rows + meta/info.json.
- Apply template-compatibility normalisation (task_categories, splits, configs, …).
- Extract annotation-level sub-task and task-result data.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List

from robocoin_dataset.metadata._feature_extractors import extract_sensor_list_from_features


# ============================================================================
# Filesystem helpers
# ============================================================================


def format_size_bytes(num_bytes: int) -> str:
    """Convert a byte count to a human-readable size string (e.g. ``"4.20 GB"``)."""
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(max(num_bytes, 0))
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def calculate_directory_size_bytes(root_path: Path) -> int:
    """Return the total size in bytes of all files under ``root_path``."""
    total_bytes = 0
    for file_path in root_path.rglob("*"):
        if file_path.is_file():
            try:
                total_bytes += file_path.stat().st_size
            except OSError:
                continue
    return total_bytes


def infer_file_pattern(dataset_path: Path, suffix: str, fallback: str) -> str:
    """
    Infer a path pattern such as ``data/chunk-{id}/episode_{id}.parquet`` by
    finding the first matching file and replacing all digit sequences with ``{id}``.

    Input:
        dataset_path (Path): Dataset root.
        suffix (str): File extension to search for (e.g. ``".parquet"``).
        fallback (str): Pattern to return when no file is found.

    Output:
        str: Inferred path pattern string.
    """
    for file_path in dataset_path.rglob(f"*{suffix}"):
        if file_path.is_file():
            rel_path = file_path.relative_to(dataset_path).as_posix()
            return re.sub(r"\d+", "{id}", rel_path)
    return fallback


def build_data_structure_tree(
    root_path: Path,
    depth_limit: int = 3,
    entries_per_level_limit: int = 12,
) -> str:
    """
    Build an ASCII directory tree string for ``root_path``.

    Input:
        root_path (Path): Root directory to render.
        depth_limit (int): Maximum depth to traverse.
        entries_per_level_limit (int): Max entries shown per directory level.

    Output:
        str: Multi-line ASCII tree.
    """
    lines: List[str] = [f"{root_path.name}/"]

    def _walk(current_path: Path, prefix: str, depth: int) -> None:
        if depth >= depth_limit:
            return
        try:
            entries = sorted(
                list(current_path.iterdir()),
                key=lambda p: (p.is_file(), p.name.lower()),
            )
        except OSError:
            return

        trimmed_entries = entries[:entries_per_level_limit]
        for idx, entry in enumerate(trimmed_entries):
            connector = "`-- " if idx == len(trimmed_entries) - 1 else "|-- "
            lines.append(f"{prefix}{connector}{entry.name}")
            if entry.is_dir():
                next_prefix = prefix + ("    " if idx == len(trimmed_entries) - 1 else "|   ")
                _walk(entry, next_prefix, depth + 1)

        hidden_count = len(entries) - len(trimmed_entries)
        if hidden_count > 0:
            lines.append(f"{prefix}`-- ... ({hidden_count} more entries)")

    _walk(root_path, "", 0)
    return "\n".join(lines)


# ============================================================================
# Annotation extraction
# ============================================================================


def extract_sub_tasks_from_annotations(records: List[Dict[str, Any]]) -> List[Any]:
    """
    Extract unique sub-task entries from annotation JSONL records.

    Input:
        records (List[Dict[str, Any]]): Loaded annotation rows.

    Output:
        List[Any]: Deduplicated sub-task entries (plain strings or
            ``{"subtask": ..., "subtask_index": ...}`` dicts).
    """
    import json as _json

    extracted: List[Any] = []
    seen: set[str] = set()
    for row in records:
        subtask = (
            row.get("subtask")
            or row.get("sub_task")
            or row.get("task_description")
            or row.get("description")
        )
        if subtask in (None, ""):
            continue

        item: Any = subtask
        if "subtask_index" in row:
            item = {"subtask": subtask, "subtask_index": row.get("subtask_index")}

        key = _json.dumps(item, sort_keys=True, ensure_ascii=True)
        if key not in seen:
            extracted.append(item)
            seen.add(key)
    return extracted


def extract_task_result_from_annotations(records: List[Dict[str, Any]]) -> Any:
    """
    Extract distinct task-result values from annotation JSONL records.

    Input:
        records (List[Dict[str, Any]]): Loaded annotation rows.

    Output:
        str | List[str] | None: Single string, list, or None if not found.
    """
    result_values: List[str] = []
    for row in records:
        value = (
            row.get("task_result")
            or row.get("result")
            or row.get("status")
            or row.get("label")
        )
        if isinstance(value, bool):
            value = "success" if value else "failure"
        if value in (None, ""):
            continue
        value_str = str(value)
        if value_str not in result_values:
            result_values.append(value_str)

    if not result_values:
        return None
    if len(result_values) == 1:
        return result_values[0]
    return result_values


# ============================================================================
# Statistics builder
# ============================================================================


def _extract_frames_from_episode_stats_row(row: Dict[str, Any]) -> int | None:
    frame_count = row.get("length") or row.get("num_frames") or row.get("frames")
    if isinstance(frame_count, int):
        return frame_count

    stats = row.get("stats")
    if not isinstance(stats, dict):
        return None
    frame_stats = stats.get("frame_index")
    if not isinstance(frame_stats, dict):
        return None
    count = frame_stats.get("count")
    if isinstance(count, list) and count and isinstance(count[0], int):
        return int(count[0])
    if isinstance(count, int):
        return int(count)
    return None


def build_statistics_from_sources(
    meta_info: Dict[str, Any],
    episode_stats_rows: List[Dict[str, Any]],
    dataset_path: Path,
    sub_tasks: Any = None,
) -> Dict[str, Any]:
    """
    Aggregate episode statistics from meta/info.json and meta/episodes_stats.jsonl.

    Input:
        meta_info (Dict[str, Any]): Parsed meta/info.json.
        episode_stats_rows (List[Dict[str, Any]]): Parsed meta/episodes_stats.jsonl rows.
        dataset_path (Path): Dataset root (used to count videos / chunks as fallback).
        sub_tasks (Any): Already-resolved sub_tasks list (for total_tasks count).

    Output:
        Dict[str, Any]: ``statistics`` sub-dict with keys such as
            total_episodes, total_frames, fps, total_tasks, total_videos,
            total_chunks, state_dim, action_dim, camera_views.
    """
    stats: Dict[str, Any] = {}

    total_episodes = len(episode_stats_rows) if episode_stats_rows else None
    if total_episodes is not None:
        stats["total_episodes"] = total_episodes

    if episode_stats_rows:
        total_frames = 0
        fps_candidates: List[float] = []
        for row in episode_stats_rows:
            frame_count = _extract_frames_from_episode_stats_row(row)
            if isinstance(frame_count, int):
                total_frames += frame_count
            fps_value = row.get("fps")
            if isinstance(fps_value, (int, float)):
                fps_candidates.append(float(fps_value))
        if total_frames > 0:
            stats["total_frames"] = total_frames
        if fps_candidates:
            stats["fps"] = round(sum(fps_candidates) / len(fps_candidates), 2)

    if "fps" not in stats:
        meta_fps = meta_info.get("fps")
        if isinstance(meta_fps, (int, float)):
            stats["fps"] = meta_fps
    if "total_frames" not in stats and isinstance(meta_info.get("total_frames"), int):
        stats["total_frames"] = meta_info["total_frames"]

    if isinstance(sub_tasks, list):
        stats["total_tasks"] = len(sub_tasks)
    elif isinstance(meta_info.get("total_tasks"), int):
        stats["total_tasks"] = meta_info["total_tasks"]

    if isinstance(meta_info.get("total_videos"), int):
        stats["total_videos"] = meta_info["total_videos"]
    else:
        video_files = [p for p in dataset_path.rglob("*.mp4") if p.is_file()]
        if video_files:
            stats["total_videos"] = len(video_files)

    if isinstance(meta_info.get("total_chunks"), int):
        stats["total_chunks"] = meta_info["total_chunks"]
    else:
        chunk_dirs = [p for p in (dataset_path / "data").glob("chunk-*") if p.is_dir()]
        if chunk_dirs:
            stats["total_chunks"] = len(chunk_dirs)
        else:
            parquet_files = [p for p in dataset_path.rglob("*.parquet") if p.is_file()]
            if parquet_files:
                stats["total_chunks"] = len(parquet_files)

    if isinstance(meta_info.get("chunks_size"), int):
        stats["chunks_size"] = meta_info["chunks_size"]

    features = meta_info.get("features")
    if isinstance(features, dict):
        state_def = features.get("observation.state")
        action_def = features.get("action")
        if isinstance(state_def, dict) and isinstance(state_def.get("shape"), list):
            shape = state_def["shape"]
            if shape and isinstance(shape[0], int):
                stats["state_dim"] = shape[0]
        if isinstance(action_def, dict) and isinstance(action_def.get("shape"), list):
            shape = action_def["shape"]
            if shape and isinstance(shape[0], int):
                stats["action_dim"] = shape[0]
        camera_views = extract_sensor_list_from_features(features)
        if camera_views:
            stats["camera_views"] = len(camera_views)
    return stats


# ============================================================================
# Auto-field completion and template compatibility
# ============================================================================


def apply_auto_fields(
    context_data: Dict[str, Any],
    dataset_path: Path,
    logger: logging.Logger,
) -> None:
    """
    Fill fields that are computed from the filesystem rather than a source file.

    Mutates ``context_data`` in-place. Fills: dataset_size, data_structure,
    data_path, video_path, annotations, sub_tasks, and ensures statistics is a dict.

    Input:
        context_data (Dict[str, Any]): Partially resolved context dict.
        dataset_path (Path): Dataset root directory.
        logger (logging.Logger): Logger instance.
    """
    if context_data.get("dataset_size") in (None, "", "dataset_size"):
        total_bytes = calculate_directory_size_bytes(dataset_path)
        context_data["dataset_size"] = format_size_bytes(total_bytes)

    if context_data.get("data_structure") in (None, "", "data_structure_str"):
        context_data["data_structure"] = build_data_structure_tree(dataset_path)

    if context_data.get("data_path") in (None, "", "data_path"):
        meta_data_path = None
        stats_data = context_data.get("statistics")
        if isinstance(stats_data, dict):
            meta_data_path = stats_data.get("data_path")
        context_data["data_path"] = meta_data_path or infer_file_pattern(
            dataset_path=dataset_path,
            suffix=".parquet",
            fallback="data/chunk-{id}/episode_{id}.parquet",
        )

    if context_data.get("video_path") in (None, "", "video_path"):
        context_data["video_path"] = infer_file_pattern(
            dataset_path=dataset_path,
            suffix=".mp4",
            fallback="videos/chunk-{id}/{video_key}/episode_{id}.mp4",
        )

    if "annotations" in context_data and context_data.get("annotations") in (
        None,
        "",
        ["annotation_1", "annotation_2", "annotation_3"],
    ):
        annotations_dir = dataset_path / "annotations"
        if annotations_dir.exists() and annotations_dir.is_dir():
            ann_files = sorted([p.name for p in annotations_dir.iterdir() if p.is_file()])
            if ann_files:
                context_data["annotations"] = ann_files

    if "sub_tasks" in context_data and context_data.get("sub_tasks") in (
        None,
        "",
        ["sub_task_1", "sub_task_2", "sub_task_3"],
    ):
        task_instruction = context_data.get("task_instruction")
        if isinstance(task_instruction, list) and task_instruction:
            context_data["sub_tasks"] = task_instruction
        elif isinstance(task_instruction, str) and task_instruction.strip():
            context_data["sub_tasks"] = [task_instruction]

    stats = context_data.get("statistics")
    if not isinstance(stats, dict):
        stats = {}
    if context_data.get("dataset_size"):
        stats.setdefault("dataset_size", context_data["dataset_size"])
    context_data["statistics"] = stats

    logger.info("[AUTO_FIELDS] Auto field completion finished")


def apply_template_compatibility(context_data: Dict[str, Any]) -> None:
    """
    Normalise field values so the Jinja2 template receives expected types.

    Mutates ``context_data`` in-place. Handles: task_categories, language,
    configs, splits, and the base_robtation_dim typo alias.

    Input:
        context_data (Dict[str, Any]): Resolved context dict.
    """
    if isinstance(context_data.get("task_categories"), str):
        context_data["task_categories"] = [context_data["task_categories"]]
    if isinstance(context_data.get("language"), str):
        context_data["language"] = [context_data["language"]]

    configs = context_data.get("configs")
    if isinstance(configs, str):
        context_data["configs"] = [
            {
                "config_name": configs,
                "data_files": context_data.get("data_path", "data/**"),
            }
        ]
    elif isinstance(configs, dict):
        context_data["configs"] = [configs]
    elif not isinstance(configs, list):
        context_data["configs"] = []

    splits = context_data.get("splits")
    if isinstance(splits, str):
        context_data["splits"] = {"train": splits}
    elif not isinstance(splits, dict):
        context_data["splits"] = {"train": "all"}

    if "base_robtation_dim" in context_data and "base_rotation_dim" not in context_data:
        context_data["base_rotation_dim"] = context_data["base_robtation_dim"]
