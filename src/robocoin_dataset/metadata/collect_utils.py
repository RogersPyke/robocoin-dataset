"""
Metadata field resolution and context building utilities.

Purpose:
    Core logic for resolving all schema fields from distributed dataset sources
    (local_dataset_info.yaml, meta/info.json, annotations/, etc.).
    
    This module is the centerpiece of the Collect stage and is used exclusively
    by metadata/collect.py. It has no dependencies on readme/ modules, ensuring
    clean separation between Collect (metadata) and Render (README) stages.

Dependencies:
    - json, re, pathlib: Data loading and manipulation
    - yaml: YAML parsing
    - logging: Audit logging
    - robocoin_dataset.utils.log_config: Colored logging

Usage:
    Called by InfoCollector.collect() to resolve all fields and build context dict.
    Not intended for direct use outside metadata/collect.py.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from robocoin_dataset.utils.log_config import log_error


# ============================================================================
# YAML Loading (independent of readme module)
# ============================================================================


def load_yaml_file(yaml_path: Path, logger: logging.Logger) -> Dict[str, Any]:
    """
    Load and parse YAML file into dictionary.

    Input:
        yaml_path (Path): Path to the YAML file to load.
        logger (logging.Logger): Logger instance for error reporting.

    Output:
        Dict[str, Any]: Parsed YAML content as dictionary.
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
    Load info.yaml schema as raw dictionary.

    Input:
        schema_yaml_path (Path): Absolute path to schema YAML (assets/info.yaml).
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


# ============================================================================
# Field resolution helpers
# ============================================================================


def _normalize_aliases(alias_value: Any) -> List[str]:
    if alias_value is None:
        return []
    if isinstance(alias_value, str):
        return [alias_value]
    if isinstance(alias_value, list):
        return [item for item in alias_value if isinstance(item, str)]
    return []


def _resolve_from_mapping(
    data: Dict[str, Any],
    field_name: str,
    aliases: List[str],
) -> Any:
    if field_name in data and data[field_name] not in (None, ""):
        return data[field_name]
    for alias in aliases:
        if alias in data and data[alias] not in (None, ""):
            return data[alias]
    return None


def _extract_scene_type_from_local_dataset_info(local_dataset_info: Dict[str, Any]) -> Dict[str, Any] | None:
    """
    Extract hierarchical scene levels from local_dataset_info.yaml.

    Input:
        local_dataset_info (Dict[str, Any]): Parsed local dataset metadata.

    Output:
        Dict[str, Any] | None: Mapping like {"level1": "...", "level2": "..."} when available.
    """
    scene_levels: Dict[str, Any] = {}
    has_any_level = False
    for i in range(1, 6):
        key = f"scene_level{i}"
        value = local_dataset_info.get(key)
        if value not in (None, ""):
            has_any_level = True
            scene_levels[f"level{i}"] = value
        else:
            scene_levels[f"level{i}"] = None

    if has_any_level:
        return scene_levels

    legacy_scene_level = local_dataset_info.get("scene_level")
    if isinstance(legacy_scene_level, dict):
        normalized = {f"level{i}": legacy_scene_level.get(f"level{i}") for i in range(1, 6)}
        if any(v not in (None, "") for v in normalized.values()):
            return normalized
    if isinstance(legacy_scene_level, list):
        normalized = {}
        for i in range(1, 6):
            idx = i - 1
            normalized[f"level{i}"] = legacy_scene_level[idx] if idx < len(legacy_scene_level) else None
        if any(v not in (None, "") for v in normalized.values()):
            return normalized
    if isinstance(legacy_scene_level, str) and legacy_scene_level.strip():
        parts = [part.strip() for part in legacy_scene_level.split("-") if part.strip()]
        normalized = {}
        for i in range(1, 6):
            idx = i - 1
            normalized[f"level{i}"] = parts[idx] if idx < len(parts) else None
        return normalized

    return None


def _load_json_file(json_path: Path, logger: logging.Logger) -> Dict[str, Any]:
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


def _load_jsonl_file(jsonl_path: Path) -> List[Dict[str, Any]]:
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


def _extract_sensor_list_from_features(features: Any) -> List[str]:
    sensors: List[str] = []
    if not isinstance(features, dict):
        return sensors

    for key in features:
        if isinstance(key, str) and key.startswith("observation.images."):
            sensor_name = key.split("observation.images.", 1)[1]
            if sensor_name and sensor_name not in sensors:
                sensors.append(sensor_name)

    observation = features.get("observation")
    if isinstance(observation, dict):
        images = observation.get("images")
        if isinstance(images, dict):
            for sensor_name in images.keys():
                if isinstance(sensor_name, str) and sensor_name not in sensors:
                    sensors.append(sensor_name)
    return sensors


def _extract_camera_info_from_features(features: Any) -> Dict[str, str]:
    camera_info: Dict[str, str] = {}
    if not isinstance(features, dict):
        return camera_info

    def _format_cam_spec(cam_feature: Dict[str, Any]) -> str:
        dtype = cam_feature.get("dtype")
        shape = cam_feature.get("shape")
        info = cam_feature.get("info") if isinstance(cam_feature.get("info"), dict) else {}

        shape_str = None
        if isinstance(shape, list) and len(shape) >= 2 and all(
            isinstance(v, int) for v in shape
        ):
            shape_str = "x".join(str(v) for v in shape)

        width = info.get("video.width")
        height = info.get("video.height")
        codec = info.get("video.codec")
        pix_fmt = info.get("video.pix_fmt")

        parts: List[str] = []
        if isinstance(dtype, str) and dtype:
            parts.append(f"dtype={dtype}")
        if shape_str:
            parts.append(f"shape={shape_str}")
        if isinstance(width, int) and isinstance(height, int):
            parts.append(f"resolution={width}x{height}")
        if isinstance(codec, str) and codec:
            parts.append(f"codec={codec}")
        if isinstance(pix_fmt, str) and pix_fmt:
            parts.append(f"pix_fmt={pix_fmt}")
        return ", ".join(parts) if parts else "unknown"

    for key, value in features.items():
        if isinstance(key, str) and key.startswith("observation.images.") and isinstance(value, dict):
            sensor_name = key.split("observation.images.", 1)[1]
            camera_info[sensor_name] = _format_cam_spec(value)

    observation = features.get("observation")
    if isinstance(observation, dict):
        images = observation.get("images")
        if isinstance(images, dict):
            for sensor_name, value in images.items():
                if isinstance(sensor_name, str) and isinstance(value, dict):
                    camera_info[sensor_name] = _format_cam_spec(value)
    return camera_info


def _extract_depth_enabled_from_features(features: Any) -> bool:
    if not isinstance(features, dict):
        return False

    for key, value in features.items():
        key_str = str(key).lower()
        if "depth" in key_str:
            return True
        if isinstance(value, dict):
            info = value.get("info")
            if isinstance(info, dict) and bool(info.get("video.is_depth_map")):
                return True
            shape = value.get("shape")
            if isinstance(shape, list) and len(shape) >= 3:
                channels = shape[-1]
                if isinstance(channels, int) and channels >= 4:
                    return True
    return False


def _infer_unit_from_names(names: List[str], keywords: List[str], suffix_to_unit: Dict[str, str]) -> str | None:
    """
    Infer a physical unit from feature names by keyword and suffix.

    Input:
        names (List[str]): Feature name list.
        keywords (List[str]): Required keyword fragments.
        suffix_to_unit (Dict[str, str]): Mapping from suffix token to unit label.

    Output:
        str | None: Resolved unit text, or None if unknown.
    """
    for raw_name in names:
        name = str(raw_name).lower()
        if not all(keyword in name for keyword in keywords):
            continue
        for suffix, unit in suffix_to_unit.items():
            if name.endswith(suffix):
                return unit
    return None


def _extract_dimension_units_from_features(features: Any) -> Dict[str, str]:
    """
    Extract dimension unit fields from meta/info.json features.

    Input:
        features (Any): Source features object from meta/info.json.

    Output:
        Dict[str, str]: Unit mapping for known dimension fields.
    """
    if not isinstance(features, dict):
        return {}
    state_def = features.get("observation.state")
    names = state_def.get("names") if isinstance(state_def, dict) else None
    if not isinstance(names, list):
        return {}

    inferred: Dict[str, str] = {}
    joint_rot = _infer_unit_from_names(
        names=names,
        keywords=["joint"],
        suffix_to_unit={"_rad": "radian", "_deg": "degree"},
    )
    if joint_rot:
        inferred["joint_rotation_dim"] = joint_rot

    eef_rot = _infer_unit_from_names(
        names=names,
        keywords=["eef", "rot"],
        suffix_to_unit={"_rad": "radian", "_deg": "degree"},
    )
    if eef_rot:
        inferred["end_rotation_dim"] = eef_rot

    eef_trans = _infer_unit_from_names(
        names=names,
        keywords=["eef", "pos"],
        suffix_to_unit={"_mm": "millimeter", "_m": "meter"},
    )
    if eef_trans:
        inferred["end_translation_dim"] = eef_trans

    base_rot = _infer_unit_from_names(
        names=names,
        keywords=["base", "rot"],
        suffix_to_unit={"_rad": "radian", "_deg": "degree"},
    )
    if base_rot:
        inferred["base_robtation_dim"] = base_rot

    base_trans = _infer_unit_from_names(
        names=names,
        keywords=["base", "pos"],
        suffix_to_unit={"_mm": "millimeter", "_m": "meter"},
    )
    if base_trans:
        inferred["base_translation_dim"] = base_trans

    return inferred


def _format_size_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(max(num_bytes, 0))
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def _calculate_directory_size_bytes(root_path: Path) -> int:
    total_bytes = 0
    for file_path in root_path.rglob("*"):
        if file_path.is_file():
            try:
                total_bytes += file_path.stat().st_size
            except OSError:
                continue
    return total_bytes


def _infer_file_pattern(dataset_path: Path, suffix: str, fallback: str) -> str:
    matched_file = None
    for file_path in dataset_path.rglob(f"*{suffix}"):
        if file_path.is_file():
            matched_file = file_path
            break
    if matched_file is None:
        return fallback
    rel_path = matched_file.relative_to(dataset_path).as_posix()
    return re.sub(r"\d+", "{id}", rel_path)


def _build_data_structure_tree(
    root_path: Path,
    depth_limit: int = 3,
    entries_per_level_limit: int = 12,
) -> str:
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


def _extract_sub_tasks_from_annotations(records: List[Dict[str, Any]]) -> List[Any]:
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

        key = json.dumps(item, sort_keys=True, ensure_ascii=True)
        if key not in seen:
            extracted.append(item)
            seen.add(key)
    return extracted


def _extract_task_result_from_annotations(records: List[Dict[str, Any]]) -> Any:
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


def _build_statistics_from_sources(
    meta_info: Dict[str, Any],
    episode_stats_rows: List[Dict[str, Any]],
    dataset_path: Path,
    sub_tasks: Any = None,
) -> Dict[str, Any]:
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
        camera_views = _extract_sensor_list_from_features(features)
        if camera_views:
            stats["camera_views"] = len(camera_views)
    return stats


def _resolve_source_path(dataset_path: Path, source_rel: str) -> Path | None:
    source_path = dataset_path / source_rel
    if source_path.exists():
        return source_path

    if source_rel.startswith("annotations/") and source_rel.endswith(".jsonl"):
        annotation_files = sorted((dataset_path / "annotations").glob("*.jsonl"))
        if source_rel.endswith("subtask_annotations.jsonl"):
            subtask_candidates = [
                p for p in annotation_files if "subtask" in p.name or "task" in p.name
            ]
            if len(subtask_candidates) == 1:
                return subtask_candidates[0]
        if len(annotation_files) == 1:
            return annotation_files[0]
    return None


def _resolve_value_from_source(
    field_name: str,
    source: Any,
    aliases: List[str],
    default_value: Any,
    dataset_path: Path,
    local_dataset_info: Dict[str, Any],
    source_cache: Dict[str, Any],
    logger: logging.Logger,
) -> Any:
    if source in (None, "", "auto", "fixed", "urdf"):
        return None

    if source == "local_dataset_info.yaml":
        if field_name == "scene_type":
            scene_type = _extract_scene_type_from_local_dataset_info(local_dataset_info)
            if scene_type is not None:
                return scene_type
        return _resolve_from_mapping(local_dataset_info, field_name, aliases)

    source_rel = str(source)
    if source_rel.rstrip("/") == "annotations":
        annotations_dir = dataset_path / "annotations"
        if not annotations_dir.exists() or not annotations_dir.is_dir():
            return None
        annotation_files = sorted(
            [p.name for p in annotations_dir.iterdir() if p.is_file()]
        )
        return annotation_files if annotation_files else None

    source_path = _resolve_source_path(dataset_path, source_rel)
    if source_path is None:
        logger.warning(
            f"[SOURCE_RESOLVE] Source file for '{field_name}' does not exist: "
            f"{dataset_path / source_rel}"
        )
        return None

    cache_key = source_path.as_posix()
    if cache_key not in source_cache:
        try:
            if source_path.suffix == ".json":
                source_cache[cache_key] = _load_json_file(source_path, logger)
            elif source_path.suffix == ".jsonl":
                source_cache[cache_key] = _load_jsonl_file(source_path)
            elif source_path.suffix in (".yaml", ".yml"):
                source_cache[cache_key] = load_yaml_file(source_path, logger)
            else:
                source_cache[cache_key] = None
        except Exception as e:
            logger.warning(f"[SOURCE_RESOLVE] Failed loading source file {source_path}: {e}")
            source_cache[cache_key] = None

    source_data = source_cache.get(cache_key)
    if isinstance(source_data, dict):
        resolved = _resolve_from_mapping(source_data, field_name, aliases)
        if resolved not in (None, ""):
            return resolved

    source_path_str = source_path.as_posix()
    if source_path_str.endswith("meta/info.json") and isinstance(source_data, dict):
        features = source_data.get("features")
        if field_name == "sensor_list":
            sensors = _extract_sensor_list_from_features(features)
            return sensors if sensors else None
        if field_name == "came_info":
            cam_info = _extract_camera_info_from_features(features)
            return cam_info if cam_info else None
        if field_name == "depth_enabled":
            return _extract_depth_enabled_from_features(features)
        if field_name == "splits":
            if isinstance(source_data.get("splits"), dict):
                return source_data.get("splits")
            episodes = source_data.get("total_episodes")
            if isinstance(episodes, int) and episodes > 0:
                return {"train": f"0:{episodes - 1}"}
            return {"train": "all"}
        if field_name == "features":
            return features
        if field_name == "frame_num":
            total_frames = source_data.get("total_frames") or source_data.get("frame_num")
            if isinstance(total_frames, int):
                return total_frames
        if field_name in (
            "joint_rotation_dim",
            "end_rotation_dim",
            "end_translation_dim",
            "base_robtation_dim",
            "base_translation_dim",
        ):
            inferred_units = _extract_dimension_units_from_features(features)
            value = inferred_units.get(field_name)
            return value if value not in (None, "") else None

    if source_path.name.endswith(".jsonl") and isinstance(source_data, list):
        if field_name == "sub_tasks":
            sub_tasks = _extract_sub_tasks_from_annotations(source_data)
            return sub_tasks if sub_tasks else None
        if field_name == "task_result":
            return _extract_task_result_from_annotations(source_data)

    if source_path_str.endswith("meta/episodes_stats.jsonl") and isinstance(source_data, list):
        if field_name == "statistics":
            meta_cache_key = (dataset_path / "meta/info.json").as_posix()
            meta_info = source_cache.get(meta_cache_key)
            if not isinstance(meta_info, dict):
                meta_info = {}
            sub_tasks = source_cache.get("__resolved_sub_tasks__", default_value)
            return _build_statistics_from_sources(
                meta_info=meta_info,
                episode_stats_rows=source_data,
                dataset_path=dataset_path,
                sub_tasks=sub_tasks,
            )

    return None


def _apply_auto_fields(
    context_data: Dict[str, Any],
    dataset_path: Path,
    logger: logging.Logger,
) -> None:
    if context_data.get("dataset_size") in (None, "", "dataset_size"):
        total_bytes = _calculate_directory_size_bytes(dataset_path)
        context_data["dataset_size"] = _format_size_bytes(total_bytes)

    if context_data.get("data_structure") in (None, "", "data_structure_str"):
        context_data["data_structure"] = _build_data_structure_tree(dataset_path)

    if context_data.get("data_path") in (None, "", "data_path"):
        meta_data_path = None
        stats_data = context_data.get("statistics")
        if isinstance(stats_data, dict):
            meta_data_path = stats_data.get("data_path")
        context_data["data_path"] = meta_data_path or _infer_file_pattern(
            dataset_path=dataset_path,
            suffix=".parquet",
            fallback="data/chunk-{id}/episode_{id}.parquet",
        )

    if context_data.get("video_path") in (None, "", "video_path"):
        context_data["video_path"] = _infer_file_pattern(
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


def _apply_template_compatibility(context_data: Dict[str, Any]) -> None:
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


def resolve_context_from_schema(
    schema_yaml_path: Path,
    dataset_path: Path,
    local_dataset_info_path: Path,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """
    Resolve all schema fields and build a flat context dictionary.

    This is the core context-building function used by InfoCollector to resolve
    all metadata and prepare it for serialization to info.yaml.

    Input:
        schema_yaml_path (Path): Path to schema info.yaml.
        dataset_path (Path): Dataset root directory.
        local_dataset_info_path (Path): Path to local_dataset_info.yaml.
        logger (logging.Logger): Logger instance.

    Output:
        Dict[str, Any]: Flat context dict with all resolved fields.
            This dict includes:
              - Fields from schema with resolved values
              - Auto-computed fields (dataset_size, data_structure, etc.)
              - Template compatibility transformations applied

    Logic:
        1. Load schema and local_dataset_info.
        2. Iterate schema fields; for each field, resolve its value:
           - If source=="fixed": use explicit value or default.
           - Otherwise: load from source file(s), apply transformations.
        3. Apply auto-field computation and template compatibility.
        4. Return flat dict suitable for YAML serialization.
    """
    schema = load_schema_yaml(schema_yaml_path, logger)
    local_dataset_info = load_yaml_file(local_dataset_info_path, logger)

    context_data: Dict[str, Any] = {}
    source_cache: Dict[str, Any] = {}
    source_type_counter: Dict[str, int] = {}

    for field_name, field_spec in schema.items():
        if not isinstance(field_spec, dict):
            context_data[field_name] = field_spec
            continue

        source = field_spec.get("source")
        required_status = field_spec.get("required")
        source_key = str(source)
        source_type_counter[source_key] = source_type_counter.get(source_key, 0) + 1

        if required_status is False:
            continue

        aliases = _normalize_aliases(field_spec.get("alias"))
        default_value = field_spec.get("default")

        if source == "fixed":
            explicit_value = field_spec.get("value")
            if explicit_value not in (None, ""):
                context_data[field_name] = explicit_value
            elif required_status is True:
                context_data[field_name] = default_value
            elif required_status == "optional":
                if default_value not in (None, ""):
                    context_data[field_name] = default_value
            continue

        resolved = _resolve_value_from_source(
            field_name=field_name,
            source=source,
            aliases=aliases,
            default_value=default_value,
            dataset_path=dataset_path,
            local_dataset_info=local_dataset_info,
            source_cache=source_cache,
            logger=logger,
        )
        if field_name == "sub_tasks" and resolved not in (None, ""):
            source_cache["__resolved_sub_tasks__"] = resolved

        explicit_value = field_spec.get("value")
        if explicit_value not in (None, ""):
            context_data[field_name] = explicit_value
        elif resolved not in (None, ""):
            if required_status is True or required_status == "optional":
                context_data[field_name] = resolved
        elif required_status is True:
            context_data[field_name] = default_value
        elif required_status == "optional":
            pass
        else:
            pass

    _apply_auto_fields(context_data=context_data, dataset_path=dataset_path, logger=logger)
    _apply_template_compatibility(context_data=context_data)

    logger.info(
        "[RESOLVE_CONTEXT] Context resolved. Source types: "
        + ", ".join(
            [f"{k}={v}" for k, v in sorted(source_type_counter.items())]
        )
    )
    return context_data
