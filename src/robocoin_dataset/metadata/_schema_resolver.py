"""
Schema-driven field resolution for the Collect stage.

This module contains the core logic that iterates the assets/info.yaml schema,
loads each field's declared source file, applies field-specific extraction
logic, and returns a flat context dictionary.

Public API:
    resolve_context_from_schema(...)  — master entry point used by InfoCollector

All heavy I/O (YAML/JSON/JSONL loading) is delegated to _yaml_io.
All feature extraction is delegated to _feature_extractors.
All auto-field and compat transforms are delegated to _auto_fields.

Schema ``required`` (metadata/assets/info.yaml):
    - ``true``: if the source does not resolve, fall back to schema ``default`` (legacy).
    - ``strict``: must resolve from source; refuse schema defaults and refuse values that
      still equal the schema default placeholder. Raises ValueError on failure.
    - ``optional`` / ``false``: unchanged.

"""

import logging
from pathlib import Path
from typing import Any, Dict, List

from robocoin_dataset.metadata._auto_fields import (
    apply_auto_fields,
    apply_template_compatibility,
    build_statistics_from_sources,
    extract_sub_tasks_from_annotations,
    extract_task_result_from_annotations,
)
from robocoin_dataset.metadata._feature_extractors import (
    extract_camera_info_from_features,
    extract_depth_enabled_from_features,
    extract_dimension_units_from_features,
    extract_sensor_list_from_features,
)
from robocoin_dataset.metadata._yaml_io import (
    load_json_file,
    load_jsonl_file,
    load_schema_yaml,
    load_yaml_file,
)


# ============================================================================
# Field-level resolution helpers
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


def _extract_scene_type_from_local_dataset_info(
    local_dataset_info: Dict[str, Any],
) -> Dict[str, Any] | None:
    """
    Extract hierarchical scene levels from local_dataset_info.yaml.

    Input:
        local_dataset_info (Dict[str, Any]): Parsed local dataset metadata.

    Output:
        Dict[str, Any] | None: Mapping like ``{"level1": "...", "level2": "..."}``
            when available, otherwise None.
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


_SCENE_LEVEL_PLACEHOLDER_TOKENS: frozenset[str] = frozenset(
    f"scene_level{i}" for i in range(1, 6)
)


def _scene_type_value_is_usable(resolved: Any) -> bool:
    """
    Return True if scene_type has at least one real label (not template tokens).

    Supports dict (level1..level5), str, or list[str] from local_dataset_info.yaml.
    """
    if resolved in (None, ""):
        return False
    if isinstance(resolved, dict):
        for i in range(1, 6):
            k = f"level{i}"
            v = resolved.get(k)
            if not isinstance(v, str):
                continue
            s = v.strip()
            if not s:
                continue
            if s in _SCENE_LEVEL_PLACEHOLDER_TOKENS:
                continue
            if s == f"scene_level{i}":
                continue
            return True
        return False
    if isinstance(resolved, list):
        for item in resolved:
            if not isinstance(item, str):
                continue
            s = item.strip()
            if s and s not in _SCENE_LEVEL_PLACEHOLDER_TOKENS:
                return True
        return False
    if isinstance(resolved, str):
        s = resolved.strip()
        return bool(s) and s not in _SCENE_LEVEL_PLACEHOLDER_TOKENS
    return False


def _required_is_strict(required_status: Any) -> bool:
    return required_status == "strict"


def _value_matches_schema_default(resolved: Any, default_value: Any) -> bool:
    """True if resolved value is exactly the schema default (template placeholder)."""
    if default_value is None:
        return False
    return resolved == default_value


def _raise_scene_type_strict(*, reason: str, local_dataset_info_path: Path) -> None:
    """Raise ValueError with a clear message; traceback is preserved by default."""
    msg = (
        "[scene_type] required: strict — no usable scene type. "
        f"{reason} "
        f"Fix local_dataset_info.yaml at {local_dataset_info_path}: set scene_level1..scene_level5 "
        "and/or legacy scene_level with real labels from the scene library. "
        "Placeholder defaults are refused."
    )
    raise ValueError(msg)


def _raise_strict_unresolved(
    field_name: str,
    field_spec: Dict[str, Any],
    *,
    dataset_path: Path,
    local_dataset_info_path: Path,
) -> None:
    src = field_spec.get("source")
    raise ValueError(
        f"[{field_name}] required: strict — could not resolve value from source={src!r}. "
        f"dataset_path={dataset_path} local_dataset_info={local_dataset_info_path}. "
        "Schema default is refused."
    )


def _raise_strict_resolved_equals_default(
    field_name: str,
    field_spec: Dict[str, Any],
    *,
    dataset_path: Path,
    local_dataset_info_path: Path,
) -> None:
    src = field_spec.get("source")
    raise ValueError(
        f"[{field_name}] required: strict — resolved value equals schema default placeholder. "
        f"source={src!r} dataset_path={dataset_path} local_dataset_info={local_dataset_info_path}."
    )


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
                source_cache[cache_key] = load_json_file(source_path, logger)
            elif source_path.suffix == ".jsonl":
                source_cache[cache_key] = load_jsonl_file(source_path)
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
            sensors = extract_sensor_list_from_features(features)
            return sensors if sensors else None
        if field_name == "came_info":
            cam_info = extract_camera_info_from_features(features)
            return cam_info if cam_info else None
        if field_name == "depth_enabled":
            return extract_depth_enabled_from_features(features)
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
            inferred_units = extract_dimension_units_from_features(features)
            value = inferred_units.get(field_name)
            return value if value not in (None, "") else None

    if source_path.name.endswith(".jsonl") and isinstance(source_data, list):
        if field_name == "sub_tasks":
            sub_tasks = extract_sub_tasks_from_annotations(source_data)
            return sub_tasks if sub_tasks else None
        if field_name == "task_result":
            return extract_task_result_from_annotations(source_data)

    if source_path_str.endswith("meta/episodes_stats.jsonl") and isinstance(source_data, list):
        if field_name == "statistics":
            meta_cache_key = (dataset_path / "meta/info.json").as_posix()
            meta_info = source_cache.get(meta_cache_key)
            if not isinstance(meta_info, dict):
                meta_info = {}
            sub_tasks = source_cache.get("__resolved_sub_tasks__", default_value)
            return build_statistics_from_sources(
                meta_info=meta_info,
                episode_stats_rows=source_data,
                dataset_path=dataset_path,
                sub_tasks=sub_tasks,
            )

    return None


# ============================================================================
# Public entry point
# ============================================================================


def resolve_context_from_schema(
    schema_yaml_path: Path,
    dataset_path: Path,
    local_dataset_info_path: Path,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """
    Resolve all schema fields and return a flat context dictionary.

    This is the core context-building function used by InfoCollector to resolve
    all metadata and prepare it for serialisation to info.yaml.

    Input:
        schema_yaml_path (Path): Path to schema info.yaml (assets/info.yaml).
        dataset_path (Path): Dataset root directory.
        local_dataset_info_path (Path): Path to local_dataset_info.yaml.
        logger (logging.Logger): Logger instance.

    Output:
        Dict[str, Any]: Flat context dict with all resolved fields.
            Includes fields from schema with resolved values, auto-computed
            fields (dataset_size, data_structure, …), and template-compat
            transformations.

    Raises:
        ValueError: If any field has ``required: strict`` and resolution fails or matches a placeholder.

    Logic:
        1. Load schema and local_dataset_info.
        2. Iterate schema fields; for each field, resolve its value by source.
        3. Apply auto-field computation and template compatibility.
        4. Return flat dict suitable for YAML serialisation.
    """
    schema = load_schema_yaml(schema_yaml_path, logger)
    local_dataset_info = load_yaml_file(local_dataset_info_path, logger)

    context_data: Dict[str, Any] = {}
    source_cache: Dict[str, Any] = {}
    source_type_counter: Dict[str, int] = {}

    for field_name, field_spec in schema.items():
        if str(field_name).startswith("_"):
            continue
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
            elif required_status in (True, "strict"):
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
            if required_status in (True, "strict", "optional"):
                if _required_is_strict(required_status):
                    if field_name == "scene_type" and not _scene_type_value_is_usable(resolved):
                        _raise_scene_type_strict(
                            reason="Resolved scene_type is empty or only template placeholders.",
                            local_dataset_info_path=local_dataset_info_path,
                        )
                    elif field_name != "scene_type" and _value_matches_schema_default(
                        resolved, default_value
                    ):
                        _raise_strict_resolved_equals_default(
                            field_name,
                            field_spec,
                            dataset_path=dataset_path,
                            local_dataset_info_path=local_dataset_info_path,
                        )
                context_data[field_name] = resolved
        elif required_status in (True, "strict"):
            if _required_is_strict(required_status):
                if field_name == "scene_type":
                    _raise_scene_type_strict(
                        reason=(
                            "Could not resolve scene hierarchy from local_dataset_info.yaml "
                            "(no scene_level1..scene_level5 and no usable legacy scene_level)."
                        ),
                        local_dataset_info_path=local_dataset_info_path,
                    )
                _raise_strict_unresolved(
                    field_name,
                    field_spec,
                    dataset_path=dataset_path,
                    local_dataset_info_path=local_dataset_info_path,
                )
            context_data[field_name] = default_value
        elif required_status == "optional":
            pass
        else:
            pass

    apply_auto_fields(context_data=context_data, dataset_path=dataset_path, logger=logger)
    apply_template_compatibility(context_data=context_data)

    logger.info(
        "[RESOLVE_CONTEXT] Context resolved. Source types: "
        + ", ".join(
            [f"{k}={v}" for k, v in sorted(source_type_counter.items())]
        )
    )
    return context_data
