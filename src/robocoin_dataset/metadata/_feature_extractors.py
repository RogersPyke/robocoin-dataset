"""
Feature extraction helpers for the Collect stage.

Parse the ``features`` dict found in ``meta/info.json`` and produce
sensor lists, camera specifications, depth flags, and dimension units.
No I/O is performed here; all functions accept already-loaded dicts.
"""

from typing import Any, Dict, List


# ============================================================================
# Sensor / camera helpers
# ============================================================================


def extract_sensor_list_from_features(features: Any) -> List[str]:
    """
    Extract camera / sensor names from the ``features`` mapping.

    Supports both flat ``observation.images.<name>`` keys and the nested
    ``observation.images`` sub-dict layout.

    Input:
        features (Any): The ``features`` value from meta/info.json.

    Output:
        List[str]: Ordered, deduplicated list of sensor names.
    """
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


def extract_camera_info_from_features(features: Any) -> Dict[str, str]:
    """
    Build a mapping of sensor name -> human-readable camera spec string.

    Input:
        features (Any): The ``features`` value from meta/info.json.

    Output:
        Dict[str, str]: ``{sensor_name: spec_string}`` for every image feature.
    """
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


def extract_depth_enabled_from_features(features: Any) -> bool:
    """
    Detect whether depth-image data is present in the features mapping.

    Input:
        features (Any): The ``features`` value from meta/info.json.

    Output:
        bool: True if any depth feature is detected.
    """
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


# ============================================================================
# Dimension unit helpers
# ============================================================================


def _infer_unit_from_names(
    names: List[str],
    keywords: List[str],
    suffix_to_unit: Dict[str, str],
) -> str | None:
    """
    Infer a physical unit from feature names by keyword and suffix matching.

    Input:
        names (List[str]): Feature name list (from observation.state.names).
        keywords (List[str]): All keywords must appear in the lowercased name.
        suffix_to_unit (Dict[str, str]): ``{suffix: unit_label}`` mapping.

    Output:
        str | None: Resolved unit label, or None if no match.
    """
    for raw_name in names:
        name = str(raw_name).lower()
        if not all(keyword in name for keyword in keywords):
            continue
        for suffix, unit in suffix_to_unit.items():
            if name.endswith(suffix):
                return unit
    return None


def extract_dimension_units_from_features(features: Any) -> Dict[str, str]:
    """
    Extract dimension unit fields (joint/eef/base rotation and translation) from features.

    Input:
        features (Any): The ``features`` value from meta/info.json.

    Output:
        Dict[str, str]: Subset of dimension-unit fields that could be inferred.
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
