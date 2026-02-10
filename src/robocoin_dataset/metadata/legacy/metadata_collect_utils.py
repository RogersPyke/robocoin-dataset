"""
`prepare_metadata` 目录下的“元数据采集工具集”。

这个文件刻意只放**无状态**的纯函数（或尽量接近纯函数），供上层的
`metadata_collect.create_unified_metadata()` 组装使用。

为什么需要这个文件：
- `metadata_collect.py` 需要同时读取 DB / YAML / meta/ / annotations/ / 文件系统结构；
  如果把所有辅助逻辑都塞在一个文件里，主流程会很难读、也很难复用。
- README 生成器（例如 `hub_upload/gen_readme/single_dataset_readme_generator.py`）
  也需要复用少量“目录结构树”等展示型信息。

本文件按“数据来源/职责”做了分区（仅重排与注释增强，不改变行为）：
- **目录名 -> 设备名映射**：`match_device_name_from_folder`
- **annotations/ 提取**：`collect_subtask_annotations`
- **原始 YAML 读取与字段抽取**：`collect_from_yaml`
- **meta/ 目录读取**：`collect_meta_info`
- **features 摘要（相机/观测/动作空间）**：`build_*_from_features`
- **文件系统统计/展示**：`calculate_dataset_size`、`format_file_size`、`generate_size_label`、`generate_folder_structure`
"""

import json
import logging
from pathlib import Path
from typing import Any

import yaml

from robocoin_dataset.prepare_metadata.unified_metadata_def import UnifiedMetadata

_logger = logging.getLogger(__name__)

# =============================================================================
# 目录名 -> 设备名映射（names.yml）
# =============================================================================

def match_device_name_from_folder(dataset_folder_name: str) -> str | None:
    """
    根据数据集文件夹名，尝试从 `prepare_metadata/names.yml` 中匹配一个“设备/机器人名”。

    使用位置：
    - `metadata_collect.create_unified_metadata()` 会用它作为 robot_type 的候选值之一

    规则：
    - `names.yml` 是一个字符串列表；只要其中某个字符串是 folder_name 的子串就算命中
    - 命中返回该字符串，否则返回 None

    注意：
    - 这是一个“启发式匹配”，不保证 100% 正确；上层会有优先级回退（meta/info.json、DB 等）。
    """
    # names.yml is colocated with this module under prepare_metadata/
    names_file = Path(__file__).parent / "names.yml"
    if not names_file.exists():
        _logger.warning(
            "Names file does not exist: %s. Cannot match device name.", names_file
        )
        return None

    try:
        with names_file.open(encoding="utf-8") as f:
            device_names = yaml.safe_load(f)
    except Exception as exc:
        _logger.error("Failed to load names.yml: %s. Cannot match device name.", exc)
        return None

    if not isinstance(device_names, list):
        _logger.error(
            "names.yml should contain a list, but got %s. Cannot match device name.",
            type(device_names),
        )
        return None

    for device_name in device_names:
        if device_name in dataset_folder_name:
            _logger.info(
                "Matched device name '%s' in dataset name '%s'",
                device_name,
                dataset_folder_name,
            )
            return device_name

    _logger.debug(
        "No device name from names.yml matched in dataset name '%s'",
        dataset_folder_name,
    )
    return None


# =============================================================================
# annotations/ 提取（子任务等）
# =============================================================================

def collect_subtask_annotations(annotations_dir: Path) -> list[str]:
    """
    从 `annotations/subtask_annotations.jsonl` 中抽取子任务列表（去重 + 稳定排序）。

    使用位置：
    - `metadata_collect.create_unified_metadata()`：写入 `UnifiedMetadata.sub_tasks`

    行为细节（保持现状）：
    - 文件不存在/不可读/解析失败：直接返回空列表（容错）
    - 去重：不区分大小写（使用 lower 后的 key）
    - 排序：优先使用 `subtask_index` 或 `index`（如果能解析为 int），否则放到最后；
      在 index 相同/缺失时使用行号保持稳定
    """
    if not annotations_dir.exists():
        return []

    subtask_file = annotations_dir / "subtask_annotations.jsonl"
    if not subtask_file.exists():
        return []

    seen_subtasks: set[str] = set()
    entries: list[tuple[int | None, int, str]] = []
    try:
        with subtask_file.open(encoding="utf-8") as f:
            for line_number, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if "subtask" not in data:
                    continue
                subtask = data["subtask"]
                key = str(subtask).lower()
                if key in seen_subtasks:
                    continue
                seen_subtasks.add(key)
                index_value = _extract_subtask_index(data)
                entries.append((index_value, line_number, subtask))
    except Exception:
        return []

    entries.sort(key=lambda item: (item[0] if item[0] is not None else float("inf"), item[1]))
    return [entry[2] for entry in entries]


def _extract_subtask_index(data: dict[str, Any]) -> int | None:
    """提取排序用的索引：优先 subtask_index，其次 index；无法解析则返回 None。"""
    for key in ("subtask_index", "index"):
        if key not in data:
            continue
        value = data[key]
        if value is None:
            continue
        if isinstance(value, str):
            value = value.strip()
        try:
            return int(value)
        except (ValueError, TypeError):
            continue
    return None


# =============================================================================
# 原始 YAML 读取与字段抽取（dataset_info.yml）
# =============================================================================

def collect_from_yaml(
    yaml_file_path: str | Path, dataset_name: str, dataset_uuid: str
) -> dict[str, Any]:
    """
    从“原始 YAML”（通常是 `dataset_info.yml`）中抽取可用于展示/筛选的字段。

    使用位置：
    - `metadata_collect._load_yaml_payload()`：在 DB 字段缺失时做补全（scene_type、atomic_actions、objects 等）

    设计说明：
    - YAML 内容可能存在错误或不完整，因此上层会优先使用 DB 中结构化字段。
    - 这里保留 raw_yaml，用于调试或下游模板引用。
    """
    raw_yaml = _load_raw_yaml(yaml_file_path)

    scene_type = []
    if raw_yaml.get("scene_type"):
        scene_type = raw_yaml["scene_type"] if isinstance(raw_yaml["scene_type"], list) else []

    atomic_actions = []
    if raw_yaml.get("atomic_actions"):
        atomic_actions = (
            raw_yaml["atomic_actions"]
            if isinstance(raw_yaml["atomic_actions"], list)
            else []
        )

    objects = []
    raw_objects = raw_yaml.get("objects")
    if isinstance(raw_objects, list):
        objects.extend(
            {
                "object_name": obj.get("object_name"),
                "level1": obj.get("level1"),
                "level2": obj.get("level2"),
                "level3": obj.get("level3"),
                "level4": obj.get("level4"),
                "level5": obj.get("level5"),
            }
            for obj in raw_objects
            if isinstance(obj, dict) and "object_name" in obj
        )

    if not scene_type:
        _logger.warning(
            "Dataset %s (UUID: %s) has empty scene_type in YAML. YAML file: %s",
            dataset_name,
            dataset_uuid,
            yaml_file_path,
        )
    if not atomic_actions:
        _logger.warning(
            "Dataset %s (UUID: %s) has empty atomic_actions in YAML. YAML file: %s",
            dataset_name,
            dataset_uuid,
            yaml_file_path,
        )
    if not objects:
        _logger.warning(
            "Dataset %s (UUID: %s) has empty objects in YAML. YAML file: %s",
            dataset_name,
            dataset_uuid,
            yaml_file_path,
        )

    return {
        "raw_yaml": raw_yaml,
        "scene_type": scene_type,
        "atomic_actions": atomic_actions,
        "objects": objects,
    }


def _load_raw_yaml(yaml_file_path: str | Path | None) -> dict[str, Any]:
    """
    安全读取 YAML，任何异常都返回空 dict。

    为什么“吞掉异常”：
    - 这一步是“可选补全”，主流程应该尽量继续（尤其在批处理/分布式上传场景）。
    """
    if not yaml_file_path:
        return {}

    path = Path(yaml_file_path).expanduser()
    if not path.exists():
        return {}

    try:
        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


# =============================================================================
# meta/ 目录读取（info.json / tasks.jsonl）
# =============================================================================

def collect_meta_info(meta_dir: Path) -> tuple[dict[str, Any], str]:
    """
    读取 `meta/` 目录下的关键信息：
    - `meta/info.json`（统计信息、特征定义、robot_type 等）
    - `meta/tasks.jsonl`（自然语言任务描述列表）

    使用位置：
    - `metadata_collect.create_unified_metadata()`：合并统计信息、features、tasks 等字段
    """
    info_file = meta_dir / "info.json"
    tasks_file = meta_dir / "tasks.jsonl"
    return _load_meta_info(info_file), _load_tasks(tasks_file)


def _load_meta_info(meta_info_file: Path) -> dict[str, Any]:
    """
    从 `meta/info.json` 中**挑选式**抽取字段（而不是原样全部拷贝）。

    目的：
    - `UnifiedMetadata` 希望持有稳定、可控的展示字段；meta/info.json 可能会不断加新字段
    - 抽取能减少 YAML 输出噪音，也便于前端/README 的字段契约保持稳定
    """
    if not meta_info_file.exists():
        return {}

    try:
        with meta_info_file.open(encoding="utf-8") as f:
            meta_info: dict[str, Any] = json.load(f)
    except Exception:
        return {}

    extracted: dict[str, Any] = {}

    for key in ("robot_type", "codebase_version"):
        if key in meta_info:
            extracted[key] = meta_info[key]

    statistics: dict[str, Any] = {}
    for key in [
        "total_episodes",
        "total_frames",
        "total_tasks",
        "total_videos",
        "total_chunks",
        "chunks_size",
        "fps",
        "total_duration",
        "video_resolution",
        "state_dim",
        "action_dim",
        "camera_views",
    ]:
        if key in meta_info:
            statistics[key] = meta_info[key]
    if statistics:
        extracted["statistics"] = statistics

    for key in ("splits", "data_path", "video_path"):
        if key in meta_info:
            extracted[key] = meta_info[key]

    if "features" in meta_info:
        features = meta_info["features"]
        extracted["features"] = features
        depth_enabled = False
        for key, value in features.items():
            if key.startswith("observation.images.") and isinstance(value, dict):
                info = value.get("info", {})
                if info.get("video.is_depth_map", False):
                    depth_enabled = True
                    break
        extracted["depth_enabled"] = depth_enabled

    return extracted


def _load_tasks(tasks_file: Path) -> str:
    """
    读取 `meta/tasks.jsonl`，把每行的 `task` 字段用换行拼接成一个字符串。

    使用位置：
    - `metadata_collect.create_unified_metadata()`：写入 `UnifiedMetadata.tasks`
    """
    if not tasks_file.exists():
        return ""

    tasks: list[str] = []
    try:
        with tasks_file.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    task = data.get("task", "")
                    if task:
                        tasks.append(task)
                except json.JSONDecodeError:
                    continue
    except Exception:
        return ""

    return "\n".join(tasks)


# =============================================================================
# features 摘要（相机/观测空间/动作空间）
# =============================================================================

def collect_directory_structure(ds_path: Path) -> dict[str, Any]:
    """
    生成数据集目录结构摘要。

    兼容性说明：
    - 这个函数目前只是 `generate_folder_structure()` 的薄封装（历史遗留）。
    - 对外保留以避免已有调用方断裂。
    """
    return generate_folder_structure(ds_path, max_files_per_dir=5)


def build_cameras_from_features(features: dict[str, Any]) -> list[dict[str, Any]]:
    """
    从 `features` 中提取“相机摘要列表”。

    使用位置：
    - `metadata_collect.create_unified_metadata()`：写入 `UnifiedMetadata.cameras`
    """
    cameras: list[dict[str, Any]] = []
    for key, value in features.items():
        if not (isinstance(key, str) and key.startswith("observation.images.")):
            continue
        if not isinstance(value, dict):
            continue
        name = key.split(".")[-1]
        info = value.get("info", {}) if isinstance(value.get("info", {}), dict) else {}
        cameras.append(
            {
                "key": key,
                "name": name,
                "dtype": value.get("dtype"),
                "shape": value.get("shape"),
                "resolution": [info.get("video.height"), info.get("video.width")],
                "fps": info.get("video.fps"),
                "is_depth": bool(info.get("video.is_depth_map", False)),
            }
        )
    return cameras


def build_observation_space_from_features(features: dict[str, Any]) -> dict[str, Any]:
    """
    从 `features` 中提炼 `observation_space`。

    说明：
    - 返回值的基础结构来自 `UnifiedMetadata().observation_space` 的默认值
      （这样可以保证没有 images/state 时仍输出稳定结构）。
    """
    images: list[dict[str, Any]] = []
    for key, value in features.items():
        if not (isinstance(key, str) and key.startswith("observation.images.")):
            continue
        if not isinstance(value, dict):
            continue
        images.append(
            {
                "key": key,
                "dtype": value.get("dtype"),
                "shape": value.get("shape"),
                "names": value.get("names"),
            }
        )

    state_info: dict[str, Any] | None = None
    state_feat = features.get("observation.state")
    if isinstance(state_feat, dict):
        state_info = {
            "dtype": state_feat.get("dtype"),
            "shape": state_feat.get("shape"),
            "names": state_feat.get("names"),
        }

    obs_space = UnifiedMetadata().observation_space
    if images:
        obs_space["images"] = images
    if state_info is not None:
        obs_space["state"] = state_info
    return obs_space


def build_action_space_from_features(features: dict[str, Any]) -> dict[str, Any] | str:
    """
    从 `features` 中提炼 `action_space`。

    返回值：
    - 如果 features["action"] 存在且为 dict：返回包含 dtype/shape/names 的 dict
    - 否则：返回 `UnifiedMetadata().action_space` 的默认值（通常为 "auto_generated"）
    """
    action_feat = features.get("action")
    if isinstance(action_feat, dict):
        return {
            "dtype": action_feat.get("dtype"),
            "shape": action_feat.get("shape"),
            "names": action_feat.get("names"),
        }
    return UnifiedMetadata().action_space


# =============================================================================
# 文件系统统计/展示（体积、帧数标签、目录树）
# =============================================================================

def calculate_dataset_size(ds_path: Path) -> int:
    """
    递归统计目录总大小（字节）。

    使用位置：
    - `metadata_collect._build_auto_generated_fields()`：写入 `UnifiedMetadata.dataset_size`
    """
    if not ds_path.exists():
        return 0

    total_size = 0
    try:
        for file_path in ds_path.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
    except Exception:
        return 0
    return total_size


def format_file_size(size_bytes: int) -> str:
    """把字节数格式化为人类可读字符串（B/KB/MB/GB/TB）。"""
    if size_bytes == 0:
        return "0B"
    size_names = ["B", "KB", "MB", "GB", "TB"]
    size_index = 0
    size = float(size_bytes)
    while size >= 1024 and size_index < len(size_names) - 1:
        size /= 1024
        size_index += 1
    if size_index == 0:
        return f"{int(size)}{size_names[size_index]}"
    return f"{size:.1f}{size_names[size_index]}"


def generate_size_label(size: int) -> str:
    """
    把帧数映射为离散区间标签（用于展示与筛选）。

    使用位置：
    - `metadata_collect._build_auto_generated_fields()`：根据 total_frames 生成 frame_range
    """
    if size < 1000:
        return "<1K"
    if size < 10000:
        return "1K-10K"
    if size < 100000:
        return "10K-100K"
    if size < 1000000:
        return "100K-1M"
    if size < 10000000:
        return "1M-10M"
    if size < 100000000:
        return "10M-100M"
    if size < 1000000000:
        return "100M-1B"
    if size < 10000000000:
        return "1B-10B"
    if size < 100000000000:
        return "10B-1T"
    return ">1T"


def generate_folder_structure(root_path: Path, max_files_per_dir: int = 5) -> str:
    """
    Generate a folder structure tree showing only leaf directories with limited files.

    This function traverses the directory tree and creates a visual representation where:
    - Only the deepest leaf directories are expanded
    - Each leaf directory shows at most max_files_per_dir files
    - Remaining files are represented as "(...)"

    Args:
        root_path: Root directory path to generate structure from.
        max_files_per_dir: Maximum number of files to show per leaf directory. Defaults to 5.

    Returns:
        str: Formatted folder structure tree as a string.
    """

    def is_leaf_directory(path: Path) -> bool:
        """Check if a directory is a leaf (contains no subdirectories)."""
        try:
            return not any(item.is_dir() for item in path.iterdir())
        except (PermissionError, OSError):
            return True

    def get_sorted_items(path: Path) -> tuple[list[Path], list[Path]]:
        """Get sorted directories and files from a path."""
        try:
            items = list(path.iterdir())
            dirs = sorted([item for item in items if item.is_dir()], key=lambda x: x.name)
            files = sorted([item for item in items if item.is_file()], key=lambda x: x.name)
            return dirs, files
        except (PermissionError, OSError):
            return [], []

    def build_tree(path: Path, prefix: str = "", is_last: bool = True) -> list[str]:
        """Recursively build the tree structure."""
        lines: list[str] = []

        if not path.exists():
            return lines

        connector = "└── " if is_last else "├── "
        if path == root_path:
            lines.append(f"{path.name}/")
        else:
            lines.append(f"{prefix}{connector}{path.name}/")

        dirs, files = get_sorted_items(path)

        new_prefix = ""
        if path != root_path:
            new_prefix = prefix + ("    " if is_last else "│   ")

        if is_leaf_directory(path) and files:
            files_to_show = files[:max_files_per_dir]
            has_more = len(files) > max_files_per_dir

            for i, file in enumerate(files_to_show):
                is_last_file = (i == len(files_to_show) - 1) and not has_more
                file_connector = "└── " if is_last_file else "├── "
                lines.append(f"{new_prefix}{file_connector}{file.name}")

            if has_more:
                lines.append(f"{new_prefix}└── (...)")

        for i, subdir in enumerate(dirs):
            is_last_dir = i == len(dirs) - 1
            lines.extend(build_tree(subdir, new_prefix, is_last_dir))

        return lines

    try:
        tree_lines = build_tree(root_path)
        return "\n".join(tree_lines)
    except Exception as exc:  # pragma: no cover - defensive
        return f"Error generating folder structure: {exc}"
