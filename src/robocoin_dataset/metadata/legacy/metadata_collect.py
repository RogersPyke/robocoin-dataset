"""
本脚本整合网页静态资源与 Hub README 生成所需的元数据收集逻辑。

所有外部调用统一通过 ``create_unified_metadata``，该函数会：
1. 读取 SQLite 数据库，解析 DatasetDB 记录（场景、物体、末端执行器等结构化信息）
2. 读取原始 YAML（dataset_info.yml）补充 YAML 原生字段
3. 扫描 hardlink 数据集目录下的 meta/ 与 annotations/ 内容，构建统计信息、任务描述等
4. 自动生成目录结构、文件大小、帧数范围等派生字段

最终返回 ``UnifiedMetadata``，供网页（YAML）和 README 模板复用，确保两端信息完全一致。

使用位置（面向调用方）：
- `robocoin_dataset.prepare_metadata.metadata_service.MetadataSyncService`：页面同步与 README 生成统一入口
- `robocoin_dataset.hub_upload.lerobot.hub_upload_server.HubUploadServer`：服务端预先构建元数据，下发给分布式客户端

设计原则：
- **不做“业务含义”上的修正**：这里只负责把不同来源的字段合并到一个对象中，字段值的优先级保持稳定。
- **不做重型计算的重复**：派生字段统一在本模块内计算，避免在 README 渲染器等下游再次做同样逻辑。
"""

import logging
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from robocoin_dataset.database.database import DatasetDatabase
from robocoin_dataset.database.models import DatasetDB, DatasetHardLinkDB, ObjectDB
from robocoin_dataset.prepare_metadata.metadata_collect_utils import (
    build_action_space_from_features,
    build_cameras_from_features,
    build_observation_space_from_features,
    calculate_dataset_size,
    collect_from_yaml,
    collect_meta_info,
    collect_subtask_annotations,
    format_file_size,
    generate_folder_structure,
    generate_size_label,
    match_device_name_from_folder,
)
from robocoin_dataset.prepare_metadata.unified_metadata_def import UnifiedMetadata

LOGGER = logging.getLogger(__name__)

def _escape_yaml_single_quoted_string(value: str) -> str:
    """
    把普通字符串转换为适合写入 YAML 单引号字符串的形式。

    背景：
    - HuggingFace Hub 的 gated dataset 配置最终会被写入 YAML header。
    - YAML 单引号内如果出现单引号，需要用两个单引号表示（YAML 语法规则）。
    - 同时我们尽量避免换行/制表符等控制字符把 YAML 结构“打断”。

    注意：
    - 这是一个“字符串清理”工具，不改变语义，只保证序列化安全。
    """
    # 对于 YAML 单引号字符串，我们需要转义单引号为双单引号
    # 同时确保内容中没有可能破坏 YAML 结构的控制字符
    return (
        value.replace("'", "''")
        .replace("\n", " ")
        .replace("\r", " ")
        .replace("\t", " ")
    )


def get_default_gated_access_config() -> dict[str, Any]:
    """
    获取默认的数据集访问控制配置。

    这个函数封装了所有数据集的标准访问控制配置，用于生成 HuggingFace Hub 的 README 头部。
    统一管理所有数据集的访问控制规则，便于后续修改和维护。

    Returns:
        dict[str, Any]: 包含以下字段的字典：
            - extra_gated_prompt (str): 用户访问数据集时显示的提示信息
            - extra_gated_fields (dict): 用户需要填写的表单字段配置
    """
    return {
        "extra_gated_prompt": _escape_yaml_single_quoted_string(
            "By accessing this dataset, you agree to cite the associated paper in your research/publications—see the \"Citation\" section for details. "
            "You agree to not use the dataset to conduct experiments that cause harm to human subjects."
        ),
        "extra_gated_fields": {
            "Company/Organization": {
                "type": _escape_yaml_single_quoted_string("text"),
                "description": _escape_yaml_single_quoted_string(
                    'e.g., "ETH Zurich", "Boston Dynamics", "Independent Researcher"'
                ),
            },
            "Country": {
                "type": _escape_yaml_single_quoted_string("country"),
                "description": _escape_yaml_single_quoted_string(
                    'e.g., "Germany", "China", "United States"'
                ),
            }
        },
    }


def create_unified_metadata(
    hardlink_path: str | Path,
    db_file_path: str | Path,
    dataset_uuid: str | None = None,
) -> UnifiedMetadata:
    """
    构建单个数据集的统一元数据，确保 README 与网页静态资源使用同一份内容。
    """
    dataset_path = Path(hardlink_path).expanduser().absolute()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    db_path = Path(db_file_path).expanduser().absolute()
    if not db_path.exists():
        raise FileNotFoundError(f"Database file does not exist: {db_path}")

    db_payload = _load_dataset_payload(
        db_path=db_path,
        dataset_path=dataset_path,
        dataset_uuid=dataset_uuid,
    )
    yaml_payload = _load_yaml_payload(db_payload)

    meta_info, tasks_text = collect_meta_info(dataset_path / "meta")
    sub_tasks = collect_subtask_annotations(dataset_path / "annotations")

    features = meta_info.get("features", {}) if isinstance(meta_info, dict) else {}
    statistics = _merge_statistics(
        meta_info.get("statistics") if isinstance(meta_info, dict) else None,
        db_payload,
    )
    auto_fields = _build_auto_generated_fields(
        dataset_path=dataset_path,
        meta_info=meta_info,
        statistics=statistics,
        db_payload=db_payload,
    )

    metadata = UnifiedMetadata()
    metadata.update(**get_default_gated_access_config())
    metadata.update(
        dataset_name=db_payload.get("dataset_name", dataset_path.name),
        dataset_uuid=db_payload.get("dataset_uuid"),
        scene_type=db_payload.get("scene_types") or yaml_payload.get("scene_type") or [],
        atomic_actions=db_payload.get("atomic_actions") or yaml_payload.get("atomic_actions") or [],
        objects=db_payload.get("objects") or yaml_payload.get("objects") or [],
        end_effector_type=_normalize_to_list(
            db_payload.get("end_effector_type")
            or (yaml_payload.get("raw_yaml") or {}).get("end_effector_type")
        ),
        operation_platform_height=db_payload.get("operation_platform_height")
        or (yaml_payload.get("raw_yaml") or {}).get("operation_platform_height"),
        # ========== 自动派生/自动生成字段（单独封装，便于复用与审查） ==========
        # 这些字段不直接来自 DB/YAML/meta 文件，而是基于它们推导出来的“展示/索引字段”。
        **auto_fields,
        # ========== meta/info.json 与 meta/tasks.jsonl 来源字段 ==========
        codebase_version=meta_info.get("codebase_version", "") if isinstance(meta_info, dict) else "",
        statistics=statistics or metadata.statistics,
        splits=meta_info.get("splits") if isinstance(meta_info, dict) and meta_info.get("splits") else metadata.splits,
        data_path=meta_info.get("data_path") if isinstance(meta_info, dict) and meta_info.get("data_path") else metadata.data_path,
        video_path=meta_info.get("video_path") if isinstance(meta_info, dict) and meta_info.get("video_path") else metadata.video_path,
        features=features or metadata.features,
        depth_enabled=bool(meta_info.get("depth_enabled")) if isinstance(meta_info, dict) else False,
        tasks=tasks_text,
        sub_tasks=sub_tasks,
    )

    if features:
        # ========== features ->（相机/观测空间/动作空间）摘要 ==========
        # 注意：这一步不会修改 features 本身，只是把其中的结构“提炼”为更适合展示的摘要字段。
        metadata.cameras = build_cameras_from_features(features)
        metadata.observation_space = build_observation_space_from_features(features)
        metadata.action_space = build_action_space_from_features(features)

    metadata.raw = {
        "db_record": db_payload,
        "yaml": yaml_payload.get("raw_yaml", {}),
        "meta": meta_info,
    }

    return metadata


def _normalize_to_list(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        separators = [",", "/", "|"]
        for sep in separators:
            if sep in value:
                return [segment.strip() for segment in value.split(sep) if segment.strip()]
        return [value.strip()] if value.strip() else []
    return [str(value)]


def _derive_dataset_slug(folder_name: str) -> str:
    return (
        folder_name.removesuffix("_qced_hardlink")
        .removesuffix("_hardlink")
        .removesuffix("/")
    )

def _build_auto_generated_fields(
    *,
    dataset_path: Path,
    meta_info: dict[str, Any],
    statistics: dict[str, Any],
    db_payload: dict[str, Any],
) -> dict[str, Any]:
    """
    计算并返回“自动生成/派生字段”集合。

    这些字段的特点：
    - **只依赖输入**（hardlink 路径、meta/info.json 抽取结果、DB/YAML 合并结果），不产生副作用
    - **用于展示与索引**：例如页面资源路径、帧数范围标签、数据集体积、目录结构树等

    被谁用：
    - 本模块 `create_unified_metadata` 会把这里返回的 dict 通过 `metadata.update(**auto_fields)` 写回 `UnifiedMetadata`

    为什么要单独封装：
    - 方便审查“派生字段”的来源与逻辑，不和 DB/YAML 合并逻辑混在一起
    - 未来如果需要在别的入口做同样计算（例如只给页面生成 YAML），可以复用同一套实现
    """
    dataset_slug = _derive_dataset_slug(dataset_path.name)

    frame_total = statistics.get("total_frames") if statistics else None
    frame_range = generate_size_label(int(frame_total or 0))

    dataset_size = format_file_size(calculate_dataset_size(dataset_path))
    dataset_structure = generate_folder_structure(dataset_path, max_files_per_dir=5)

    # robot_type 的优先级保持不变：
    # 1) meta/info.json 的 robot_type（如果存在）
    # 2) 根据文件夹名匹配 names.yml
    # 3) DB 里的 device_model（兼容旧字段/历史数据）
    robot_type = (
        meta_info.get("robot_type") if isinstance(meta_info, dict) else None
    ) or match_device_name_from_folder(dataset_path.name) or db_payload.get("device_model") or ""

    return {
        "path": dataset_slug,
        "video_url": f"./assets/videos/{dataset_slug}.mp4",
        "thumbnail_url": f"./assets/thumbnails/{dataset_slug}.jpg",
        "frame_range": frame_range,
        "dataset_size": dataset_size,
        "robot_type": robot_type,
        "structure": dataset_structure,
    }


def _load_dataset_payload(
    db_path: Path,
    dataset_path: Path,
    dataset_uuid: str | None,
) -> dict[str, Any]:
    db = DatasetDatabase(db_path)
    with db.with_session() as session:
        resolved_uuid = dataset_uuid or _lookup_uuid_by_hardlink(session, dataset_path)
        record = _query_dataset_record(session, resolved_uuid, dataset_path.name)
        if record is None:
            raise ValueError(
                f"Failed to locate dataset metadata in DB for dataset '{dataset_path.name}'"
            )
        return _serialize_dataset_record(record)


def _lookup_uuid_by_hardlink(session: Session, dataset_path: Path) -> str | None:
    entry = (
        session.query(DatasetHardLinkDB)
        .filter(DatasetHardLinkDB.hard_link_path == str(dataset_path))
        .first()
    )
    if entry:
        return entry.dataset_uuid
    LOGGER.warning(
        "Hardlink mapping missing for %s, falling back to dataset name lookup",
        dataset_path,
    )
    return None


def _query_dataset_record(
    session: Session,
    dataset_uuid: str | None,
    dataset_name: str,
) -> DatasetDB | None:
    query = session.query(DatasetDB)
    if dataset_uuid:
        dataset = query.filter(DatasetDB.dataset_uuid == dataset_uuid).first()
        if dataset:
            return dataset

    dataset = query.filter(DatasetDB.dataset_name == dataset_name).first()
    if dataset:
        return dataset

    return (
        query.filter(DatasetDB.convert_path.like(f"%{dataset_name}%"))
        .order_by(DatasetDB.id.desc())
        .first()
    )


def _serialize_dataset_record(dataset: DatasetDB) -> dict[str, Any]:
    def _object_to_dict(obj: ObjectDB) -> dict[str, Any]:
        return {
            "object_name": obj.object_name,
            "level1": obj.level1_category,
            "level2": obj.level2_category,
            "level3": obj.level3_category,
            "level4": obj.level4_category,
            "level5": obj.level5_category,
        }

    return {
        "id": dataset.id,
        "dataset_name": dataset.dataset_name,
        "dataset_uuid": dataset.dataset_uuid,
        "device_model": dataset.device_model,
        "end_effector_type": dataset.end_effector_type,
        "operation_platform_height": dataset.operation_platform_height,
        "yaml_file_path": dataset.yaml_file_path,
        "convert_path": dataset.convert_path,
        "scene_types": [scene.name for scene in dataset.scene_types],
        "atomic_actions": [action.action_name for action in dataset.atomic_actions],
        "objects": [_object_to_dict(obj) for obj in dataset.objects],
        "total_episodes": dataset.total_episodes,
    }


def _load_yaml_payload(db_payload: dict[str, Any]) -> dict[str, Any]:
    yaml_path = db_payload.get("yaml_file_path")
    if not yaml_path:
        return {"raw_yaml": {}, "scene_type": [], "atomic_actions": [], "objects": []}

    path = Path(str(yaml_path)).expanduser()
    if not path.exists():
        LOGGER.warning("YAML file %s does not exist, skipping YAML metadata merge", path)
        return {"raw_yaml": {}, "scene_type": [], "atomic_actions": [], "objects": []}

    try:
        return collect_from_yaml(
            yaml_file_path=path,
            dataset_name=db_payload.get("dataset_name", ""),
            dataset_uuid=db_payload.get("dataset_uuid", ""),
        )
    except Exception as exc:  # noqa: PERF203
        LOGGER.warning("Failed to parse YAML file %s: %s", path, exc)
        return {"raw_yaml": {}, "scene_type": [], "atomic_actions": [], "objects": []}


def _merge_statistics(
    meta_stats: dict[str, Any] | None,
    db_payload: dict[str, Any],
) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    if isinstance(meta_stats, dict):
        stats.update({k: v for k, v in meta_stats.items() if v is not None})

    if "total_episodes" not in stats and db_payload.get("total_episodes") is not None:
        stats["total_episodes"] = db_payload["total_episodes"]
    return stats
