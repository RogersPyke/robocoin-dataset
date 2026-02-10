"""
Metadata Sync Service
=====================

该模块提供统一的元数据收集入口，确保 README 生成与页面静态资源
构建依赖同一套逻辑。任何需要从数据库和硬链接目录组合信息的流程，
都应该通过这里暴露的服务进行调用，从而避免逻辑漂移。

它被哪些地方调用（帮助你快速定位调用链）：
- `robocoin_dataset/page_sync/page_sync.py`：页面工程需要的 `assets/dataset_info/*.yml` 由这里统一生成
- `robocoin_dataset/hub_upload/lerobot/hub_upload_util.py`：本地上传流程可复用服务生成统一 metadata

这个文件的定位（为什么单独存在）：
- `metadata_collect.py` 是“纯函数式的聚合器”（输入 hardlink + db，输出 UnifiedMetadata）
- `metadata_service.py` 把“db_path 解析/校验 + 日志上下文 + 输出 YAML 落盘”封装成有状态对象，
  让上层业务代码更短、更不容易漏掉路径校验/目录创建等样板逻辑。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .metadata_collect import create_unified_metadata
from .unified_metadata_def import UnifiedMetadata


@dataclass(slots=True)
class MetadataSyncService:
    """
    统一的元数据收集服务。

    该服务将数据库路径、日志器等上下文封装为一个有状态对象，对外暴露
    「收集」与「落盘」两个方法，供 README 上传与页面资源生成复用。
    """

    db_file_path: str | Path
    logger: logging.Logger | None = None

    def __post_init__(self) -> None:
        # 统一解析并验证数据库路径，避免在多个入口重复编写样板代码
        resolved = Path(self.db_file_path).expanduser().absolute()
        if not resolved.exists():
            raise FileNotFoundError(f"数据库文件不存在: {resolved}")
        self.db_file_path = resolved

    def collect_unified_metadata(
        self,
        *,
        hardlink_path: str | Path,
        dataset_uuid: str | None = None,
    ) -> UnifiedMetadata:
        """
        从硬链接目录与数据库中提取完整的 UnifiedMetadata。

        Args:
            hardlink_path: 数据集硬链接目录（*_hardlink）
            dataset_uuid: 可选，显式指定数据集 UUID

        Returns:
            UnifiedMetadata: 统一的元数据对象
        """
        metadata = create_unified_metadata(
            hardlink_path=hardlink_path,
            db_file_path=self.db_file_path,
            dataset_uuid=dataset_uuid,
        )
        if self.logger:
            self.logger.debug(
                "构建统一元数据成功: dataset=%s, uuid=%s",
                Path(hardlink_path).name,
                dataset_uuid,
            )
        return metadata

    def write_unified_metadata_yaml(
        self,
        *,
        dst_yaml_path: str | Path,
        hardlink_path: str | Path,
        dataset_uuid: str | None = None,
    ) -> tuple[UnifiedMetadata, str]:
        """
        生成 YAML 文件并返回元数据对象，供调用方二次复用。

        Args:
            dst_yaml_path: 输出 YAML 路径
            hardlink_path: 数据集硬链接目录
            dataset_uuid: 数据集 UUID，可选

        Returns:
            (UnifiedMetadata, str): 元数据对象及 YAML 字符串内容
        """
        metadata = self.collect_unified_metadata(
            hardlink_path=hardlink_path,
            dataset_uuid=dataset_uuid,
        )

        dst_path = Path(dst_yaml_path).expanduser().absolute()
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        yaml_str = metadata.to_yaml(str(dst_path))

        if self.logger:
            self.logger.debug(
                "写入统一元数据 YAML 成功: %s (size=%d)",
                dst_path,
                len(yaml_str),
            )

        return metadata, yaml_str


__all__ = ["MetadataSyncService"]
