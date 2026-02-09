#!/usr/bin/env python3

import logging
from pathlib import Path

from .utils import (
    UploadConfig,
    LocalDsUploadUtil,
)


def upload_datasets_from_database_local(config: UploadConfig, logger: logging.Logger | None = None) -> None:
    """
    使用数据库管理上传数据集到远程 Hub（本地单机版）。

    直接调用 hub_upload_util 和 hub_upload_task 中的现成代码，不实现私有函数。

    Args:
        config: 上传配置
        logger: Logger 实例（可选）
    """
    import time

    from sqlalchemy.sql.expression import and_
    from tqdm import tqdm

    from robocoin_dataset.database.database import DatasetDatabase
    from robocoin_dataset.database.models import DatasetDB, TaskStatus

    from .task import (
        _gen_one_upload_task,
        _get_field,
        _get_hardlink_path_by_uuid,
        _mark_upload_completed,
        _mark_upload_failed,
        _sync_upload_status,
    )

    _logger = logger or logging.getLogger(__name__)

    start_time = time.time()

    # 初始化上传器
    _logger.info("正在初始化上传器...")
    uploader = LocalDsUploadUtil(config)

    # 验证并解析数据库路径
    db_path = Path(config.pg_cfg_path).expanduser().absolute()

    # Get the correct namespace based on hub_name
    hub_name = getattr(config, 'hub_name', 'huggingface')
    if hub_name == "modelscope" or hub_name == "ms":
        namespace = config.ms_namespace
        hub_display = "modelscope"
    else:  # huggingface or hf
        namespace = config.hf_namespace
        hub_display = "huggingface"

    # 打印初始配置
    _logger.info(f"🚀 Upload: {hub_display}/{namespace}")
    tqdm.write(f"🚀 Upload: {hub_display}/{namespace}")

    # 连接数据库
    db = DatasetDatabase(db_path)

    # 处理数据集的计数器
    uploaded_count = 0
    failed_count = 0
    skipped_count = 0

    # 创建进度条（稍后会在第一次同步后更新 total）
    pbar = tqdm(
        desc="📤 Uploading",
        unit="ds",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
    )

    ##### 主循环 #####
    try:
        while True:
            # 获取上传状态字段
            hub_name = getattr(config, 'hub_name', 'huggingface')
            upload_status_field = _get_field(DatasetDB, "upload_status", hub_name)
            upload_status_col = getattr(DatasetDB, upload_status_field)

            with db.with_session() as session:
                # 同步数据集上传状态
                _sync_upload_status(session, hub_name=hub_name, logger=_logger)

                # 统计待上传的数据集数量
                pending_count = session.query(DatasetDB).filter(
                    and_(
                        DatasetDB.visualize_check_status == TaskStatus.COMPLETED,
                        upload_status_col == TaskStatus.PENDING,
                    )
                ).count()

                if pending_count == 0:
                    break

                # 更新进度条总数
                if pbar.total is None or pbar.total != pending_count + uploaded_count + failed_count + skipped_count:
                    pbar.total = pending_count + uploaded_count + failed_count + skipped_count

                # 获取下一个待上传的数据集任务
                hub_name = getattr(config, 'hub_name', 'huggingface')
                dataset_uuid = _gen_one_upload_task(
                    session, hub_name=hub_name, logger=_logger
                )

            if dataset_uuid is None:
                # 没有更多任务
                break

            # 获取 hardlink path
            with db.with_session() as session:
                hardlink_path = _get_hardlink_path_by_uuid(session, dataset_uuid, logger=_logger)
            
            if hardlink_path is None:
                _logger.warning(f"Hardlink path not found for dataset {dataset_uuid}, skipping")
                continue

            # 获取数据集名称用于日志
            dataset_name = hardlink_path.name.removesuffix("_qced_hardlink").removesuffix("_hardlink")
            pbar.set_description(f"📤 {dataset_name[:30]:30s}")

            # 打印状态到控制台
            tqdm.write(f"  📦 Processing: {dataset_name}")
            tqdm.write(f"     UUID: {dataset_uuid}")
            tqdm.write(f"     Path: {hardlink_path}")

            ##### 上传数据集 - 直接调用现成的方法 #####
            success, error_msg = uploader._upload_one_dataset(hardlink_path)

            # 处理结果
            hub_name = getattr(config, 'hub_name', 'huggingface')
            if success:
                with db.with_session() as session:
                    _mark_upload_completed(session, dataset_uuid, hub_name, _logger)
                uploaded_count += 1
                _logger.info(f"{dataset_name}: ✅ Successfully uploaded")
            else:
                with db.with_session() as session:
                    _mark_upload_failed(session, dataset_uuid, error_msg, hub_name, _logger)
                failed_count += 1
                tqdm.write(f"  ❌ Failed: {error_msg}")
                _logger.error(f"{dataset_name}: {error_msg}")

            pbar.update(1)

    finally:
        # 确保进度条被关闭
        pbar.close()

    # 计算耗时
    end_time = time.time()
    elapsed_seconds = end_time - start_time
    hours, remainder = divmod(int(elapsed_seconds), 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours > 0:
        time_str = f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        time_str = f"{minutes}m {seconds}s"
    else:
        time_str = f"{seconds}s"

    # 最终汇总
    total_processed = uploaded_count + failed_count + skipped_count
    if total_processed == 0:
        _logger.info("No datasets to upload")
        tqdm.write("No datasets to upload")
        return

    status = f"✅ {uploaded_count}/{total_processed}"
    if failed_count > 0:
        status += f" | ❌ {failed_count}"
    if skipped_count > 0:
        status += f" | ⏭️  {skipped_count}"
    status += f" | ⏱️  {time_str}"

    _logger.info(status)
    _logger.info(f"Total upload time: {elapsed_seconds:.2f}s ({time_str})")
    tqdm.write("")  # 空行用于间隔
    tqdm.write(status)


def upload_datasets_main(config: UploadConfig,
logger: logging.Logger | None = None
) -> None:
    """
    使用数据库管理上传数据集到远程 Hub。

    这是主要的业务逻辑函数，编排上传流程。

    Args:
        config: 上传配置
        logger: Logger 实例（可选）
    """
    from tqdm import tqdm

    _logger = logger or logging.getLogger(__name__)

    try:
        _logger.info(f"Config readme_only: {getattr(config, 'readme_only', 'NOT_SET')}")
        if config.upload_readme_only:
            # README-only mode: only generate README files
            _logger.info("开始README-only流程...")
            upload_datasets_from_database_local(config, logger=_logger)
        else:
            # Normal upload mode
            _logger.info("开始上传流程...")
            upload_datasets_from_database_local(config, logger=_logger)

        _logger.info("✅ 流程完成")
        tqdm.write("\n✅ 流程完成")

    except KeyboardInterrupt:
        _logger.warning("\n⚠️  上传被用户中断")
        tqdm.write("\n⚠️  上传被用户中断")
        raise
    except Exception as e:
        _logger.error(f"❌ 上传失败: {e}", exc_info=True)
        tqdm.write(f"\n❌ 上传失败: {e}")
        raise
