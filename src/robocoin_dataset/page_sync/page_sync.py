#!/usr/bin/env python3
"""
Page Sync Orchestration Module

This module contains the main orchestration logic for syncing dataset information
to the page project. It coordinates the following workflow:
1. Detect and create directory structure (assets/dataset_info, assets/videos, assets/info)
2. Loop through pending tasks:
   - Sync task status (mark eligible datasets as PENDING)
   - Generate one task (mark PENDING -> PROCESSING)
   - Copy YAML file to dataset_info
   - Sample and compress video to videos directory
   - Align video name to match dataset name
   - Mark task as COMPLETED or FAILED
3. Generate consolidated metadata files:
   - consolidated_datasets.json: All metadata in one file
   - data_index.json: List of all YAML files

The actual business logic is implemented in utils.py and task.py.
"""

import logging
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from robocoin_dataset.database.database import DatasetDatabase


def _ensure_info_yaml_exists(
    hardlink_path: str,
    info_yaml_path: str,
    logger: logging.Logger,
) -> str:
    """
    Ensure info.yaml exists under dataset hardlink path.
    If missing, trigger metadata collect stage to generate it.
    """
    from robocoin_dataset.metadata.collect import InfoCollector

    info_yaml = Path(info_yaml_path)
    if info_yaml.exists():
        logger.debug("info.yaml already exists: %s", info_yaml)
        return str(info_yaml)

    logger.warning(
        "info.yaml missing at %s, triggering metadata collect for dataset path %s",
        info_yaml,
        hardlink_path,
    )
    collector = InfoCollector(
        dataset_path=hardlink_path,
        output_info_yaml_path=info_yaml,
    )
    generated_path = collector.collect()
    logger.info("Generated info.yaml via metadata collect: %s", generated_path)
    return str(generated_path)


def construce_target_file(
    db: "DatasetDatabase",
    session: "Session",
    target_dir: str,
    crf: int = 18,
    update_videos: bool = False,
    force_regenerate: bool = False,
    logger: logging.Logger | None = None,
) -> None:
    """
    Construct target file structure with upsert logic.
    The main orchestration function for page-needed-data construction.

    Target structure:
    target_dir/ (root of page project)
        assets/
            dataset_info/
                *.yml files
            videos/
                *.mp4 files
            thumbnails/
                *.jpg files
            info/
                consolidated_datasets.json
                data_index.json

    Args:
        db: Database connection
        session: SQLAlchemy session
        target_dir: Root directory of the page project
        crf: CRF value for video compression (default: 18, range: 0-51, lower = better quality)
        update_videos: If True, always regenerate videos and thumbnails; if False, skip existing ones (default: False)
        force_regenerate: If True, ignore existing COMPLETED status and rebuild assets whenever prerequisites are ready
        logger: Optional logger instance
    """
    from robocoin_dataset.page_sync.task import (
        _gen_one_page_sync_task,
        _mark_task_completed,
        _mark_task_failed,
        _sync_page_sync_status,
    )
    from robocoin_dataset.page_sync.utils import (
        _align_video_name_with_yaml,
        _copy_info_yaml,
        _compress_video_to_dst,
        _copy_robot_aliases_and_exclude,
        _gen_consolidation,
        _gen_data_index,
        _gen_video_thumbnail,
        _get_dataset_name,
        _sample_one_video_path,
        _validate_exist,
    )

    _logger = logger or logging.getLogger(__name__)
    target_root = Path(target_dir)

    # 1. Detect and create assets folder if it doesn't exist
    assets_dir = target_root / "assets"
    if not assets_dir.exists():
        assets_dir.mkdir(parents=True, exist_ok=True)
        _logger.debug(f"Created assets directory: {assets_dir}")
    else:
        _logger.debug(f"Assets directory already exists: {assets_dir}")

    # 2. Detect and create dataset_info and videos folders if they don't exist
    dataset_info_dir = assets_dir / "dataset_info"
    if not dataset_info_dir.exists():
        dataset_info_dir.mkdir(parents=True, exist_ok=True)
        _logger.debug(f"Created dataset_info directory: {dataset_info_dir}")
    else:
        _logger.debug(f"Dataset_info directory already exists: {dataset_info_dir}")

    videos_dir = assets_dir / "videos"
    if not videos_dir.exists():
        videos_dir.mkdir(parents=True, exist_ok=True)
        _logger.debug(f"Created videos directory: {videos_dir}")
    else:
        _logger.debug(f"Videos directory already exists: {videos_dir}")

    info_dir = assets_dir / "info"
    if not info_dir.exists():
        info_dir.mkdir(parents=True, exist_ok=True)
        _logger.debug(f"Created info directory: {info_dir}")
    else:
        _logger.debug(f"Info directory already exists: {info_dir}")

    thumbnails_dir = assets_dir / "thumbnails"
    if not thumbnails_dir.exists():
        thumbnails_dir.mkdir(parents=True, exist_ok=True)
        _logger.debug(f"Created thumbnails directory: {thumbnails_dir}")
    else:
        _logger.debug(f"Thumbnails directory already exists: {thumbnails_dir}")

    # 3-8. Main loop: sync -> generate task -> copy info yaml -> copy & compress videos -> align video name -> mark completed
    task_count = 0
    while True:
        # 3. Sync the task status
        _logger.debug("Syncing page sync status...")
        _sync_page_sync_status(session, _logger, force_regenerate=force_regenerate)

        # 4. Generate one task
        _logger.debug("Generating next task...")
        info_yaml_path, hardlink_path, dataset_uuid = _gen_one_page_sync_task(session)

        if info_yaml_path is None and hardlink_path is None and dataset_uuid is None:
            _logger.info("No more pending tasks to process")
            break

        if not dataset_uuid:
            _logger.error("No dataset_uuid returned from task generation")
            continue

        task_count += 1
        _logger.info(f"Processing task {task_count}: dataset_uuid={dataset_uuid}")
        _logger.debug(f"  info_yaml_path: {info_yaml_path}")
        _logger.debug(f"  hardlink_path: {hardlink_path}")

        try:
            info_yaml_path = _ensure_info_yaml_exists(
                hardlink_path=str(hardlink_path),
                info_yaml_path=str(info_yaml_path),
                logger=_logger,
            )
            if not _validate_exist(info_yaml_path, hardlink_path):
                _logger.error(
                    f"Validation failed for dataset {dataset_uuid}: "
                    f"info_yaml_path={info_yaml_path}, hardlink_path={hardlink_path}. "
                    f"Both paths must exist. Marking as FAILED."
                )
                err_msg = (
                    "Page sync validation failed: info_yaml_path and hardlink_path must both exist. "
                    f"info_yaml_path={info_yaml_path}, hardlink_path={hardlink_path}"
                )
                _mark_task_failed(session, dataset_uuid, err_msg)
                continue

            # 5. Copy pre-generated info.yaml into page dataset_info assets
            _logger.debug(f"Getting dataset name for {dataset_uuid}...")
            dataset_name = _get_dataset_name(session, dataset_uuid)
            if not dataset_name:
                _logger.error(f"Failed to get dataset name for dataset {dataset_uuid}")
                err_msg = f"Failed to get dataset name for dataset_uuid={dataset_uuid}"
                _mark_task_failed(session, dataset_uuid, err_msg)
                continue

            _logger.info(f"Dataset name: {dataset_name}")
            yaml_dst = dataset_info_dir / f"{dataset_name}.yaml"

            _logger.debug(
                "Copying info.yaml for dataset %s from %s to %s",
                dataset_uuid,
                info_yaml_path,
                yaml_dst,
            )
            _copy_info_yaml(str(info_yaml_path), str(yaml_dst))
            _logger.info("Copied info YAML to %s", yaml_dst)

            # 6. Sample and compress videos
            _logger.debug(f"Sampling video from hardlink path: {hardlink_path}...")
            sampled_video_path = _sample_one_video_path(hardlink_path)
            if not sampled_video_path:
                _logger.error(f"Failed to sample video from {hardlink_path}")
                err_msg = (
                    "Failed to sample video for page sync: no suitable video found under "
                    f"hardlink_path={hardlink_path}"
                )
                _mark_task_failed(session, dataset_uuid, err_msg)
                continue

            _logger.info(f"Sampled video: {sampled_video_path}")
            _logger.debug(f"Starting video compression with CRF={crf}...")
            _compress_video_to_dst(sampled_video_path, str(videos_dir), crf=crf, force_update=update_videos)
            _logger.info(f"Compressed video from {sampled_video_path} into {videos_dir}")

            # 7. Alighment-Rename videos
            _logger.debug("Aligning video name with dataset name...")
            compressed_video_name = Path(sampled_video_path).name
            compressed_video_path = videos_dir / compressed_video_name
            _align_video_name_with_yaml(str(yaml_dst), str(compressed_video_path), dataset_name)
            _logger.info(f"Aligned video name to {dataset_name}")

            # 7.5. Generate thumbnail after video is renamed
            video_suffix = compressed_video_path.suffix
            final_video_path = videos_dir / f"{dataset_name}{video_suffix}"
            _gen_video_thumbnail(str(final_video_path), str(thumbnails_dir), force_update=update_videos)
            _logger.info(f"Generated thumbnail for {dataset_name}")

            # 8. Update task status to COMPLETED
            _mark_task_completed(session, dataset_uuid)
            _logger.info(f"Successfully processed dataset: {dataset_name} ({dataset_uuid})")

        except Exception as e:
            _logger.error(f"Error processing task {dataset_uuid}: {e}", exc_info=True)
            err_msg = f"Error processing page sync task for dataset_uuid={dataset_uuid}: {e}\n{traceback.format_exc()}"
            _mark_task_failed(session, dataset_uuid, err_msg)

    # 9. Generate consolidated datasets and data index files
    _logger.info("Generating consolidated metadata files...")
    try:
        consolidated_path = info_dir / "consolidated_datasets.json"
        _logger.debug(f"Generating consolidated datasets at: {consolidated_path}")
        _gen_consolidation(str(dataset_info_dir), str(consolidated_path))

        data_index_path = info_dir / "data_index.json"
        _logger.debug(f"Generating data index at: {data_index_path}")
        _gen_data_index(str(dataset_info_dir), str(data_index_path))

        _logger.debug("Copying robot aliases file into info directory")
        _copy_robot_aliases_and_exclude(str(info_dir))

        _logger.info("Successfully generated consolidated metadata files")
    except Exception as e:
        _logger.error(f"Error generating consolidated metadata files: {e}", exc_info=True)

    _logger.info(f"Target file structure construction completed at: {target_dir}")


def main(
    db_path: str,
    target_dir: str,
    crf: int = 18,
    update_videos: bool = False,
    force_regenerate: bool = False,
    log_level: str = "INFO",
) -> None:
    """
    Main entry point for page sync operation.

    Args:
        db_path: Path to the SQLite database
        target_dir: Root directory of the page project
        crf: CRF value for video compression (default: 18, range: 0-51, lower = better quality)
        update_videos: If True, always regenerate videos and thumbnails; if False, skip existing ones (default: False)
        force_regenerate: If True, ignore existing COMPLETED status and rebuild assets whenever prerequisites are ready
        log_level: Logging level (default: INFO)
    """
    from datetime import datetime

    from robocoin_dataset.database.database import DatasetDatabase

    # Setup logging to logs/page/ directory
    log_dir = Path("logs/page")
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"page_sync_{timestamp}.log"

    # Configure logging with both file and console handlers
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Log file created at: {log_file}")

    # Initialize database
    db = DatasetDatabase(db_path)
    logger.info(f"Initialized database at: {db_path}")

    # Run the sync operation
    with db.with_session() as session:
        construce_target_file(
            db=db,
            session=session,
            target_dir=target_dir,
            crf=crf,
            update_videos=update_videos,
            force_regenerate=force_regenerate,
            logger=logger,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Sync dataset information to page project"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        required=True,
        help="Path to the SQLite database",
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        required=True,
        help="Root directory of the page project",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=18,
        help="CRF value for video compression (default: 18, range: 0-51, lower = better quality)",
    )
    parser.add_argument(
        "--update-videos",
        action="store_true",
        help="Force regenerate videos and thumbnails even if they exist (default: False)",
    )
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="Regenerate datasets even if their status already shows as COMPLETED",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    main(
        db_path=args.db_path,
        target_dir=args.target_dir,
        crf=args.crf,
        update_videos=args.update_videos,
        force_regenerate=args.force_regenerate,
        log_level=args.log_level,
    )
