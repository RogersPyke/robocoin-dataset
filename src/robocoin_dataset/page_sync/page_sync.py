#!/usr/bin/env python3
"""
Page Sync Orchestration Module

Page sync consumes info.yaml only (same pattern as readme generation):
- If consumable info.yaml exists at dataset root, use it for (1) video collect/compress,
  (2) thumbnail, (3) meta file write and consolidated file for the page, (4) optional upload.
- If no info.yaml exists, call metadata/ (InfoCollector) to generate it, then consume.
  If info.yaml generation fails, raise and do not continue.

No internal data collection from DB for content: dataset name and display fields come
from info.yaml. Task list (which datasets to process) still comes from DB; per-dataset
content is read only from info.yaml.

Workflow:
1. Create directory structure (assets/dataset_info, videos, thumbnails, info).
2. Get pending entries (hardlink_path, dataset_uuid) from DB.
3. For each entry: ensure info.yaml (generate via metadata if missing; raise on failure),
   then consume info.yaml to get dataset_name and do: copy YAML, sample/compress video,
   thumbnail, align names; mark COMPLETED/FAILED in DB.
4. Generate consolidated_datasets.json and data_index.json from assets/dataset_info.
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
    Ensure consumable info.yaml exists. If missing, call metadata collect to generate it.
    If collect raises, the exception propagates (caller must not continue).
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
    # Do not catch: on failure caller must raise and refuse to continue.
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
    from robocoin_dataset.page_sync._task import (
        get_pending_page_sync_entries,
        _mark_task_completed,
        _mark_task_failed,
        _mark_task_processing,
    )
    from robocoin_dataset.page_sync._utils import (
        _align_video_name_with_yaml,
        _copy_info_yaml,
        _compress_video_to_dst,
        _copy_robot_aliases_and_exclude,
        _gen_consolidation,
        _gen_data_index,
        _gen_video_thumbnail,
        _get_dataset_name_from_info_yaml,
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

    # 3. Get pending entries (hardlink_path, dataset_uuid); content will come from info.yaml only.
    _logger.debug("Getting pending page sync entries...")
    entries = get_pending_page_sync_entries(session, _logger, force_regenerate=force_regenerate)
    if not entries:
        _logger.info("No pending tasks to process")
    else:
        _logger.info("Processing %d pending task(s)", len(entries))

    task_count = 0
    for hardlink_path, dataset_uuid in entries:
        info_yaml_path = str(Path(hardlink_path) / "info.yaml")
        task_count += 1
        _logger.info("Processing task %d: dataset_uuid=%s", task_count, dataset_uuid)
        _logger.debug("  hardlink_path: %s", hardlink_path)
        _logger.debug("  info_yaml_path: %s", info_yaml_path)

        _mark_task_processing(session, dataset_uuid)

        # Ensure consumable info.yaml; if missing, call metadata collect. On failure, raise and stop.
        info_yaml_path = _ensure_info_yaml_exists(
            hardlink_path=hardlink_path,
            info_yaml_path=info_yaml_path,
            logger=_logger,
        )

        try:
            if not _validate_exist(info_yaml_path, hardlink_path):
                err_msg = (
                    "Page sync validation failed: info_yaml_path and hardlink_path must both exist. "
                    f"info_yaml_path={info_yaml_path}, hardlink_path={hardlink_path}"
                )
                _mark_task_failed(session, dataset_uuid, err_msg)
                continue

            # Consume info.yaml only for dataset name (no DB lookup for content).
            dataset_name = _get_dataset_name_from_info_yaml(info_yaml_path, _logger)
            if not dataset_name:
                err_msg = f"dataset_name missing in info.yaml and fallback empty: {info_yaml_path}"
                _mark_task_failed(session, dataset_uuid, err_msg)
                continue

            _logger.info("Dataset name: %s", dataset_name)
            yaml_dst = dataset_info_dir / f"{dataset_name}.yaml"

            _logger.debug("Copying info.yaml from %s to %s", info_yaml_path, yaml_dst)
            _copy_info_yaml(info_yaml_path, str(yaml_dst))
            _logger.info("Copied info YAML to %s", yaml_dst)

            _logger.debug("Sampling video from hardlink path: %s", hardlink_path)
            sampled_video_path = _sample_one_video_path(hardlink_path)
            if not sampled_video_path:
                err_msg = (
                    "Failed to sample video for page sync: no suitable video found under "
                    f"hardlink_path={hardlink_path}"
                )
                _mark_task_failed(session, dataset_uuid, err_msg)
                continue

            _logger.info("Sampled video: %s", sampled_video_path)
            _compress_video_to_dst(sampled_video_path, str(videos_dir), crf=crf, force_update=update_videos)
            _logger.info("Compressed video into %s", videos_dir)

            compressed_video_name = Path(sampled_video_path).name
            compressed_video_path = videos_dir / compressed_video_name
            _align_video_name_with_yaml(str(yaml_dst), str(compressed_video_path), dataset_name)
            _logger.info("Aligned video name to %s", dataset_name)

            video_suffix = compressed_video_path.suffix
            final_video_path = videos_dir / f"{dataset_name}{video_suffix}"
            _gen_video_thumbnail(str(final_video_path), str(thumbnails_dir), force_update=update_videos)
            _logger.info("Generated thumbnail for %s", dataset_name)

            _mark_task_completed(session, dataset_uuid)
            _logger.info("Successfully processed dataset: %s (%s)", dataset_name, dataset_uuid)

        except Exception as e:
            _logger.error("Error processing task %s: %s", dataset_uuid, e, exc_info=True)
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
        db_path: Path to the PostgreSQL YAML config file
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
        "--db-cfg-path",
        type=str,
        required=True,
        help="Path to the PostgreSQL YAML config file",
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
        db_path=args.db_cfg_path,
        target_dir=args.target_dir,
        crf=args.crf,
        update_videos=args.update_videos,
        force_regenerate=args.force_regenerate,
        log_level=args.log_level,
    )
