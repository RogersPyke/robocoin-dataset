#!/usr/bin/env python3
"""
Sync Hub Upload Status to Database

Fetches dataset names from HuggingFace and ModelScope, matches with database dataset_name,
updates upload statuses accordingly.

Workflow:
1. HF matched: huggingface_upload_status = COMPLETED, dataset_info_sync_status = PENDING
2. MS matched: ms_upload_status = COMPLETED
3. Compare HF vs MS repos, identify extra dataset on ModelScope

Usage:
    python scripts/sync_hub_status_to_db.py --db-cfg-path db/postgresql_config.yaml

Dependencies:
    - psycopg2-binary
    - sqlalchemy
    - huggingface-hub
    - requests
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

import requests
from huggingface_hub import HfApi

from robocoin_dataset.database.database import DatasetDatabase
from robocoin_dataset.database.models import DatasetDB, TaskStatus


DEFAULT_PG_CONFIG = project_root / "db" / "postgresql_config.yaml"
HF_NAMESPACE = "RoboCOIN"
MS_NAMESPACE = "RoboCOIN"
LOG_DIR = project_root / "logs"


def setup_logger() -> logging.Logger:
    """Setup logger with both file and console handlers."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    log_file = LOG_DIR / f"sync_hub_status_{timestamp}.log"

    logger = logging.getLogger("sync_hub_status")
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter("[%(name)s] [%(levelname)s] %(message)s")
    file_handler.setFormatter(file_format)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter("[%(levelname)s] %(message)s")
    console_handler.setFormatter(console_format)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def get_hf_datasets(namespace: str, logger: logging.Logger) -> list[str]:
    """
    Fetch dataset repo names from HuggingFace.

    @input: namespace - HuggingFace organization name
    @output: list of dataset repo names (without namespace prefix)
    @scenario: List all public datasets under a HuggingFace org
    """
    logger.info("[HuggingFace] Fetching datasets for namespace: %s", namespace)

    try:
        api = HfApi()
        repos = api.list_datasets(author=namespace)

        repo_names = []
        for repo in repos:
            repo_id = repo.id
            name = repo_id.split("/", 1)[1] if "/" in repo_id else repo_id
            repo_names.append(name)

        logger.info("[HuggingFace] Found %d datasets", len(repo_names))
        logger.debug("[HuggingFace] Datasets: %s", repo_names[:10])
        return repo_names

    except Exception as e:
        logger.error("[HuggingFace] Failed to fetch datasets: %s", e)
        raise


def get_ms_datasets(namespace: str, logger: logging.Logger) -> list[str]:
    """
    Fetch dataset names from ModelScope via public API.

    @input: namespace - ModelScope organization name
    @output: list of dataset repo names
    @scenario: List all public datasets under a ModelScope org
    """
    logger.info("[ModelScope] Fetching datasets for namespace: %s", namespace)

    try:
        url = "https://www.modelscope.cn/api/v1/datasets"
        all_names = []
        page_number = 1
        page_size = 50

        while True:
            params = {
                "owner": namespace,
                "PageNumber": page_number,
                "PageSize": page_size,
            }

            response = requests.get(url, params=params, timeout=30)

            if response.status_code != 200:
                logger.warning("[ModelScope] API returned HTTP %s", response.status_code)
                break

            data = response.json()
            datasets = data.get("Data", [])

            for dataset in datasets:
                name = dataset.get("Name", "")
                if name:
                    all_names.append(name)

            total_count = data.get("TotalCount", 0)
            if len(all_names) >= total_count:
                break

            page_number += 1

        logger.info("[ModelScope] Found %d datasets", len(all_names))
        logger.debug("[ModelScope] Datasets: %s", all_names[:10])
        return all_names

    except Exception as e:
        logger.error("[ModelScope] Failed to fetch datasets: %s", e)
        raise


def get_db_dataset_names(pg_config_path: Path, logger: logging.Logger) -> dict[str, str]:
    """
    Get all dataset names from database, mapping dataset_name -> dataset_uuid.

    @input: pg_config_path - path to PostgreSQL config YAML
    @output: dict mapping dataset_name.lower() -> dataset_uuid
    @scenario: Read all dataset records from database
    """
    logger.info("[Database] Fetching dataset names...")

    db = DatasetDatabase(pg_config_path)
    name_to_uuid = {}

    with db.with_session() as session:
        results = session.query(
            DatasetDB.dataset_uuid,
            DatasetDB.dataset_name,
        ).all()

        for row in results:
            uuid, name = row
            if name:
                name_lower = name.lower()
                if name_lower in name_to_uuid:
                    logger.warning(
                        "[Database] Duplicate dataset_name (case-insensitive): %s, keeping first",
                        name,
                    )
                else:
                    name_to_uuid[name_lower] = uuid

    logger.info("[Database] Found %d unique dataset names", len(name_to_uuid))
    return name_to_uuid


def update_hf_status(
    pg_config_path: Path,
    matched_uuids: list[str],
    logger: logging.Logger,
) -> int:
    """
    Update huggingface_upload_status to COMPLETED and dataset_info_sync_status to PENDING.

    @input: pg_config_path, matched_uuids - list of dataset UUIDs to update
    @output: number of records updated
    @scenario: Mark HF upload as completed for matched datasets
    """
    logger.info(
        "[Database] Updating HuggingFace upload status for %d datasets...", len(matched_uuids)
    )

    if not matched_uuids:
        return 0

    db = DatasetDatabase(pg_config_path)
    updated = 0

    with db.with_session() as session:
        for uuid in matched_uuids:
            dataset = session.query(DatasetDB).filter(DatasetDB.dataset_uuid == uuid).first()

            if dataset:
                dataset.huggingface_upload_status = TaskStatus.COMPLETED
                dataset.dataset_info_sync_status = TaskStatus.PENDING
                updated += 1
                logger.debug(
                    "[Database] Updated HF status for dataset_uuid=%s, dataset_name=%s",
                    uuid,
                    dataset.dataset_name,
                )

        session.commit()

    logger.info("[Database] Updated %d HuggingFace records", updated)
    return updated


def update_ms_status(
    pg_config_path: Path,
    matched_uuids: list[str],
    logger: logging.Logger,
) -> int:
    """
    Update ms_upload_status to COMPLETED.

    @input: pg_config_path, matched_uuids - list of dataset UUIDs to update
    @output: number of records updated
    @scenario: Mark MS upload as completed for matched datasets
    """
    logger.info(
        "[Database] Updating ModelScope upload status for %d datasets...", len(matched_uuids)
    )

    if not matched_uuids:
        return 0

    db = DatasetDatabase(pg_config_path)
    updated = 0

    with db.with_session() as session:
        for uuid in matched_uuids:
            dataset = session.query(DatasetDB).filter(DatasetDB.dataset_uuid == uuid).first()

            if dataset:
                dataset.ms_upload_status = TaskStatus.COMPLETED
                updated += 1
                logger.debug(
                    "[Database] Updated MS status for dataset_uuid=%s, dataset_name=%s",
                    uuid,
                    dataset.dataset_name,
                )

        session.commit()

    logger.info("[Database] Updated %d ModelScope records", updated)
    return updated


def generate_report(
    hf_datasets: list[str],
    ms_datasets: list[str],
    hf_matched: list[str],
    ms_matched: list[str],
    hf_updated: int,
    ms_updated: int,
    extra_in_ms: list[str],
    output_path: Path,
    logger: logging.Logger,
) -> dict:
    """
    Generate detailed report and save to JSON.

    @input: all collected data
    @output: report dict
    @scenario: Create summary report of sync operation
    """
    report = {
        "timestamp": datetime.now().isoformat(),
        "huggingface": {
            "total_repos": len(hf_datasets),
            "matched_in_db": len(hf_matched),
            "db_updated_count": hf_updated,
            "matched_names": hf_matched,
        },
        "modelscope": {
            "total_repos": len(ms_datasets),
            "matched_in_db": len(ms_matched),
            "db_updated_count": ms_updated,
            "matched_names": ms_matched,
        },
        "comparison": {
            "only_in_huggingface": sorted(set(hf_datasets) - set(ms_datasets)),
            "only_in_modelscope": extra_in_ms,
            "in_both": sorted(set(hf_datasets) & set(ms_datasets)),
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info("[Report] Saved to %s", output_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync HuggingFace and ModelScope upload status to database"
    )
    parser.add_argument(
        "--db-cfg-path",
        type=Path,
        default=DEFAULT_PG_CONFIG,
        help=f"PostgreSQL config YAML path (default: {DEFAULT_PG_CONFIG})",
    )
    args = parser.parse_args()

    logger = setup_logger()

    logger.info("=" * 70)
    logger.info("SYNC HUB UPLOAD STATUS TO DATABASE")
    logger.info("=" * 70)
    logger.info("PostgreSQL config: %s", args.db_cfg_path)
    logger.info("")

    try:
        logger.info("[Step 1] Fetch datasets from HuggingFace...")
        logger.info("-" * 70)
        hf_datasets = get_hf_datasets(HF_NAMESPACE, logger)
        hf_datasets_lower = {name.lower(): name for name in hf_datasets}
        logger.info("")

        logger.info("[Step 2] Fetch datasets from ModelScope...")
        logger.info("-" * 70)
        ms_datasets = get_ms_datasets(MS_NAMESPACE, logger)
        ms_datasets_lower = {name.lower(): name for name in ms_datasets}
        logger.info("")

        logger.info("[Step 3] Fetch dataset names from database...")
        logger.info("-" * 70)
        db_name_to_uuid = get_db_dataset_names(args.db_cfg_path, logger)
        logger.info("")

        logger.info("[Step 4] Match and update HuggingFace status...")
        logger.info("-" * 70)
        hf_matched_uuids = []
        hf_matched_names = []
        for db_name_lower, uuid in db_name_to_uuid.items():
            if db_name_lower in hf_datasets_lower:
                hf_matched_uuids.append(uuid)
                hf_matched_names.append(hf_datasets_lower[db_name_lower])

        hf_updated = update_hf_status(args.db_cfg_path, hf_matched_uuids, logger)
        logger.info("")

        logger.info("[Step 5] Match and update ModelScope status...")
        logger.info("-" * 70)
        ms_matched_uuids = []
        ms_matched_names = []
        for db_name_lower, uuid in db_name_to_uuid.items():
            if db_name_lower in ms_datasets_lower:
                ms_matched_uuids.append(uuid)
                ms_matched_names.append(ms_datasets_lower[db_name_lower])

        ms_updated = update_ms_status(args.db_cfg_path, ms_matched_uuids, logger)
        logger.info("")

        logger.info("[Step 6] Compare HuggingFace vs ModelScope...")
        logger.info("-" * 70)
        hf_lower = set(hf_datasets_lower.keys())
        ms_lower = set(ms_datasets_lower.keys())

        extra_in_ms_lower = ms_lower - hf_lower
        extra_in_ms = [ms_datasets_lower[name] for name in extra_in_ms_lower]

        if extra_in_ms:
            logger.info(
                "[Comparison] ModelScope has %d EXTRA datasets (not in HuggingFace):",
                len(extra_in_ms),
            )
            for name in sorted(extra_in_ms):
                logger.info("  - %s", name)
        else:
            logger.info("[Comparison] No extra datasets on ModelScope")
        logger.info("")

        logger.info("[Step 7] Generate report...")
        logger.info("-" * 70)
        output_path = (
            LOG_DIR / f"sync_hub_status_report_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
        )
        report = generate_report(
            hf_datasets=hf_datasets,
            ms_datasets=ms_datasets,
            hf_matched=hf_matched_names,
            ms_matched=ms_matched_names,
            hf_updated=hf_updated,
            ms_updated=ms_updated,
            extra_in_ms=extra_in_ms,
            output_path=output_path,
            logger=logger,
        )
        logger.info("")

        logger.info("=" * 70)
        logger.info("SUMMARY REPORT")
        logger.info("=" * 70)
        logger.info("")
        logger.info("HuggingFace:")
        logger.info("  Total repos on HF:         %d", len(hf_datasets))
        logger.info("  Matched in DB:             %d", len(hf_matched_names))
        logger.info("  DB records updated:        %d", hf_updated)
        logger.info(
            "  Status: huggingface_upload_status=COMPLETED, dataset_info_sync_status=PENDING"
        )
        logger.info("")
        logger.info("ModelScope:")
        logger.info("  Total repos on MS:         %d", len(ms_datasets))
        logger.info("  Matched in DB:             %d", len(ms_matched_names))
        logger.info("  DB records updated:        %d", ms_updated)
        logger.info("  Status: ms_upload_status=COMPLETED")
        logger.info("")
        logger.info("Comparison:")
        logger.info("  Only in HuggingFace:       %d", len(set(hf_datasets) - set(ms_datasets)))
        logger.info("  Only in ModelScope:        %d", len(extra_in_ms))
        logger.info("  In both platforms:         %d", len(set(hf_datasets) & set(ms_datasets)))
        logger.info("")
        if extra_in_ms:
            logger.info("  ModelScope EXTRA datasets (investigation needed):")
            for name in sorted(extra_in_ms):
                logger.info("    - %s", name)
        logger.info("")
        logger.info("Report saved to: %s", output_path)
        logger.info("=" * 70)

    except Exception as e:
        logger.error("[ERROR] Sync failed: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
