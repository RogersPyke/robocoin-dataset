"""Local execution component for dataloader checker.

This module provides:
- DataLoaderChecker: Local execution class to check datasets sequentially.
"""

import logging
import traceback
from pathlib import Path

from robocoin_dataset.database.database import DatasetDatabase
from robocoin_dataset.database.models import DatasetDB, TaskStatus
from robocoin_dataset.dataloader_check.task import (
    _gen_one_dataloader_check_task,
    _sync_dataloader_check_tasks,
)
from robocoin_dataset.dataloader_check.utils import load_repo

class DataLoaderChecker:
    """
    Function: 
    Local execution class for dataloader checking.
    
    Expected input format:
    - config_path: str or Path, path to PostgreSQL configuration file (YAML format).
    - sample_rate: float, sampling rate for checking (0.0-1.0).
    - num_workers: int, number of workers for dataloader.
    - logger: logging.Logger, optional.
    - target_dataset_uuid: str, optional, specific dataset UUID to check.
    
    Expected output format:
    - None. Updates the database status.
    
    Expected usage/scenario:
    - Used for local, sequential checking of datasets against PostgreSQL.
    """
    def __init__(
        self,
        db_file_path: str | Path,
        sample_rate: float = 0.1,
        num_workers: int = 8,
        logger: logging.Logger | None = None,
        target_dataset_uuid: str = "",
    ) -> None:
        self.db_file_path: Path = Path(db_file_path).expanduser().absolute()
        self.db = DatasetDatabase(self.db_file_path)
        self.sample_rate = sample_rate
        self.num_workers = num_workers
        self.logger = logger or logging.getLogger(__name__)
        self.target_dataset_uuid = target_dataset_uuid

    def check_one_repo(self) -> None:
        """
        Processes repositories from the database until none pending.
        Syncs tasks, claims one, runs the check. On error: report, write to DB, continue.
        """
        while True:
            with self.db.with_session() as session:
                _sync_dataloader_check_tasks(session=session, target_dataset_uuid=self.target_dataset_uuid)
                dataset_uuid, hardlink = _gen_one_dataloader_check_task(
                    session=session, target_dataset_uuid=self.target_dataset_uuid
                )
                if dataset_uuid is None:
                    self.logger.info("No pending dataloader check tasks found.")
                    return

            self.logger.info(f"[START] Local check for dataset: {dataset_uuid}")
            try:
                load_repo(hardlink, sample_rate=self.sample_rate, num_workers=self.num_workers)
                with self.db.with_session() as session:
                    ds_item = (
                        session.query(DatasetDB).filter(DatasetDB.dataset_uuid == dataset_uuid).first()
                    )
                    if ds_item:
                        ds_item.data_loader_detection_status = TaskStatus.COMPLETED
                        ds_item.data_loader_detection_err_msg = None
                        session.commit()
                self.logger.info(f"[SUCCESS] Local check completed for dataset: {dataset_uuid}")
            except Exception:
                err_msg = traceback.format_exc()
                self.logger.error(f"[FAILED] Local check failed for dataset: {dataset_uuid}\n{err_msg}")
                try:
                    with self.db.with_session() as session:
                        ds_item = (
                            session.query(DatasetDB).filter(DatasetDB.dataset_uuid == dataset_uuid).first()
                        )
                        if ds_item:
                            ds_item.data_loader_detection_status = TaskStatus.FAILED
                            ds_item.data_loader_detection_err_msg = err_msg
                            session.commit()
                except Exception as db_err:
                    self.logger.error(f"[DB_ERR] Failed to write error to DB for {dataset_uuid}: {db_err}")

__all__ = ["DataLoaderChecker"]
