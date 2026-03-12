"""Server component for distributed dataloader checker.

This module provides:
- DataLoaderCheckerServer: Orchestrates task distribution to clients.
"""

import logging
from pathlib import Path

from robocoin_dataset.database.database import DatasetDatabase
from robocoin_dataset.database.models import DatasetDB, TaskStatus
from robocoin_dataset.distribution_computation.constant import (
    DATASET_UUID,
    ERR_MSG,
    TASK_RESULT_STATUS,
    TASK_SUCCESS,
)
from robocoin_dataset.distribution_computation.task_server import TaskServer
from robocoin_dataset.dataloader_check.task import (
    _gen_one_dataloader_check_task,
    _sync_dataloader_check_tasks,
)

# Constants for task communication
NUM_WORKERS = "num_workers"
SAMPLE_RATE = "sample_rate"
HARD_LINK_PATH = "hard_link_path"

class DataLoaderCheckerServer(TaskServer):
    """
    Function: 
    Distributed server for dataloader checking tasks against PostgreSQL.
    
    Expected input format:
    - db_file_path: str or Path, path to PostgreSQL configuration file (YAML format).
    - host: str, server host.
    - port: int, server port.
    - heartbeat_interval: float, interval for heartbeat pings.
    - timeout: float, timeout for pong responses.
    - logger: logging.Logger, optional.
    - sample_rate: float, sampling rate for checking (0.0-1.0).
    - num_workers: int, number of workers for dataloader.
    - target_dataset_uuid: str, optional, specific dataset UUID to check.
    
    Expected output format:
    - Distributed task content to clients.
    
    Expected usage/scenario:
    - Used to run dataloader checks across multiple machines/processes against PostgreSQL.
    """
    def __init__(
        self,
        db_file_path: str | Path,
        host: str = "0.0.0.0",
        port: int = 2010,
        heartbeat_interval: float = 30.0,
        timeout: float = 15.0,
        logger: logging.Logger | None = None,
        sample_rate: float = 0.1,
        num_workers: int = 8,
        target_dataset_uuid: str = "",
    ) -> None:
        super().__init__(
            logger=logger,
            host=host,
            port=port,
            heartbeat_interval=heartbeat_interval,
            timeout=timeout,
        )
        self.db_file_path: Path = Path(db_file_path).expanduser().absolute()
        self.db = DatasetDatabase(self.db_file_path)
        self.sample_rate = sample_rate
        self.num_workers = num_workers
        self.logger = logger or logging.getLogger(__name__)
        self.target_dataset_uuid = target_dataset_uuid

    def get_task_category(self) -> str:
        """Returns the category of tasks handled by this server."""
        return "dataset dataloader checker"

    def generate_task_content(self) -> dict | None:
        """
        Generates task content for a client.
        Syncs tasks and claims one from the database.
        """
        with self.db.with_session() as session:
            _sync_dataloader_check_tasks(session=session, target_dataset_uuid=self.target_dataset_uuid)
            dataset_uuid, hardlink_path = _gen_one_dataloader_check_task(
                session=session, target_dataset_uuid=self.target_dataset_uuid
            )

        if not dataset_uuid:
            return None
            
        return {
            DATASET_UUID: dataset_uuid,
            HARD_LINK_PATH: str(hardlink_path),
            NUM_WORKERS: self.num_workers,
            SAMPLE_RATE: self.sample_rate,
        }

    def handle_task_result(self, task_content: dict, task_result_content: dict) -> None:
        """
        Handles the result returned by a client.
        Updates the database status based on success or failure.
        Uses task_result_content for dataset_uuid when task_content is empty (e.g. server restart).
        """
        ds_uuid = task_content.get(DATASET_UUID) or task_result_content.get(DATASET_UUID)
        task_status = task_result_content.get(TASK_RESULT_STATUS)
        err_msg = task_result_content.get(ERR_MSG)

        if not ds_uuid:
            self.logger.warning(
                "Dataset UUID missing in task_content and task_result_content; cannot update status."
            )
            return

        with self.db.with_session() as session:
            item = session.query(DatasetDB).filter(DatasetDB.dataset_uuid == ds_uuid).first()
            if not item:
                self.logger.warning(f"Dataset {ds_uuid} not found in database during result handling.")
                return

            if task_status == TASK_SUCCESS:
                item.data_loader_detection_status = TaskStatus.COMPLETED
                item.data_loader_detection_err_msg = None
                self.logger.info(f"[SUCCESS] Dataset {ds_uuid} dataloader check completed.")
            else:
                item.data_loader_detection_status = TaskStatus.FAILED
                item.data_loader_detection_err_msg = err_msg
                # Log full error (and stack) so it is in logs; same content is stored in DB on commit
                self.logger.error(
                    "[FAILED] Dataset %s dataloader check failed. Error (stored in DB): %s",
                    ds_uuid,
                    err_msg or "",
                )
            session.commit()

__all__ = ["DataLoaderCheckerServer"]
