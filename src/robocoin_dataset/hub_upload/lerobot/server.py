"""Server component for distributed dataloader detection tasks.

This module provides the server-side orchestration for:
- Task distribution to multiple clients
- Hardlink validation before task assignment
- Database updates based on client results
"""

import logging
import traceback
from pathlib import Path

from robocoin_dataset.database.database import DatasetDatabase
from robocoin_dataset.database.models import DatasetDB
from robocoin_dataset.distribution_computation.constant import DATASET_UUID
from robocoin_dataset.distribution_computation.task_server import TaskServer
from robocoin_dataset.format_converter.tolerobot.constant import LEFORMAT_PATH
from ..config.constant import DatasetsHubEnum
from .task import (
    _gen_one_dataset_upload_task,
    _mark_upload_completed,
    _mark_upload_failed,
    _sync_datasets_upload_status,
)
from robocoin_dataset.prepare_metadata.metadata_collect import create_unified_metadata

TASK_CATEGORY = "hub_upload"


class HubUploadServer(TaskServer):
    """Task distribution server for hub upload."""

    def __init__(
        self,
        db_file_path: str | Path,
        summary_logger: logging.Logger,
        hub_name: DatasetsHubEnum = DatasetsHubEnum.huggingface,
        token: str = "",
        namespace: str = "",
        output_path: str = "",
        force_overwrite: bool = False,
        readme_only: bool = False,
        host: str = "0.0.0.0",
        port: int = 2100,
        heartbeat_interval: float = 30.0,
        timeout: float = 90.0,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize hub upload server.
        Parameters that wont change in the whole loop
        get initialized here, including token, db, and namespace...
        """
        super().__init__(
            logger=logger,
            host=host,
            port=port,
            heartbeat_interval=heartbeat_interval,
            timeout=timeout,
        )

        if not db_file_path:
            raise ValueError("db_file_path is required and cannot be None or empty")
        self.db_file_path: Path = Path(db_file_path).expanduser().absolute()
        if not self.db_file_path.exists():
            raise FileNotFoundError(f"Database file not found: {self.db_file_path}")
        if not self.db_file_path.is_file():
            raise ValueError(f"Database path is not a file: {self.db_file_path}")

        self.db = DatasetDatabase(self.db_file_path)
        self.logger = logger or logging.getLogger(__name__)
        self.summary_logger = summary_logger
        self.hub_name = hub_name

        # Store client configuration parameters to be sent with tasks
        # NOTE: Client should not need to know any database path – all metadata
        # is prepared on the server side and sent with each task.
        self.client_token = token
        self.client_namespace = namespace
        self.client_output_path = output_path or "./dataset_info"
        self.client_force_overwrite = force_overwrite
        self.client_readme_only = readme_only
        self.logger.info(f"Server db_file_path: {self.db_file_path}")

        self.datasets_succeeded = 0
        self.datasets_failed = 0

    def get_task_category(self) -> str:
        return TASK_CATEGORY

    def generate_task_content(self) -> dict | None:
        while True:
            dataset_uuid = None

            # Step 1: Sync and claim task (with DB session, includes hardlink validation)
            try:
                with self.db.with_session() as session:
                    _sync_datasets_upload_status(session, self.hub_name, logger=self.logger)

                    dataset_uuid, hardlink_path = _gen_one_dataset_upload_task(
                        session, self.hub_name, logger=self.logger
                    )
                    if dataset_uuid is None:
                        break
                if hardlink_path is None:
                    raise FileNotFoundError(f"No hard_link_path found for dataset {dataset_uuid}")
                if not hardlink_path.exists():
                    raise FileNotFoundError(f"No such hardlink found in {hardlink_path}")

                self.logger.debug(f"Using existing hardlink for client: {hardlink_path}")

                # Step 2: Build unified metadata on the server side so that
                # clients never need to access the database.
                try:
                    unified_metadata = create_unified_metadata(
                        hardlink_path=hardlink_path,
                        db_file_path=self.db_file_path,
                        dataset_uuid=dataset_uuid,
                    )
                    metadata_dict = unified_metadata.to_dict()
                except Exception as e:  # noqa: PERF203
                    err_msg = (
                        f"Unified metadata collection failed on server for dataset {dataset_uuid}: {e}\n"
                        f"{traceback.format_exc()}"
                    )
                    self.logger.exception(f"❌ {dataset_uuid}: {err_msg}")
                    with self.db.with_session() as session:
                        _mark_upload_failed(session, dataset_uuid, err_msg, self.hub_name, logger=self.logger)
                    self.summary_logger.debug(f"❌ {dataset_uuid}: {err_msg}")
                    self.datasets_failed += 1
                    # Try to fetch next available task
                    continue

                # in case of success:
                task_config = {
                    DATASET_UUID: dataset_uuid,
                    LEFORMAT_PATH: str(hardlink_path),  # Send hardlink path to client
                    "metadata": metadata_dict,  # Send pre-built unified metadata
                    # Send client configuration parameters with the task
                    "client_config": {
                        "token": self.client_token,
                        "namespace": self.client_namespace,
                        "hub_name": self.hub_name.value,
                        "output_path": self.client_output_path,
                        "force_overwrite": self.client_force_overwrite,
                        "readme_only": self.client_readme_only,
                    },
                }
                self.logger.debug(f"Sending task config for dataset {dataset_uuid}")
                return task_config

                # in case of failure:
            except FileNotFoundError as e:
                err_msg = f"Hardlink assertion failed: {e}\n{traceback.format_exc()}"
                self.logger.exception(f"❌ {dataset_uuid}: {err_msg}")
                with self.db.with_session() as session:
                    _mark_upload_failed(session, dataset_uuid, err_msg, self.hub_name, logger=self.logger)
                self.summary_logger.debug(f"❌ {dataset_uuid}: {err_msg}")
                self.datasets_failed += 1
                self.logger.debug("Attempting to fetch next task...")
                continue
            except Exception as e:
                if dataset_uuid:
                    err_msg = f"Unexpected error during task generation: {e}\n{traceback.format_exc()}"
                    self.logger.exception(f"❌ {dataset_uuid}: {err_msg}")
                    with self.db.with_session() as session:
                        _mark_upload_failed(session, dataset_uuid, err_msg, self.hub_name, logger=self.logger)
                    self.summary_logger.debug(f"❌ {dataset_uuid}: {err_msg}")
                    self.datasets_failed += 1
                else:
                    self.logger.exception(f"❌ Unexpected error before task claimed: {e}")
                self.logger.debug("Attempting to fetch next task...")
                continue

        return None

    def handle_task_result(self, task_content: dict, task_result_content: dict) -> None:
        """Handle task result from client and update database."""
        dataset_uuid = task_content.get(DATASET_UUID)
        upload_result = task_result_content or {}
        upload_success = upload_result.get("success", False)

        with self.db.with_session() as session:
            item = session.query(DatasetDB).filter(DatasetDB.dataset_uuid == dataset_uuid).first()
            if item is None:
                self.logger.error(f"Dataset {dataset_uuid} not found in dataset DB.")
                return

            # in case of success:
            if upload_success:
                _mark_upload_completed(session, dataset_uuid, self.hub_name, logger=self.logger, item=item)
                self.datasets_succeeded += 1
                self.logger.info(f"Task result: SUCCESS | UUID: {dataset_uuid}")
                self.summary_logger.debug(f"✅ {dataset_uuid}: Upload completed successfully")

                # Log cumulative statistics
                total_datasets = self.datasets_succeeded + self.datasets_failed
                self.summary_logger.debug(
                    f"📊 Cumulative: {total_datasets} datasets "
                    f"({self.datasets_succeeded} ✅, {self.datasets_failed} ❌)"
                )
                self.logger.debug(f"Marked {item.convert_path} upload as COMPLETED")

            # in case of failure:
            else:
                error_message = upload_result.get("error_message") or "Upload failed"
                _mark_upload_failed(session, dataset_uuid, error_message, self.hub_name, logger=self.logger, item=item)
                self.datasets_failed += 1
                self.logger.info(f"Task result: FAILED | UUID: {dataset_uuid} | Error: {error_message}")
                self.summary_logger.debug(f"❌ {dataset_uuid}: {error_message}")

                # Log cumulative statistics
                total_datasets = self.datasets_succeeded + self.datasets_failed
                self.summary_logger.debug(
                    f"📊 Cumulative: {total_datasets} datasets "
                    f"({self.datasets_succeeded} ✅, {self.datasets_failed} ❌)"
                )

                self.logger.debug(f"Marked {item.convert_path} upload as FAILED: {error_message}")

    def get_statistics(self) -> dict:
        return {
            "datasets_succeeded": self.datasets_succeeded,
            "datasets_failed": self.datasets_failed,
        }


__all__ = [
    "HubUploadServer",
    "TASK_CATEGORY",
]
