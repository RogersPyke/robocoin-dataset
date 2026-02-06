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
from .task import (
    _gen_one_upload_task,
    _mark_upload_completed,
    _mark_upload_failed,
    _sync_upload_status,
)
from robocoin_dataset.prepare_metadata.metadata_collect import create_unified_metadata

TASK_CATEGORY = "hub_upload"

class HubUploadServer(TaskServer):
    """Task distribution server for hub upload."""

    def __init__(
        self,
        cfg: UploadConfig,
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

        if not cfg.pg_cfg_path:
            raise ValueError("pg_cfg_path is required to specify database and cannot be None or empty")
        self.cfg.pg_cfg_path: Path = Path(cfg.pg_cfg_path).expanduser().absolute()
        if not self.cfg.pg_cfg_path.exists():
            raise FileNotFoundError(f"Database config file not found: {self.cfg.pg_cfg_path}")

        self.database = DatasetDatabase(self.cfg.pg_cfg_path)
        self.logger = logger or logging.getLogger(__name__)

        self.succeed_cnt = 0
        self.fail_cnt = 0

    def get_task_category(self) -> str:
        return TASK_CATEGORY

    def gen_task_content(self) -> dict | None:
        while True:
            dataset_uuid = None

            # Step 1: Sync and claim task with both huggingface and modelscope
            try:
                with self.database.with_session() as session:
                    _sync_upload_status(session, hub_name="huggingface", logger=self.logger)
                    _sync_upload_status(session, hub_name="modelscope", logger=self.logger)

                    hf_uuid = _gen_one_upload_task(session, hub_name="huggingface", logger=self.logger)
                    ms_uuid = _gen_one_upload_task(session, hub_name="modelscope", logger=self.logger)
                    if hf_uuid is None and ms_uuid is None:
                        self.logger.debug("No PENDING tasks found")
                        break
                    hf_hardlink_path = _get_hardlink_path(session, hf_uuid, logger=self.logger)
                    ms_hardlink_path = _get_hardlink_path(session, ms_uuid, logger=self.logger)
                    if hf_hardlink_path is None and ms_hardlink_path is None:
                        self.logger.debug("No hardlink path found for dataset")
                        break

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
                    with self.database.with_session() as session:
                        _mark_upload_failed(session, dataset_uuid, err_msg, self.hub_name, logger=self.logger)
                    self.summary_logger.debug(f"❌ {dataset_uuid}: {err_msg}")
                    self.fail_cnt += 1
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
                with self.database.with_session() as session:
                    _mark_upload_failed(session, dataset_uuid, err_msg, self.hub_name, logger=self.logger)
                self.summary_logger.debug(f"❌ {dataset_uuid}: {err_msg}")
                self.fail_cnt += 1
                self.logger.debug("Attempting to fetch next task...")
                continue
            except Exception as e:
                if dataset_uuid:
                    err_msg = f"Unexpected error during task generation: {e}\n{traceback.format_exc()}"
                    self.logger.exception(f"❌ {dataset_uuid}: {err_msg}")
                    with self.database.with_session() as session:
                        _mark_upload_failed(session, dataset_uuid, err_msg, self.hub_name, logger=self.logger)
                    self.summary_logger.debug(f"❌ {dataset_uuid}: {err_msg}")
                    self.fail_cnt += 1
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

        with self.database.with_session() as session:
            item = session.query(DatasetDB).filter(DatasetDB.dataset_uuid == dataset_uuid).first()
            if item is None:
                self.logger.error(f"Dataset {dataset_uuid} not found in dataset DB.")
                return

            # in case of success:
            if upload_success:
                _mark_upload_completed(session, dataset_uuid, self.hub_name, logger=self.logger, item=item)
                self.succeed_cnt += 1
                self.logger.info(f"Task result: SUCCESS | UUID: {dataset_uuid}")
                self.summary_logger.debug(f"✅ {dataset_uuid}: Upload completed successfully")

                # Log cumulative statistics
                total_datasets = self.succeed_cnt + self.fail_cnt
                self.summary_logger.debug(
                    f"📊 Cumulative: {total_datasets} datasets "
                    f"({self.succeed_cnt} ✅, {self.fail_cnt} ❌)"
                )
                self.logger.debug(f"Marked {item.convert_path} upload as COMPLETED")

            # in case of failure:
            else:
                error_message = upload_result.get("error_message") or "Upload failed"
                _mark_upload_failed(session, dataset_uuid, error_message, self.hub_name, logger=self.logger, item=item)
                self.fail_cnt += 1
                self.logger.info(f"Task result: FAILED | UUID: {dataset_uuid} | Error: {error_message}")
                self.summary_logger.debug(f"❌ {dataset_uuid}: {error_message}")

                # Log cumulative statistics
                total_datasets = self.succeed_cnt + self.fail_cnt
                self.summary_logger.debug(
                    f"📊 Cumulative: {total_datasets} datasets "
                    f"({self.succeed_cnt} ✅, {self.fail_cnt} ❌)"
                )

                self.logger.debug(f"Marked {item.convert_path} upload as FAILED: {error_message}")

    def get_statistics(self) -> dict:
        return {
            "datasets_succeeded": self.succeed_cnt,
            "datasets_failed": self.fail_cnt,
        }


__all__ = [
    "HubUploadServer",
    "TASK_CATEGORY",
]
