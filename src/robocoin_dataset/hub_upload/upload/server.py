"""Server component for distributed hub upload tasks.

This module provides the server-side orchestration for:
- Task distribution to multiple clients for a SINGLE hub platform
- Each server instance handles ONLY ONE hub (HuggingFace OR ModelScope)
- Hardlink validation before task assignment
- Database updates based on client results

Architecture Design:
- One server instance per hub platform (dedicated server approach)
- To process both hubs, run TWO separate server instances with different hub_name
- This design ensures complete isolation and better scalability

The server follows the principle of "do one thing and do it well":
- Server only handles task distribution and status management
- Metadata collection and file generation are handled by other components
- Upload execution is handled by clients
"""

import logging
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

from robocoin_dataset.database.database import DatasetDatabase
from robocoin_dataset.database.models import DatasetDB
from robocoin_dataset.distribution_computation.constant import DATASET_UUID
from robocoin_dataset.distribution_computation.task_server import TaskServer
from robocoin_dataset.format_converter.tolerobot.constant import LEFORMAT_PATH
from .task import (
    _gen_one_upload_task,
    _get_hardlink_path_by_uuid,
    _mark_upload_completed,
    _mark_upload_failed,
    _sync_upload_status,
)

if TYPE_CHECKING:
    from .utils import UploadConfig

TASK_CATEGORY = "hub_upload"

class UploadServer(TaskServer):
    """Task distribution server for hub upload.
    
    This server handles tasks for a SINGLE hub platform (either HuggingFace or ModelScope).
    To process both hubs, run TWO separate server instances with different hub_name.
    """

    def __init__(
        self,
        cfg: "UploadConfig",
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize hub upload server for a SPECIFIC hub platform.

        Args:
            cfg: UploadConfig containing all configuration parameters including:
                - Database path (pg_cfg_path)
                - HuggingFace configuration (hf_token, hf_namespace)
                - ModelScope configuration (ms_token, ms_namespace)
                - Common configuration (force_overwrite, readme_only)
                - Server network configuration (host, port, heartbeat_interval, timeout)
                - hub_name: Hub platform this server will handle ("huggingface" or "modelscope")
            logger: Optional logger instance

        Raises:
            ValueError: If pg_cfg_path is not provided, empty, or hub_name is invalid
            FileNotFoundError: If database config file does not exist
        """
        super().__init__(
            logger=logger,
            host=cfg.server_host,
            port=cfg.server_port,
            heartbeat_interval=cfg.server_heartbeat_interval,
            timeout=cfg.server_timeout,
        )

        self.cfg = cfg

        # Validate hub_name
        if self.cfg.hub_name not in ("huggingface", "modelscope", "hf", "ms"):
            raise ValueError(
                f"Invalid hub_name: {self.cfg.hub_name}. Must be 'huggingface', 'modelscope', 'hf', or 'ms'"
            )
        
        # Store hub_name as instance attribute for easy access
        self.cfg.hub_name = self.cfg.hub_name

        # Validate and initialize database connection
        if not self.cfg.pg_cfg_path:
            raise ValueError("pg_cfg_path is required to specify database and cannot be None or empty")
        self.cfg_pg_path: Path = Path(self.cfg.pg_cfg_path).expanduser().absolute()
        if not self.cfg_pg_path.exists():
            raise FileNotFoundError(f"Database config file not found: {self.cfg_pg_path}")

        self.database = DatasetDatabase(self.cfg_pg_path)
        self.logger = logger or logging.getLogger(__name__)

        # Store common configuration
        self.common_config = {
            "force_overwrite": cfg.force_overwrite,
            "readme_only": cfg.readme_only,
        }

        # ===== Store hub-specific configuration for THIS server's hub =====
        if self.cfg.hub_name == "huggingface" or self.cfg.hub_name == "hf":
            self.hub_config = {
                "token": cfg.hf_token,
                "namespace": cfg.hf_namespace,
                **self.common_config,
            }
        elif self.cfg.hub_name == "modelscope" or self.cfg.hub_name == "ms":
            self.hub_config = {
                "token": cfg.ms_token,
                "namespace": cfg.ms_namespace,
                **self.common_config,
            }

        # Statistics tracking
        self.succeed_cnt = 0
        self.fail_cnt = 0
        
        # Log server initialization
        self.logger.info(f"[SERVER] Initialized for hub: {self.cfg.hub_name}")
        self.logger.info(f"[SERVER] Namespace: {self.hub_config['namespace']}")
        self.logger.info(f"[SERVER] Host: {cfg.server_host}:{cfg.server_port}")

    def get_task_category(self) -> str:
        return TASK_CATEGORY

    def generate_task_content(self) -> dict | None:
        """
        Generate task content for THIS server's hub platform.
        named as generate_task_content to avoid conflict with super class.
        
        This method handles tasks for a single hub (no round-robin):
        1. Syncs upload status for this hub
        2. Tries to get one task for this hub
        3. Returns task config or None if no tasks available
        
        The server only distributes tasks and does NOT collect metadata or generate files.
        Metadata collection and file generation are handled by other components.

        Returns:
            dict | None: Task content dictionary containing dataset_uuid, hardlink_path,
                        hub_name, and client_config. Returns None if no tasks available.
        """
        while True:
            try:
                with self.database.with_session() as session:
                    # Step 1: Sync status for THIS hub only
                    try:
                        _sync_upload_status(session, hub_name=self.cfg.hub_name, logger=self.logger)
                    except Exception as e:
                        # Sync errors should not block task generation
                        err_msg = f"Status sync error: {e}\n{traceback.format_exc()}"
                        self.logger.error(f"[ERROR] {err_msg}")
                        # Continue to try task generation even if sync fails

                    # Step 2: Try to get one task for THIS hub
                    dataset_uuid = None
                    try:
                        dataset_uuid = _gen_one_upload_task(session, hub_name=self.cfg.hub_name, logger=self.logger)
                        if dataset_uuid:
                            # UUID obtained, now build task config (may raise errors)
                            return self._build_task_config(dataset_uuid, self.cfg.hub_name, session)
                    except FileNotFoundError as e:
                        # Hardlink validation failed in _build_task_config
                        # UUID is available, mark as failed and continue
                        if dataset_uuid:
                            err_msg = f"Hardlink path error: {e}"
                            self.logger.error(f"[ERROR] Dataset {dataset_uuid} ({self.cfg.hub_name}): {err_msg}")
                            _mark_upload_failed(session, dataset_uuid, err_msg, self.cfg.hub_name, logger=self.logger)
                        else:
                            # Unexpected: FileNotFoundError before UUID obtained
                            self.logger.error(f"[ERROR] Unexpected FileNotFoundError during {self.cfg.hub_name} task generation: {e}")
                        self.logger.debug("[TASK] Attempting to fetch next task...")
                        continue
                    except ValueError as e:
                        # Invalid configuration or hub_name
                        if dataset_uuid:
                            err_msg = f"Configuration error: {e}"
                            self.logger.error(f"[ERROR] Dataset {dataset_uuid} ({self.cfg.hub_name}): {err_msg}")
                            _mark_upload_failed(session, dataset_uuid, err_msg, self.cfg.hub_name, logger=self.logger)
                        else:
                            self.logger.error(f"[ERROR] Configuration error during {self.cfg.hub_name} task generation: {e}")
                        self.logger.debug("[TASK] Attempting to fetch next task...")
                        continue
                    except Exception as e:
                        # Other unexpected errors during task processing
                        err_msg = f"Unexpected error: {e}\n{traceback.format_exc()}"
                        if dataset_uuid:
                            self.logger.exception(f"[ERROR] Dataset {dataset_uuid} ({self.cfg.hub_name}): {err_msg}")
                            _mark_upload_failed(session, dataset_uuid, err_msg, self.cfg.hub_name, logger=self.logger)
                        else:
                            self.logger.exception(f"[ERROR] {self.cfg.hub_name} task generation error: {err_msg}")
                        self.logger.debug("[TASK] Attempting to fetch next task...")
                        continue

                    # Step 3: No tasks available for this hub
                    self.logger.debug(f"[TASK] No PENDING tasks found for {self.cfg.hub_name}")
                    break

            except Exception as e:
                # Catch-all for session-level errors (database connection, etc.)
                err_msg = f"Database session error: {e}\n{traceback.format_exc()}"
                self.logger.exception(f"[ERROR] {err_msg}")
                self.logger.debug("[TASK] Attempting to fetch next task...")
                continue

        return None

    def _build_task_config(
        self,
        dataset_uuid: str,
        hub_name: str,
        session,
    ) -> dict:
        """
        Build task configuration for a specific dataset.
        
        This method validates the hardlink path and constructs the task config
        without collecting metadata (metadata is handled by other components).

        Args:
            dataset_uuid: Dataset UUID
            hub_name: Hub name (should match self.cfg.hub_name)
            session: Database session instance

        Returns:
            dict: Task configuration dictionary containing:
                - dataset_uuid: Dataset UUID
                - leformat_path: Hardlink path to the dataset
                - hub_name: Hub name identifier
                - client_config: Client configuration (token, namespace, etc.)

        Raises:
            FileNotFoundError: If hardlink path is not found or does not exist
            ValueError: If hub_name doesn't match server's hub
        """
        # Validate hub_name matches server's hub
        if hub_name != self.cfg.hub_name:
            raise ValueError(
                f"Hub name mismatch: task hub_name={hub_name}, server hub_name={self.cfg.hub_name}"
            )

        # Get hardlink path from database
        hardlink_path = _get_hardlink_path_by_uuid(session, dataset_uuid, logger=self.logger)
        if hardlink_path is None:
            raise FileNotFoundError(f"Hardlink path not found for dataset {dataset_uuid}")
        if not hardlink_path.exists():
            raise FileNotFoundError(f"Hardlink path does not exist: {hardlink_path}")

        # Build task configuration (NO metadata - handled by other components)
        # Pass the entire hub_config to avoid hardcoding field names
        # Add hub_name to the config since client needs it
        client_config = {**self.hub_config, "hub_name": hub_name}
        
        task_config = {
            "dataset_uuid": dataset_uuid,
            "leformat_path": str(hardlink_path),
            "hub_name": hub_name,  # Explicitly identify which hub this task is for
            "client_config": client_config,
            "token": self.hub_config["token"],
        }
        self.logger.debug(f"[TASK] Sending task for dataset {dataset_uuid} to {hub_name}")
        return task_config

    def handle_task_result(self, task_result_content: dict) -> None:
        """
        Handle task result from client and update database status.
        
        This method updates the database status for THIS server's hub.
        The hub_name should match self.cfg.hub_name.

        Args:
            task_content: Original task content dictionary containing dataset_uuid and hub_name
            task_result_content: Result dictionary from client containing success status and error message
        """
        dataset_uuid = task_result_content.get("dataset_uuid")
        hub_name = task_result_content.get("hub_name", self.cfg.hub_name)
        upload_success = task_result_content.get("success", False)

        if not dataset_uuid:
            self.logger.error("[ERROR] Task result missing dataset_uuid")
            return

        # Validate hub_name matches server's hub
        if hub_name != self.cfg.hub_name:
            self.logger.warning(
                f"[WARNING] Hub name mismatch in result: task hub={hub_name}, server hub={self.cfg.hub_name}"
            )

        with self.database.with_session() as session:
            item = session.query(DatasetDB).filter(DatasetDB.dataset_uuid == dataset_uuid).first()
            if item is None:
                self.logger.error(f"[ERROR] Dataset {dataset_uuid} not found in database")
                return

            # Update status based on result
            if upload_success:
                _mark_upload_completed(session, dataset_uuid, hub_name, logger=self.logger)
                self.succeed_cnt += 1
                self.logger.info(f"[SUCCESS] Task completed | UUID: {dataset_uuid} | Hub: {hub_name}")

                # Log cumulative statistics
                total_datasets = self.succeed_cnt + self.fail_cnt
                self.logger.debug(
                    f"[STATS] Cumulative: {total_datasets} datasets "
                    f"({self.succeed_cnt} succeeded, {self.fail_cnt} failed)"
                )
            else:
                error_message = task_result_content.get("error_message") or "Upload failed"
                _mark_upload_failed(session, dataset_uuid, error_message, hub_name, logger=self.logger)
                self.fail_cnt += 1
                self.logger.info(f"[FAILED] Task failed | UUID: {dataset_uuid} | Hub: {hub_name} | Error: {error_message}")

                # Log cumulative statistics
                total_datasets = self.succeed_cnt + self.fail_cnt
                self.logger.debug(
                    f"[STATS] Cumulative: {total_datasets} datasets "
                    f"({self.succeed_cnt} succeeded, {self.fail_cnt} failed)"
                )

    def get_statistics(self) -> dict:
        return {
            "datasets_succeeded": self.succeed_cnt,
            "datasets_failed": self.fail_cnt,
        }


__all__ = [
    "UploadServer",
    "TASK_CATEGORY",
]
