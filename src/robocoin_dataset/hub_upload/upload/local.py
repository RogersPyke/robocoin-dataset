"""
Local Upload Module for Hub Upload Operations

This module provides the UploadLocal class for local upload operations.
It handles parameter passing and database task status updates for dataset uploads.

The main responsibility of this module is to coordinate between the upload utility
and the database task management, ensuring proper status synchronization.
"""

import logging
from pathlib import Path

from sqlalchemy.orm import Session

from .task import (
    _gen_one_upload_task,
    _mark_upload_completed,
    _mark_upload_failed,
    _sync_upload_status,
)
from .utils import UploadConfig, UploadUtil


class UploadLocal(UploadUtil):
    """
    Upload utility for local upload operations.
    
    This class extends UploadUtil to provide local upload functionality with
    database task status management. It coordinates between the core upload
    logic and the PostgreSQL database to track upload progress and status.
    
    The class handles:
    - Synchronizing upload status in the database
    - Generating upload tasks
    - Executing the actual upload via parent class
    - Updating task status (completed/failed) in the database
    
    Attributes:
        config (UploadConfig): Configuration object containing hub settings,
            authentication tokens, and upload parameters.
        logger (logging.Logger): Logger instance for audit logging.
    
    Expected Input:
        - config: UploadConfig instance with valid hub_name, tokens, and namespaces
        - hardlink_path: Path object pointing to the dataset hardlink directory
        - dataset_uuid: String UUID of the dataset to upload
        - hub_name: String hub name ("huggingface" or "modelscope")
        - pg_session: SQLAlchemy Session instance for database operations
    
    Expected Output:
        - None: Method returns None, but updates database status on success/failure
        - Raises Exception: If any step fails (sync, task generation, or upload)
    
    Expected Usage:
        This class is used in local upload mode where the upload operation
        runs directly on the local machine with direct database access.
    """

    def __init__(self, config: UploadConfig) -> None:
        """
        Initialize the local uploader with configuration.
        
        Args:
            config: UploadConfig containing hub_name, authentication tokens,
                and other upload parameters.
        
        Raises:
            ValueError: If hub_name in config is not supported.
        """
        super().__init__(config)
        self.config = config
        # Parent class UploadUtil already initializes self.logger in __init__
        # The logger is already configured and ready to use
        # We keep using the parent's logger for consistency

    def upload(
        self,
        hardlink_path: Path,
        dataset_uuid: str,
        hub_name: str,
        pg_session: Session,
    ) -> None:
        """
        Upload a single dataset to the remote hub and update PostgreSQL database.
        
        This method orchestrates the complete upload process:
        1. Sync upload status in database (mark as PENDING if needed)
        2. Generate upload task (mark as PROCESSING)
        3. Execute actual upload via parent class UploadUtil.upload()
        4. Update database status (COMPLETED on success, FAILED on error)
        
        Args:
            hardlink_path: Path to the dataset hardlink directory to upload.
                Must exist and be a valid directory.
            dataset_uuid: UUID string identifying the dataset in the database.
            hub_name: Hub platform name, must be "huggingface" (or "hf")
                or "modelscope" (or "ms").
            pg_session: SQLAlchemy Session instance for database operations.
                Must be an active session with proper database connection.
        
        Returns:
            None: Method returns None on success.
        
        Raises:
            Exception: Re-raises any exception encountered during:
                - Status synchronization
                - Task generation
                - Upload execution
            The database status will be updated to FAILED before re-raising.
        
        Expected Behavior:
            - On success: Database status updated to COMPLETED
            - On failure: Database status updated to FAILED with error message,
              then exception is re-raised
        """
        # Step 1: Sync upload status in database
        # This marks the dataset as PENDING if pre-stage is completed
        try:
            self.logger.debug(
                f"[UploadLocal.upload] Syncing upload status for dataset {dataset_uuid} (hub: {hub_name})"
            )
            _sync_upload_status(
                session=pg_session,
                specific_uuid=dataset_uuid,
                hub_name=hub_name,
                logger=self.logger,
            )
        except Exception as e:
            error_msg = f"[UploadLocal.upload] Failed to sync upload status: {e}"
            self.logger.error(error_msg)
            raise

        # Step 2: Generate upload task in database
        # This marks the dataset as PROCESSING
        try:
            self.logger.debug(
                f"[UploadLocal.upload] Generating upload task for dataset {dataset_uuid} (hub: {hub_name})"
            )
            task_uuid = _gen_one_upload_task(
                session=pg_session,
                specific_uuid=dataset_uuid,
                hub_name=hub_name,
                logger=self.logger,
            )
            if task_uuid is None:
                error_msg = f"[UploadLocal.upload] Failed to generate upload task: No task found for dataset {dataset_uuid}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"[UploadLocal.upload] Failed to generate upload task: {e}"
            self.logger.error(error_msg)
            raise

        # Step 3: Execute actual upload via parent class
        # Parent class handles retry logic and hub API calls
        try:
            self.logger.info(
                f"[UploadLocal.upload] Starting upload for dataset {dataset_uuid} from path {hardlink_path} (hub: {hub_name})"
            )
            upload_result = super().upload(hardlink_path=hardlink_path)

            # Check upload result
            # Parent class now always returns (bool, str) tuple: (success, error_msg)
            # For backward compatibility, also handle the old format where success was just True
            if isinstance(upload_result, tuple) and len(upload_result) == 2:
                success, error_msg = upload_result
                if success:
                    # Upload succeeded
                    self.logger.info(
                        f"[UploadLocal.upload] Upload succeeded for dataset {dataset_uuid} (hub: {hub_name})"
                    )
                    _mark_upload_completed(
                        session=pg_session,
                        dataset_uuid=dataset_uuid,
                        hub_name=hub_name,
                        logger=self.logger,
                    )
                else:
                    # Upload failed with error message
                    self.logger.error(
                        f"[UploadLocal.upload] Upload failed for dataset {dataset_uuid}: {error_msg}"
                    )
                    _mark_upload_failed(
                        session=pg_session,
                        dataset_uuid=dataset_uuid,
                        error_msg=error_msg,
                        hub_name=hub_name,
                        logger=self.logger,
                    )
            elif upload_result is True:
                # Backward compatibility: handle old format where success was just True
                self.logger.info(
                    f"[UploadLocal.upload] Upload succeeded for dataset {dataset_uuid} (hub: {hub_name})"
                )
                _mark_upload_completed(
                    session=pg_session,
                    dataset_uuid=dataset_uuid,
                    hub_name=hub_name,
                    logger=self.logger,
                )
            else:
                # Unexpected: unknown result format
                error_msg = f"[UploadLocal.upload] Unexpected upload result format: {upload_result}"
                self.logger.error(error_msg)
                _mark_upload_failed(
                    session=pg_session,
                    dataset_uuid=dataset_uuid,
                    error_msg=error_msg,
                    hub_name=hub_name,
                    logger=self.logger,
                )

        except Exception as e:
            # Upload failed with exception
            error_msg = str(e)
            self.logger.error(
                f"[UploadLocal.upload] Upload exception for dataset {dataset_uuid}: {error_msg}"
            )
            _mark_upload_failed(
                session=pg_session,
                dataset_uuid=dataset_uuid,
                error_msg=error_msg,
                hub_name=hub_name,
                logger=self.logger,
            )
            raise
