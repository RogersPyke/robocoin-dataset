"""Task management functions for hub upload operations."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Type
from sqlalchemy.orm import Session

# Import DatasetDB for runtime use (not just type checking)
from robocoin_dataset.database.models import DatasetDB

if TYPE_CHECKING:
    pass  # Keep TYPE_CHECKING block for future type hints


def _get_field(dataset_table: Type[DatasetDB], field_suffix: str, hub_name: str) -> str:
    """
    Get the correct field prefix for the hub, trying short version first, then long version.
    
    Args:
        dataset_table: DatasetDB class (not instance)
        field_suffix: Field suffix (e.g., "upload_status")
        hub_name: Hub name ("huggingface" or "modelscope")
        
    Returns:
        str: Full field name (e.g., "huggingface_upload_status")
    """
    hf_prefix = "huggingface" if hasattr(dataset_table, "huggingface_upload_status") else "hf"
    ms_prefix = "ms" if hasattr(dataset_table, "ms_upload_status") else "modelscope"
    hf_field_name = f"{hf_prefix}_{field_suffix}"
    ms_field_name = f"{ms_prefix}_{field_suffix}"
    if hub_name == "huggingface" or hub_name == "hf":
        return hf_field_name
    elif hub_name == "modelscope" or hub_name == "ms":
        return ms_field_name
    else:
        raise ValueError(f"Invalid hub name: {hub_name}, must be 'huggingface' or 'modelscope'")

# Initialize field name constants
HF_STATE = _get_field(DatasetDB, "upload_status", "huggingface")
MS_STATE = _get_field(DatasetDB, "upload_status", "modelscope")
HF_VERSION = _get_field(DatasetDB, "upload_version", "huggingface")
MS_VERSION = _get_field(DatasetDB, "upload_version", "modelscope")
HF_VERSION_PS = _get_field(DatasetDB, "upload_version_ps", "huggingface")
MS_VERSION_PS = _get_field(DatasetDB, "upload_version_ps", "modelscope")
HF_ERR_MSG = _get_field(DatasetDB, "upload_err_msg", "huggingface")
MS_ERR_MSG = _get_field(DatasetDB, "upload_err_msg", "modelscope")
PRE_STAGE_STATUS = "data_loader_detection_status"
PRE_STAGE_VERSION = "data_loader_detection_version"

def _sync_upload_status(
    session: Session,
    specific_uuid: str | None = None,
    hub_name: str = "huggingface",
    logger: logging.Logger | None = None,
) -> None:
    """
    Sync upload status for both HuggingFace and ModelScope hubs in one operation.
    if specific_uuid is provided, only sync the specific dataset.

    NOTE: MUST spcecify the hub_name to sync.
    
    Args:
        session: SQLAlchemy session instance
        specific_uuid: Optional specific dataset UUID to sync. If provided, only syncs that dataset.
        logger: Optional logger instance
        hub_name: The hub name to sync. Must be 'huggingface' or 'modelscope'.
    """
    from sqlalchemy.sql.expression import and_, or_
    from robocoin_dataset.database.models import DatasetDB, TaskStatus
    _logger = logger or logging.getLogger(__name__)

    # ===== HuggingFace Branch: Independent query and update =====
    if hub_name == "huggingface" or hub_name == "hf":
        if specific_uuid:
            hf_query = session.query(DatasetDB).filter(DatasetDB.dataset_uuid == specific_uuid)
        else:
            hf_query = session.query(DatasetDB).filter(
                and_(
                    getattr(DatasetDB, PRE_STAGE_STATUS) == TaskStatus.COMPLETED,
                    or_(
                        getattr(DatasetDB, HF_STATE) == TaskStatus.PENDING,
                        and_(
                            getattr(DatasetDB, HF_STATE) == TaskStatus.COMPLETED,
                            getattr(DatasetDB, HF_VERSION_PS) < getattr(DatasetDB, HF_VERSION),
                        ),
                    ),
                )
            )
        hf_items = hf_query.all()
        hf_synced_count = 0
        if hf_items:
            _logger.debug(f"Found {len(hf_items)} datasets to sync for HuggingFace hub")
            for item in hf_items:
                pre_stage_version_value = getattr(item, PRE_STAGE_VERSION, 0) or 0
                setattr(item, HF_STATE, TaskStatus.PENDING)
                setattr(item, HF_VERSION_PS, pre_stage_version_value)
                hf_synced_count += 1
        session.commit()
        if hf_items:
            _logger.debug(f"Synced HuggingFace hub for {hf_synced_count} dataset(s)")
        else:
            _logger.debug("No datasets to sync for HuggingFace hub")
            
    # ===== ModelScope Branch: Independent query and update =====
    elif hub_name == "modelscope" or hub_name == "ms":
        if specific_uuid:
            ms_query = session.query(DatasetDB).filter(DatasetDB.dataset_uuid == specific_uuid)
        else:
            ms_query = session.query(DatasetDB).filter(
                and_(
                    getattr(DatasetDB, PRE_STAGE_STATUS) == TaskStatus.COMPLETED,
                    or_(
                        getattr(DatasetDB, MS_STATE) == TaskStatus.PENDING,
                        and_(
                            getattr(DatasetDB, MS_STATE) == TaskStatus.COMPLETED,
                            getattr(DatasetDB, MS_VERSION_PS) < getattr(DatasetDB, MS_VERSION),
                        ),
                    ),
                )
            )
        ms_items = ms_query.all()
        ms_synced_count = 0
        if ms_items:
            _logger.debug(f"Found {len(ms_items)} datasets to sync for ModelScope hub")
            for item in ms_items:
                pre_stage_version_value = getattr(item, PRE_STAGE_VERSION, 0) or 0
                setattr(item, MS_STATE, TaskStatus.PENDING)
                setattr(item, MS_VERSION_PS, pre_stage_version_value)
                ms_synced_count += 1
        session.commit()
        if ms_items:
            _logger.debug(f"Synced ModelScope hub for {ms_synced_count} dataset(s)")
        else:
            _logger.debug("No datasets to sync for ModelScope hub")

def _gen_one_upload_task(
    session: Session,
    specific_uuid: str | None = None,
    hub_name: str = "huggingface",
    logger: logging.Logger | None = None,
) -> str | None:
    """
    Generate one dataset upload task by finding a pending upload, marking it as PROCESSING,
    ONLY return the dataset_uuid, and resolve the hardlink path SHOULD be done by following code.

    NOTE: MUST spcecify the hub_name to sync.
    """
    from sqlalchemy.sql.expression import and_
    from robocoin_dataset.database.models import DatasetDB, DatasetHardLinkDB, TaskStatus
    _logger = logger or logging.getLogger(__name__)

    # ===== HuggingFace Branch: Independent query and update =====
    if hub_name == "huggingface" or hub_name == "hf":
        if specific_uuid:
            query = session.query(DatasetDB).filter(DatasetDB.dataset_uuid == specific_uuid)
        else:
            query = session.query(DatasetDB).filter(
                and_(
                    getattr(DatasetDB, PRE_STAGE_STATUS) == TaskStatus.COMPLETED,
                    getattr(DatasetDB, HF_STATE) == TaskStatus.PENDING,
                )
            )
        hf_item = query.first()
        if hf_item is None:
            return None
        setattr(hf_item, HF_STATE, TaskStatus.PROCESSING)
        session.commit()
        return hf_item.dataset_uuid
    # ===== ModelScope Branch: Independent query and update =====
    elif hub_name == "modelscope" or hub_name == "ms":
        if specific_uuid:
            query = session.query(DatasetDB).filter(DatasetDB.dataset_uuid == specific_uuid)
        else:
            query = session.query(DatasetDB).filter(
                and_(
                    getattr(DatasetDB, PRE_STAGE_STATUS) == TaskStatus.COMPLETED,
                    getattr(DatasetDB, MS_STATE) == TaskStatus.PENDING,
                )
            )
        ms_item = query.first()
        if ms_item is None:
            return None
        setattr(ms_item, MS_STATE, TaskStatus.PROCESSING)
        session.commit()
        return ms_item.dataset_uuid

def _mark_upload_failed(
    session: Session,
    dataset_uuid: str,
    error_msg: str,
    hub_name: str,
    logger: logging.Logger | None = None,
) -> None:
    """
    Mark upload task as failed with error message.
    Update the corresponding status field based on hub_name.

    Args:
        session: SQLAlchemy session instance
        dataset_uuid: Dataset UUID to mark as failed
        error_msg: Error message describing the failure
        hub_name: Hub name ("huggingface" or "modelscope"), must be specified
        logger: Optional logger instance

    Raises:
        ValueError: If hub_name is invalid
    """
    from robocoin_dataset.database.models import DatasetDB, TaskStatus
    _logger = logger or logging.getLogger(__name__)

    item = session.query(DatasetDB).filter(DatasetDB.dataset_uuid == dataset_uuid).first()
    if not item:
        _logger.warning(f"Dataset {dataset_uuid} not found in database")
        return

    # Get the correct field names based on hub_name
    state_field = _get_field(DatasetDB, "upload_status", hub_name)
    err_field = _get_field(DatasetDB, "upload_err_msg", hub_name)

    # Update status and error message
    setattr(item, state_field, TaskStatus.FAILED)
    setattr(item, err_field, error_msg)
    session.commit()
    _logger.debug(f"Marked dataset {dataset_uuid} ({hub_name}) as FAILED: {error_msg}")


def _mark_upload_completed(
    session: Session,
    dataset_uuid: str,
    hub_name: str,
    logger: logging.Logger | None = None,
) -> None:
    """
    Mark upload task as completed.
    Update the corresponding status field based on hub_name.

    Args:
        session: SQLAlchemy session instance
        dataset_uuid: Dataset UUID to mark as completed
        hub_name: Hub name ("huggingface" or "modelscope"), must be specified
        logger: Optional logger instance

    Raises:
        ValueError: If hub_name is invalid
    """
    from robocoin_dataset.database.models import DatasetDB, TaskStatus
    _logger = logger or logging.getLogger(__name__)

    item = session.query(DatasetDB).filter(DatasetDB.dataset_uuid == dataset_uuid).first()
    if not item:
        _logger.warning(f"Dataset {dataset_uuid} not found in database")
        return

    # Get the correct field name based on hub_name
    state_field = _get_field(DatasetDB, "upload_status", hub_name)

    # Update status
    setattr(item, state_field, TaskStatus.COMPLETED)
    session.commit()
    _logger.debug(f"Marked dataset {dataset_uuid} ({hub_name}) as COMPLETED")

def _get_hardlink_path_by_uuid(
    session: Session,
    dataset_uuid: str,
    logger: logging.Logger | None = None,
) -> Path | None:
    """
    Get the hardlink path for a dataset.
    """
    from robocoin_dataset.database.models import DatasetHardLinkDB
    _logger = logger or logging.getLogger(__name__)
    item = session.query(DatasetHardLinkDB).filter(DatasetHardLinkDB.dataset_uuid == dataset_uuid).first()
    if item is None or not item.hard_link_path:
        return None
    return Path(item.hard_link_path).expanduser()