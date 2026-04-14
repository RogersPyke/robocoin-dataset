import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from robocoin_dataset.database.models import DatasetDB


######## TASK MANAGEMENT ########


def _sync_page_sync_status(
    session: "Session",
    logger: logging.Logger | None = None,
    force_regenerate: bool = False,
) -> None:
  '''Sync: mark PENDING if the record need to be page_synced
  I: Database. O: None, change PENDING directly.(version ++ and sync also)
  '''

  from sqlalchemy.sql.expression import and_, or_

  from robocoin_dataset.database.models import DatasetDB, TaskStatus

  hf_prefix, ms_prefix = _get_hub_field_prefix(DatasetDB)
  _logger = logger or logging.getLogger(__name__)

  # Build dynamic field names
  ms_upload_status_field = f"{ms_prefix}_upload_status"
  hf_upload_status_field = f"{hf_prefix}_upload_status"
  ms_upload_version_field = f"{ms_prefix}_upload_version"
  hf_upload_version_field = f"{hf_prefix}_upload_version"

  # Query datasets that need page sync
  status_condition = or_(
      DatasetDB.dataset_info_sync_status == TaskStatus.PENDING,
      and_(
          DatasetDB.dataset_info_sync_status == TaskStatus.COMPLETED,
          DatasetDB.dataset_info_sync_version_ps_ms < getattr(DatasetDB, ms_upload_version_field),
          DatasetDB.dataset_info_sync_version_ps_hf < getattr(DatasetDB, hf_upload_version_field),
      ),
  )

  if force_regenerate:
      status_condition = or_(
          DatasetDB.dataset_info_sync_status != TaskStatus.PROCESSING,
      )

  query = session.query(DatasetDB).filter(
    and_(
        getattr(DatasetDB, ms_upload_status_field) == TaskStatus.COMPLETED,
        getattr(DatasetDB, hf_upload_status_field) == TaskStatus.COMPLETED,
        status_condition,
    ),
  )

  items = query.all()
  if not items:
    _logger.debug("No datasets found for page sync")
    return

  _logger.info(f"Found {len(items)} datasets to sync page info")

  # Update status to PENDING for found items
  for item in items:
      item.dataset_info_sync_status = TaskStatus.PENDING

      _logger.debug(f"Marked dataset {item.dataset_uuid} as PENDING for page sync")

  session.commit()
  _logger.info(f"Successfully marked {len(items)} datasets as PENDING")


def _gen_one_page_sync_task(session: "Session"
) -> tuple[str | None, str | None, str | None]:
    '''Mark first PENDING -> PROCESSING, and return the info_yaml_path, hardlink_path, and dataset_uuid
    I: Database session.
    O: info_yaml_path, -> read the metadata.
    hardlink_path, -> the dataset in lerobot foramt.
    dataset_uuid -> to identify which record should be COMPLETED or FAILED.
    '''
    from sqlalchemy.sql.expression import and_

    from robocoin_dataset.database.models import DatasetDB, DatasetHardLinkDB, TaskStatus

    _logger = logging.getLogger(__name__)

    hf_prefix, ms_prefix = _get_hub_field_prefix(DatasetDB)
    ms_upload_status_field = f"{ms_prefix}_upload_status"
    hf_upload_status_field = f"{hf_prefix}_upload_status"
    ms_upload_version_field = f"{ms_prefix}_upload_version"
    hf_upload_version_field = f"{hf_prefix}_upload_version"

    _logger.debug("Querying for PENDING tasks...")
    query = session.query(DatasetDB).filter(
        and_(
            DatasetDB.dataset_info_sync_status == TaskStatus.PENDING,
            getattr(DatasetDB, ms_upload_status_field) == TaskStatus.COMPLETED,
            getattr(DatasetDB, hf_upload_status_field) == TaskStatus.COMPLETED,
        )
    )

    item = query.first()
    if not item:
        _logger.debug("No PENDING tasks found")
        return None, None, None

    _logger.debug("Found PENDING task, marking as PROCESSING...")
    item.dataset_info_sync_status = TaskStatus.PROCESSING

    ms_version = getattr(item, ms_upload_version_field) if hasattr(item, ms_upload_version_field) else 0
    hf_version = getattr(item, hf_upload_version_field) if hasattr(item, hf_upload_version_field) else 0
    item.dataset_info_sync_version_ps_hf = hf_version
    item.dataset_info_sync_version_ps_ms = ms_version

    # Increment dataset_info_sync_version
    current_version = item.dataset_info_sync_version if hasattr(item, 'dataset_info_sync_version') and item.dataset_info_sync_version else 0
    item.dataset_info_sync_version = current_version + 1
    session.commit()

    # Get dataset_uuid
    dataset_uuid = item.dataset_uuid if hasattr(item, 'dataset_uuid') and item.dataset_uuid else None
    _logger.debug(f"Dataset UUID: {dataset_uuid}")

    # Get hardlink path from dataset_hard_link table using dataset_uuid
    try:
        _logger.debug(f"Querying hardlink path for dataset_uuid: {dataset_uuid}")
        hardlink_record = session.query(DatasetHardLinkDB).filter(
            DatasetHardLinkDB.dataset_uuid == dataset_uuid
        ).first()
        hardlink_path = Path(hardlink_record.hard_link_path) if hardlink_record and hardlink_record.hard_link_path else None

        if hardlink_path is None:
            raise FileNotFoundError(
                f"No hardlink found in database for dataset {dataset_uuid}. "
                f"Hardlinks must be created before running page sync."
            )

        # Verify hardlink path exists on disk
        if not hardlink_path.exists():
            raise FileNotFoundError(
                f"Hardlink path in database does not exist on disk: {hardlink_path}"
            )
    except FileNotFoundError:
        # Re-raise FileNotFoundError as-is
        raise
    except Exception as e:
        # Wrap other exceptions as FileNotFoundError
        raise FileNotFoundError(
            f"Failed to retrieve or validate hardlink for dataset {dataset_uuid}: {e}"
        ) from e

    # Page sync only carries paths; info.yaml generation/check is handled in page_sync.py.
    info_yaml_path = hardlink_path / "info.yaml"
    return str(info_yaml_path), str(hardlink_path), dataset_uuid


def _mark_task_processing(session: "Session", dataset_uuid: str) -> None:
    """
    Mark the specific task as PROCESSING and set version fields (same as old task pickup).
    Called when starting to process one dataset in the page sync loop.
    """
    from robocoin_dataset.database.models import DatasetDB, TaskStatus

    hf_prefix, ms_prefix = _get_hub_field_prefix(DatasetDB)
    ms_upload_version_field = f"{ms_prefix}_upload_version"
    hf_upload_version_field = f"{hf_prefix}_upload_version"

    query = session.query(DatasetDB).filter(
        DatasetDB.dataset_uuid == dataset_uuid
    )
    item = query.first()

    if item:
        item.dataset_info_sync_status = TaskStatus.PROCESSING
        ms_version = getattr(item, ms_upload_version_field, None) or 0
        hf_version = getattr(item, hf_upload_version_field, None) or 0
        item.dataset_info_sync_version_ps_ms = ms_version
        item.dataset_info_sync_version_ps_hf = hf_version
        current_version = getattr(item, "dataset_info_sync_version", None) or 0
        item.dataset_info_sync_version = current_version + 1
        session.commit()
        logging.getLogger(__name__).debug("Marked dataset %s as PROCESSING", dataset_uuid)


def _mark_task_completed(session: "Session", dataset_uuid: str) -> None:
    """
    Mark the specific task as COMPLETED using dataset_uuid.
    """
    from robocoin_dataset.database.models import DatasetDB, TaskStatus

    query = session.query(DatasetDB).filter(
        DatasetDB.dataset_uuid == dataset_uuid
    )
    item = query.first()

    if item:
        item.dataset_info_sync_status = TaskStatus.COMPLETED
        session.commit()
        logging.getLogger(__name__).info(f"Marked dataset {dataset_uuid} as COMPLETED")


def _mark_task_failed(session: "Session", dataset_uuid: str, error_msg: str = "") -> None:
    """
    Mark the specific task as FAILED using dataset_uuid and store error message.

    Args:
        session: Database session
        dataset_uuid: UUID of the dataset
        error_msg: Error message to store in database
    """
    from robocoin_dataset.database.models import DatasetDB, TaskStatus

    query = session.query(DatasetDB).filter(
        DatasetDB.dataset_uuid == dataset_uuid
    )
    item = query.first()
    _log = logging.getLogger(__name__)

    if item:
        item.dataset_info_sync_status = TaskStatus.FAILED
        item.dataset_info_sync_err_msg = error_msg if error_msg else None
        session.commit()
        _log.error("Marked dataset %s as FAILED: %s", dataset_uuid, error_msg)
    else:
        _log.error(
            "Cannot mark dataset_info_sync_status FAILED: no row for dataset_uuid=%s. "
            "Database may still show PROCESSING for this UUID.",
            dataset_uuid,
        )


def get_pending_page_sync_entries(
    session: "Session",
    logger: logging.Logger | None = None,
    force_regenerate: bool = False,
) -> list[tuple[str, str]]:
    """
    Return list of (hardlink_path, dataset_uuid) for all datasets that are eligible
    for page sync (PENDING, with hub uploads completed). Does not change any status.

    Input:
        session: Database session.
        logger: Optional logger.
        force_regenerate: If True, same as in _sync_page_sync_status (mark eligible as PENDING).

    Output:
        list of (hardlink_path, dataset_uuid). Hardlink paths are validated to exist on disk.
    """
    from sqlalchemy.sql.expression import and_

    from robocoin_dataset.database.models import DatasetDB, DatasetHardLinkDB, TaskStatus

    _sync_page_sync_status(session, logger=logger, force_regenerate=force_regenerate)

    _logger = logger or logging.getLogger(__name__)
    hf_prefix, ms_prefix = _get_hub_field_prefix(DatasetDB)
    ms_upload_status_field = f"{ms_prefix}_upload_status"
    hf_upload_status_field = f"{hf_prefix}_upload_status"

    query = session.query(DatasetDB).filter(
        and_(
            DatasetDB.dataset_info_sync_status == TaskStatus.PENDING,
            getattr(DatasetDB, ms_upload_status_field) == TaskStatus.COMPLETED,
            getattr(DatasetDB, hf_upload_status_field) == TaskStatus.COMPLETED,
        )
    )
    items = query.all()
    result: list[tuple[str, str]] = []
    for item in items:
        uuid = getattr(item, "dataset_uuid", None)
        if not uuid:
            continue
        hardlink_record = session.query(DatasetHardLinkDB).filter(
            DatasetHardLinkDB.dataset_uuid == uuid
        ).first()
        if not hardlink_record or not hardlink_record.hard_link_path:
            _logger.warning("No hardlink path for dataset_uuid=%s, skipping", uuid)
            continue
        path = Path(hardlink_record.hard_link_path)
        if not path.exists():
            _logger.warning("Hardlink path does not exist for dataset_uuid=%s: %s", uuid, path)
            continue
        result.append((str(path), uuid))
    return result


def _get_hub_field_prefix(dataset_table: "type[DatasetDB]") -> tuple[str, str]:
    '''Return the correct field prefixes for both HuggingFace and ModelScope hubs,
    trying short versions first (hf/ms), then long versions (huggingface/modelscope).'''
    if hasattr(dataset_table, "hf_upload_status"):
        hf_prefix = "hf"
    elif hasattr(dataset_table, "huggingface_upload_status"):
        hf_prefix = "huggingface"
    else:
        raise AttributeError(
            "Neither 'hf_upload_status' nor 'huggingface_upload_status' field exists in dataset table"
        )

    if hasattr(dataset_table, "ms_upload_status"):
        ms_prefix = "ms"
    elif hasattr(dataset_table, "modelscope_upload_status"):
        ms_prefix = "modelscope"
    else:
        raise AttributeError(
            "Neither 'ms_upload_status' nor 'modelscope_upload_status' field exists in dataset table"
        )

    return hf_prefix, ms_prefix
