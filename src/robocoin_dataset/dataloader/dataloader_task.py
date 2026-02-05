# This file contains the task management logic and definitions for dataloader detection.
# If any pre-satge changed, just modify the PRE_STAGE and PRE_VERSION definitions(str).

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

PRE_STAGE = "visualize_check_status"
PRE_VERSION = "visualize_check_version"
CURRENT_STAGE = "data_loader_detection_status"
CURRENT_VERSION = "data_loader_detection_version"
CURRENT_VERSION_PS = "data_loader_detection_version_ps"
CURRENT_ERR_MSG = "data_loader_detection_err_msg"

def _sync_dataloader_detection_tasks(
    session: "Session",
    logger: logging.Logger | None = None,
) -> None:
    """
    Function: 
    Synchronize dataloader detection tasks status and version.
    scan through all items in database and mark CURRENT_STAGE to PENDING if:
    - PRE_STAGE is COMPLETED
    - CURRENT_STAGE is PENDING, or COMPLETED but outdated(CURRENT_VERSION_PS < PRE_VERSION)

    Use session directly to avoid overhead of query and commit.
    """
    from sqlalchemy.sql.expression import and_, or_

    from robocoin_dataset.database.models import DatasetDB, TaskStatus

    _logger = logger or logging.getLogger(__name__)

    query = session.query(DatasetDB).filter(
        and_(
            # MUST: PRE_STAGE must be COMPLETED, else never process.
            DatasetDB[PRE_STAGE] == TaskStatus.COMPLETED,
            or_(
                # Trigger branch 1: CURRENT_STAGE is PENDING, so mark it to PENDING.(unchanged)
                DatasetDB[CURRENT_STAGE] == TaskStatus.PENDING,
                # Trigger branch 2: CURRENT_STAGE is COMPLETED but outdated, so mark it to PENDING.
                and_(
                    DatasetDB[CURRENT_STAGE] == TaskStatus.COMPLETED,
                    DatasetDB[CURRENT_VERSION_PS] < DatasetDB[PRE_VERSION],
                ),
            ),
        ),
    )

    items = query.all()
    if not items:
        _logger.debug("No datasets found for dataloader detection")
        return

    if items:
        _logger.debug(f"Marked {len(items)} dataset(s) as PENDING for dataloader detection")

    for item in items:
        # Up we did query, now we do write operation.
        item[CURRENT_STAGE] = TaskStatus.PENDING
        # Sync: align the version_ps with the pre_version, so that we can detect the outdated.
        item[CURRENT_VERSION_PS] = item[PRE_VERSION]
        # Increment the version to mark the processed time ++
        item[CURRENT_VERSION] = (item[CURRENT_VERSION] or 0) + 1

    session.commit()


def _gen_one_dataloader_detection_task(session: "Session") -> tuple[str | None, Path | None]:
    """
    Function: 
    Scan through all items in database and find the first one that:
    - PRE_STAGE is COMPLETED
    - CURRENT_STAGE is PENDING
    - and mark it to PROCESSING.

    Also, check if the hardlink path exists in db.
    if True, return the dataset_uuid and hardlink_path, else None.
    if False, mark the task as FAILED and return (None, None).(Done by upper wrappers not here)

    Returns:
        (dataset_uuid, hardlink_path) or (None, None) if no task available.
        (dataset_uuid, None) == err: no valid hardlink path in db.

    Raises:
        NO raise since raise kills uuid passing and thus upper wrappers
        cannot update task status in database.
    """
    from robocoin_dataset.database.models import DatasetDB, DatasetHardLinkDB, TaskStatus

    item = (
        session.query(DatasetDB)
        .filter(DatasetDB[PRE_STAGE] == TaskStatus.COMPLETED)
        .filter(DatasetDB[CURRENT_STAGE] == TaskStatus.PENDING)
        .first()
    )
    if not item:
        return None, None

    # Claim the task (to PROCESSING) avoiding picking up by other workers
    item[CURRENT_STAGE] = TaskStatus.PROCESSING
    session.commit()

    # Get hardlink path from db using dataset_uuid
    hardlink_record = session.query(DatasetHardLinkDB).filter(
        DatasetHardLinkDB.dataset_uuid == item.dataset_uuid
    ).first()

    # Verify hardlink path exists on db
    if not hardlink_record or not hardlink_record.hard_link_path:
        return item.dataset_uuid, None
    #NOTE: This means Err, but handled by upper wrappers.

    hardlink_path = Path(hardlink_record.hard_link_path)

    return item.dataset_uuid, hardlink_path


def _mark_task_completed(session: "Session", dataset_uuid: str) -> None:
    """
    Mark a dataloader detection task as completed in session
    identify by uuid.
    Directly use session to avoid overhead of query and commit.
    """
    from robocoin_dataset.database.models import DatasetDB, TaskStatus

    item = session.query(DatasetDB).filter(DatasetDB.dataset_uuid == dataset_uuid).first()
    if item:
        item.data_loader_detection_status = TaskStatus.COMPLETED
        item.data_loader_detection_err_msg = None
        session.commit()


def _mark_task_failed(session: "Session", dataset_uuid: str, error_message: str) -> None:
    """
    Mark a dataloader detection task as failed in session
    identify by uuid, and set error message.(Neer str passing thus.)
    Directly use session to avoid overhead of query and commit.
    """
    from robocoin_dataset.database.models import DatasetDB, TaskStatus

    try:
        item = session.query(DatasetDB).filter(DatasetDB.dataset_uuid == dataset_uuid).first()
        if item:
            item.data_loader_detection_status = TaskStatus.FAILED
            item.data_loader_detection_err_msg = error_message
            session.commit()
    except Exception as e:
        raise RuntimeError(f"Failed to mark task as failed: {e}") from e

# Do export.
__all__ = [
    "_sync_dataloader_detection_tasks",
    "_gen_one_dataloader_detection_task",
    "_mark_task_completed",
    "_mark_task_failed",
]
