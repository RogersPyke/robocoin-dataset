"""Task management logic for dataloader checker.

This module provides:
- _sync_dataloader_check_tasks: Synchronize task status based on visualization results.
- _gen_one_dataloader_check_task: Claim a single task from the database.
"""

from pathlib import Path
from typing import TYPE_CHECKING

from sqlalchemy.sql.expression import and_, or_
from robocoin_dataset.database.models import DatasetDB, DatasetHardLinkDB, TaskStatus

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

def _sync_dataloader_check_tasks(session: "Session", target_dataset_uuid: str = "") -> None:
    """
    Function: 
    Synchronize dataloader check tasks in PostgreSQL.
    Mark tasks as PENDING if visualization is completed and dataloader check is pending or outdated.
    
    Expected input format:
    - session: SQLAlchemy Session instance connected to PostgreSQL.
    - target_dataset_uuid: str, optional, specific dataset UUID to sync.
    
    Expected output format:
    - None. Commits changes to PostgreSQL.
    
    Expected usage/scenario:
    - Called before task generation to ensure the task queue is up-to-date.
    """
    query = session.query(DatasetDB).filter(
        and_(
            # Pre-condition: Visualization check must be completed.
            DatasetDB.visualize_check_status == TaskStatus.COMPLETED,
            # Two trigger conditions:
            or_(
                # 1. Task is currently pending.
                DatasetDB.data_loader_detection_status == TaskStatus.PENDING,
                # 2. Task is completed but the version is outdated.
                and_(
                    DatasetDB.data_loader_detection_status == TaskStatus.COMPLETED,
                    DatasetDB.data_loader_detection_version_ps != DatasetDB.visualize_check_version,
                ),
            ),
        )
    )

    if target_dataset_uuid:
        query = query.filter(DatasetDB.dataset_uuid == target_dataset_uuid)

    items = query.all()

    if not items:
        return

    for item in items:
        item.data_loader_detection_status = TaskStatus.PENDING
        item.data_loader_detection_version_ps = item.visualize_check_version

    session.commit()


def _gen_one_dataloader_check_task(
    session: "Session", target_dataset_uuid: str = ""
) -> tuple[str | None, Path | None]:
    """
    Function: 
    Claim one task from the database for processing.
    
    Expected input format:
    - session: SQLAlchemy Session instance.
    - target_dataset_uuid: str, optional, specific dataset UUID to claim.
    
    Expected output format:
    - tuple(dataset_uuid, hard_link_path) or (None, None) if no task is found.
    
    Expected usage/scenario:
    - Used by both local and server-based execution to get the next task.
    """
    query = session.query(DatasetDB).filter(
        and_(
            DatasetDB.visualize_check_status == TaskStatus.COMPLETED,
            DatasetDB.data_loader_detection_status == TaskStatus.PENDING,
        )
    )

    if target_dataset_uuid:
        query = query.filter(DatasetDB.dataset_uuid == target_dataset_uuid)

    ds_item = query.first()

    if not ds_item:
        return None, None

    # Get hardlink path for the dataset.
    hard_link_item = session.query(DatasetHardLinkDB).filter(
        DatasetHardLinkDB.dataset_uuid == ds_item.dataset_uuid
    ).first()
    
    if not hard_link_item:
        return None, None
        
    hard_link_path = Path(hard_link_item.hard_link_path).expanduser()
    
    # Mark task as processing.
    ds_item.data_loader_detection_status = TaskStatus.PROCESSING
    ds_item.data_loader_detection_version = (ds_item.data_loader_detection_version or 0) + 1
    session.commit()

    return ds_item.dataset_uuid, hard_link_path

__all__ = ["_sync_dataloader_check_tasks", "_gen_one_dataloader_check_task"]
