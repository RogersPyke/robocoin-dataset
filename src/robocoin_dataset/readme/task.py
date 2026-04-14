"""Task query helpers for README generation.

This module provides:
- list_readme_tasks: Query datasets eligible for README generation from database.
"""

from pathlib import Path
from typing import TYPE_CHECKING

from sqlalchemy.sql.expression import and_, or_

from robocoin_dataset.database.models import DatasetDB, DatasetHardLinkDB, TaskStatus

if TYPE_CHECKING:
    from sqlalchemy.orm import Session


def _get_upload_status_field_names() -> tuple[str, str]:
    """Return upload status field names for huggingface and modelscope."""
    hf_field = (
        "huggingface_upload_status"
        if hasattr(DatasetDB, "huggingface_upload_status")
        else "hf_upload_status"
    )
    ms_field = (
        "ms_upload_status"
        if hasattr(DatasetDB, "ms_upload_status")
        else "modelscope_upload_status"
    )
    return hf_field, ms_field


def list_readme_tasks(
    session: "Session",
    ignore_uploaded: bool = False,
    target_dataset_uuid: str = "",
) -> list[tuple[str, Path]]:
    """List datasets eligible for README generation.

    Eligibility rules:
    1) data_loader_detection_status == COMPLETED
    2) when ignore_uploaded is False, dataset must still be not fully uploaded
       (at least one upload status is not COMPLETED)
    3) dataset must have a valid hardlink path in dataset_hard_link table
    Output tuple fields:
        - dataset_uuid (str): Unique dataset identifier.
        - hardlink_path (Path): Dataset root directory (hardlink directory).
    """
    hf_status_field, ms_status_field = _get_upload_status_field_names()

    filters = [DatasetDB.data_loader_detection_status == TaskStatus.COMPLETED]
    if not ignore_uploaded:
        filters.append(
            or_(
                getattr(DatasetDB, hf_status_field) != TaskStatus.COMPLETED,
                getattr(DatasetDB, ms_status_field) != TaskStatus.COMPLETED,
            )
        )

    query = session.query(DatasetDB).filter(and_(*filters))
    if target_dataset_uuid:
        query = query.filter(DatasetDB.dataset_uuid == target_dataset_uuid)

    datasets = query.order_by(DatasetDB.id.asc()).all()
    if not datasets:
        return []

    task_items: list[tuple[str, Path]] = []
    for ds in datasets:
        hardlink_item = session.query(DatasetHardLinkDB).filter(
            DatasetHardLinkDB.dataset_uuid == ds.dataset_uuid
        ).first()
        if not hardlink_item or not hardlink_item.hard_link_path:
            continue

        hardlink_path = Path(hardlink_item.hard_link_path).expanduser()
        if not hardlink_path.exists():
            continue
        task_items.append((ds.dataset_uuid, hardlink_path))

    return task_items


__all__ = ["list_readme_tasks"]
