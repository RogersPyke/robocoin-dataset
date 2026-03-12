"""Client component for distributed dataloader checker.

This module provides:
- DataLoaderCheckerClient: Requests and processes tasks from the server.
"""

import logging
import traceback

from robocoin_dataset.distribution_computation.task_client import TaskClient
from robocoin_dataset.dataloader_check.utils import load_repo

# Constants for task communication
NUM_WORKERS = "num_workers"
SAMPLE_RATE = "sample_rate"
HARD_LINK_PATH = "hard_link_path"


class DataLoaderCheckerClient(TaskClient):
    """
    Function: 
    Distributed client for dataloader checking tasks assigned by server.
    
    Expected input format:
    - server_uri: str, WebSocket URI of the server (e.g., ws://host:port).
    - heartbeat_interval: float, interval for heartbeat pings.
    - logger: logging.Logger, optional.
    
    Expected output format:
    - Result of task processing.
    
    Expected usage/scenario:
    - Used to process dataloader checks assigned by DataLoaderCheckerServer.
    Multiple clients can connect to the same server for parallel processing.
    """
    def __init__(
        self,
        server_uri: str = "ws://localhost:2010",
        heartbeat_interval: float = 10.0,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(
            server_uri=server_uri,
            heartbeat_interval=heartbeat_interval,
            logger=logger,
        )
        self.logger = logger or logging.getLogger(__name__)

    def get_task_category(self) -> str:
        """Returns the category of tasks handled by this client."""
        return "dataset dataloader checker"

    def generate_task_request_desc(self) -> dict:
        """Client can customize task request parameters."""
        return {}

    def _sync_process_task(self, task_content: dict) -> dict:
        """
        Processes a single task synchronously.
        Loads the dataset and iterates through it.
        """
        hard_link_path = task_content.get(HARD_LINK_PATH)
        num_workers = task_content.get(NUM_WORKERS)
        sample_rate = task_content.get(SAMPLE_RATE)

        self.logger.info(f"[START] Processing task: {hard_link_path}")
        try:
            load_repo(hard_link_path, sample_rate=sample_rate, num_workers=num_workers)
            self.logger.info(f"[SUCCESS] Task completed: {hard_link_path}")
            return {}
        except Exception as e:
            self.logger.error(
                f"[FAILED] Task failed: {hard_link_path}, error: {e}\n{traceback.format_exc()}",
                exc_info=True,
            )
            raise RuntimeError(f"dataset dataloader check {hard_link_path} failed") from e


__all__ = ["DataLoaderCheckerClient"]
