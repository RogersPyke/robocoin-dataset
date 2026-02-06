"""Client component for distributed hub upload tasks.

This module provides the client-side execution for:
- Connecting to the server and requesting tasks
- Running upload on assigned datasets
- Multi-client process management
"""

import asyncio
import logging
import multiprocessing as mp
import time
import traceback
from functools import cached_property
from pathlib import Path

from robocoin_dataset.distribution_computation.constant import (
    CLIENT_ID,
    DATASET_UUID,
    MSG_CONTENT,
    MSG_TYPE,
    TASK_ID,
    TASK_RESULT,
)
from robocoin_dataset.distribution_computation.task_client import TaskClient
from robocoin_dataset.format_converter.tolerobot.constant import LEFORMAT_PATH
from ..config.constant import DatasetsHubEnum
from .utils import (
    UploadConfig,
    LocalDsUploadUtil,
)

TASK_CATEGORY = "hub_upload"


class HubUploadClient(TaskClient):
    """Client for distributed hub upload tasks."""

    def __init__(
        self,
        server_uri: str = "ws://localhost:2100",
        hub_name: DatasetsHubEnum = DatasetsHubEnum.huggingface,
        token: str = "",
        namespace: str = "",
        output_path: str | Path = "",
        force_overwrite: bool = False,
        readme_only: bool = False,
        heartbeat_interval: float = 30.0,
        request_task_timeout: float | None = None,
        logger: logging.Logger | None = None,
        tqdm_position: int = 0,
    ) -> None:
        """Initialize hub upload client with global configuration.

        Args:
            server_uri: WebSocket URI of the server
            hub_name: Target hub platform (huggingface/modelscope)
            token: Authentication token for the hub
            namespace: Username/namespace on the hub platform
            output_path: Path to output directory for YAML/README generation
            force_overwrite: Force overwrite existing repositories
            heartbeat_interval: Heartbeat interval in seconds
            logger: Logger instance
            tqdm_position: Position for tqdm progress bar (for multi-client)
        """
        super().__init__(
            server_uri=server_uri,
            heartbeat_interval=heartbeat_interval,
            request_task_timeout=request_task_timeout,
            logger=logger,
        )
        self.tqdm_position = tqdm_position

        # Store global upload configuration
        self.hub_name = hub_name
        self.token = token
        self.namespace = namespace
        self.output_path = Path(output_path).expanduser().absolute() if output_path else Path("./dataset_info")
        self.force_overwrite = force_overwrite
        self.readme_only = readme_only

        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)

    def get_task_category(self) -> str:
        return TASK_CATEGORY

    def generate_task_request_desc(self) -> dict:
        return {}

    @cached_property
    def upload_util(self) -> LocalDsUploadUtil:
        """
        Get upload utility instance with lazy initialization.

        This property uses @cached_property so that the uploader is created only
        once per client process, based on configuration that is constant during
        the whole lifecycle (hub_name, token, namespace, output_path, etc.).
        Per-task dynamic parameters (e.g. dataset metadata) are passed directly
        into the upload call and are NOT baked into this config.

        Returns:
            LocalDsUploadUtil instance (cached or newly created)

        """
        self.logger.debug(
            "Initializing upload utility "
            f"| hub={self.hub_name.value} | namespace={self.namespace} | output_path={self.output_path}"
        )

        # Note: root_path is intentionally omitted here. The client works purely with
        # hardlink paths provided by the server and does not rely on a global root_path.
        upload_config = UploadConfig(
            hub_name=self.hub_name,
            token=self.token,
            namespace=self.namespace,
            output_path=str(self.output_path),
            pg_cfg_path="",  # Client never reads DB; metadata is provided by server
            skip_err=True,
            upload_force_overwrite=self.force_overwrite,
            upload_readme_only=self.readme_only,
        )

        return LocalDsUploadUtil(upload_config)

    def _sync_process_task(self, task_content: dict) -> dict:
        """
        Process a single upload task synchronously
        (implements abstract method from TaskClient).
        """
        # Extract task-specific parameters
        dataset_uuid = task_content.get(DATASET_UUID, "unknown")
        hardlink_path_str = task_content.get(LEFORMAT_PATH)
        metadata = task_content.get("metadata")

        # Extract client configuration from task content (server provides all necessary config)
        client_config = task_content.get("client_config", {})

        # Use server-provided config with fallback to client defaults only for optional params
        effective_token = client_config.get("token") or self.token
        effective_namespace = client_config.get("namespace") or self.namespace
        effective_hub_name_str = client_config.get("hub_name")
        effective_output_path = client_config.get("output_path") or str(self.output_path)
        effective_force_overwrite = client_config.get("force_overwrite", self.force_overwrite)
        effective_readme_only = client_config.get("readme_only", self.readme_only)

        self.logger.debug(f"Client config received: {client_config}")

        # Parse hub_name from string if provided in task
        if effective_hub_name_str:
            try:
                effective_hub_name = DatasetsHubEnum[effective_hub_name_str]
            except KeyError:
                self.logger.warning(
                    f"Unknown hub_name '{effective_hub_name_str}' in task, using default {self.hub_name.value}"
                )
                effective_hub_name = self.hub_name
        else:
            effective_hub_name = self.hub_name

        if not hardlink_path_str:
            return {
                "success": False,
                "error_message": "No hardlink path provided in task content"
            }

        hardlink_path = Path(hardlink_path_str)
        dataset_name = hardlink_path.name.removesuffix("_qced_hardlink").removesuffix("_hardlink")

        self.logger.info(f"Processing task | UUID: {dataset_uuid} | Dataset: {dataset_name}")
        self.logger.debug(f"Task details | UUID: {dataset_uuid} | Path: {hardlink_path} | Hub: {effective_hub_name.value} | Namespace: {effective_namespace}")

        try:
            # Before first use of upload_util, allow server-provided config to
            # override defaults for this client process (only once).
            if "upload_util" not in self.__dict__:
                self.hub_name = effective_hub_name
                self.token = effective_token
                self.namespace = effective_namespace
                self.output_path = Path(effective_output_path).expanduser().absolute()
                self.force_overwrite = effective_force_overwrite
                self.readme_only = effective_readme_only

            # Use pre-built metadata from server; client never touches DB.
            upload_success, upload_error = self.upload_util._upload_one_dataset(
                hardlink_path,
                metadata=metadata,
            )
            if upload_success:
                self.logger.info(f"Task completed | UUID: {dataset_uuid} | Success: True")
                return {
                    "success": True
                }
            self.logger.error(f"Task failed | UUID: {dataset_uuid} | Error: {upload_error}")
            return {
                "success": False,
                "error_message": upload_error
            }
        except Exception as e:
            tb = traceback.format_exc()
            error_msg = f"Unexpected error during upload: {e}\n\nFull traceback:\n{tb}"
            self.logger.error(f"Task exception | UUID: {dataset_uuid} | Error: {e}", exc_info=True)
            return {
                "success": False,
                "error_message": error_msg
            }


async def run_one_client_async(
    server_uri: str,
    hub_name: DatasetsHubEnum,
    token: str,
    namespace: str,
    output_path: str | Path,
    force_overwrite: bool,
    readme_only: bool,
    heartbeat_interval: float,
    request_timeout: float,
    logger: logging.Logger,
    tqdm_position: int = 0,
) -> dict:
    """Run a single client that connects to server and processes tasks until none remain.

    Args:
        server_uri: WebSocket URI of the server
        hub_name: Target hub platform
        token: Authentication token
        namespace: Username/namespace
        output_path: Output path for YAML/README
        force_overwrite: Force overwrite existing repos
        readme_only: Only update README files without uploading dataset files
        heartbeat_interval: Heartbeat interval in seconds
        logger: Logger instance
        tqdm_position: Position for tqdm progress bar

    Returns:
        Statistics dictionary with keys: tasks_processed, tasks_succeeded, tasks_failed
    """
    logger.info(f"Client starting | Server: {server_uri} | Hub: {hub_name.value} | Namespace: {namespace}")
    logger.debug(f"Configuration | Heartbeat: {heartbeat_interval}s")

    client = HubUploadClient(
        server_uri=server_uri,
        hub_name=hub_name,
        token=token,
        namespace=namespace,
        output_path=output_path,
        force_overwrite=force_overwrite,
        readme_only=readme_only,
        heartbeat_interval=heartbeat_interval,
        request_task_timeout=request_timeout if request_timeout > 0 else None,
        logger=logger,
        tqdm_position=tqdm_position,
    )

    # Track task counts
    tasks_processed = 0
    tasks_succeeded = 0
    tasks_failed = 0

    # Connect to server
    try:
        if not client.connected:
            logger.debug(f"Connecting to server at {server_uri}...")
            try:
                await client.connect_to_server()
            except ConnectionError as e:
                logger.error(f"Connection failed: {e}")
                return {
                    "tasks_processed": 0,
                    "tasks_succeeded": 0,
                    "tasks_failed": 0,
                }
            except Exception as e:
                logger.error(f"Connection error: {e}", exc_info=True)
                return {
                    "tasks_processed": 0,
                    "tasks_succeeded": 0,
                    "tasks_failed": 0,
                }

        logger.debug("Starting message receiver...")
        client._receiver_task = asyncio.create_task(client._message_receiver())

        registration_success = await client.register()
        if not registration_success or not client.client_id:
            logger.error("Registration failed - no client_id received")
            return {
                "tasks_processed": 0,
                "tasks_succeeded": 0,
                "tasks_failed": 0,
            }

        await client._start_heartbeat()
        logger.info(f"Client ready | ID: {client.client_id}")

        # Process tasks until none remain
        while True:
            logger.debug("Requesting task...")
            task = await client.request_task()
            if task is None:
                logger.info("No more tasks available")
                break

            result_content = await asyncio.to_thread(client._sync_process_task, task)
            tasks_processed += 1

            # Check if task succeeded or failed
            if result_content.get("success"):
                tasks_succeeded += 1
            else:
                tasks_failed += 1

            result = {
                MSG_TYPE: TASK_RESULT,
                MSG_CONTENT: result_content,
            }
            result[TASK_ID] = task.get(TASK_ID)
            result[CLIENT_ID] = client.client_id

            logger.debug(f"Submitting result for task {task.get(TASK_ID)}...")
            await client.submit_result(result)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Client runtime exception: {e}", exc_info=True)
    finally:
        logger.debug("Cleaning up and disconnecting...")
        await client._cleanup()
        logger.info("Client shutdown complete")
        logger.info(f"Summary | Processed: {tasks_processed} | Succeeded: {tasks_succeeded} | Failed: {tasks_failed}")

    return {
        "tasks_processed": tasks_processed,
        "tasks_succeeded": tasks_succeeded,
        "tasks_failed": tasks_failed,
    }


def run_one_client_process_main(
    server_uri: str,
    hub_name: DatasetsHubEnum,
    token: str,
    namespace: str,
    output_path: str | Path,
    force_overwrite: bool,
    readme_only: bool,
    heartbeat_interval: float,
    request_timeout: float,
    log_dir: str | Path,
    log_level: str,
    process_id: int,
    stats_queue: "mp.Queue | None" = None,
) -> int:
    """Entry point for each client process in multi-client mode."""
    import sys


    # Enable console output for DEBUG mode, otherwise suppress it
    enable_console = (log_level == "DEBUG")

    if not enable_console:
        # Suppress console output for client processes in non-DEBUG mode
        # Remove all handlers from root logger
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        logging.basicConfig(
            level=logging.CRITICAL + 1,  # Disable console output
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            handlers=[],  # No handlers
        )

    # Create per-process logger with fixed file name (client_00.log, client_01.log, etc.)
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    log_file = log_dir_path / f"client_{process_id:02d}.log"

    logger = logging.getLogger(f"hub_upload_client_{process_id:02d}")
    logger.setLevel(logging.DEBUG)  # File gets DEBUG level
    logger.propagate = False

    # Remove existing handlers
    logger.handlers.clear()

    # File handler - DEBUG level for detailed logs
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(log_format, datefmt=date_format)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)  # File gets all DEBUG logs
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler - only if DEBUG mode
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level, logging.INFO))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.info(f"Hub upload client process {process_id} started, connecting to {server_uri}")

    # Run async client (use process_id as tqdm_position for multi-client progress bars)
    try:
        stats = asyncio.run(
            run_one_client_async(
                server_uri=server_uri,
                hub_name=hub_name,
                token=token,
                namespace=namespace,
                output_path=output_path,
                force_overwrite=force_overwrite,
                readme_only=readme_only,
                heartbeat_interval=heartbeat_interval,
                request_timeout=request_timeout,
                logger=logger,
                tqdm_position=process_id,
            )
        )
        # Send statistics back to parent process
        if stats_queue is not None:
            stats_queue.put({"process_id": process_id, **stats})
        return 0 if stats["tasks_failed"] == 0 else 1
    except Exception as e:
        # Always show critical errors to console, regardless of log level
        error_msg = f"❌ Hub upload client process {process_id} failed: {e}"
        logger.error(error_msg, exc_info=(log_level == "DEBUG"))

        # Always print critical errors to stderr so user sees them
        print(f"\n{error_msg}", file=sys.stderr)
        if log_level == "DEBUG":
            print(traceback.format_exc(), file=sys.stderr)

        if stats_queue is not None:
            stats_queue.put(
                {
                    "process_id": process_id,
                    "tasks_processed": 0,
                    "tasks_succeeded": 0,
                    "tasks_failed": 0,
                    "error": str(e),
                }
            )
        return 1


def run_multi_clients(
    server_uri: str,
    num_clients: int,
    hub_name: DatasetsHubEnum,
    token: str,
    namespace: str,
    output_path: str | Path,
    force_overwrite: bool,
    readme_only: bool,
    heartbeat_interval: float,
    request_timeout: float,
    log_dir: str | Path,
    log_level: str,
) -> int:
    """Spawn multiple client processes.

    Args:
        server_uri: WebSocket URI of the server (e.g. ws://localhost:2100)
        num_clients: Number of client processes to spawn
        hub_name: Target hub platform
        token: Authentication token
        namespace: Username/namespace
        output_path: Output path for YAML/README
        force_overwrite: Force overwrite existing repos
        readme_only: Only update README files without uploading dataset files
        heartbeat_interval: Heartbeat interval in seconds
        log_dir: Directory for log files
        log_level: Logging level string (e.g. "INFO", "DEBUG")

    Returns:
        Exit code: 0 if all processes succeeded, 1 otherwise
    """
    # Create console logger for user-facing messages
    console_logger = logging.getLogger("multi_client_console")
    console_logger.setLevel(logging.INFO)
    if not console_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        console_logger.addHandler(handler)

    console_logger.info("\n" + "=" * 80)
    console_logger.info("🚀 STARTING MULTI-CLIENT HUB UPLOAD".center(80))
    console_logger.info("=" * 80)
    console_logger.info(f"\n{'CONFIGURATION'}")
    console_logger.info(f"  Clients            : {num_clients}")
    console_logger.info(f"  Server URI         : {server_uri}")
    console_logger.info(f"  Hub                : {hub_name.value}")
    console_logger.info(f"  Namespace          : {namespace}")
    console_logger.info(f"  Heartbeat interval : {heartbeat_interval}s")
    console_logger.info(f"  Log directory      : {log_dir}")
    console_logger.info(f"\n{'SPAWNING PROCESSES'}")

    # Create queue for collecting statistics from child processes
    stats_queue = mp.Queue()

    processes = []
    start_time = time.time()

    for i in range(num_clients):
        proc = mp.Process(
            target=run_one_client_process_main,
            kwargs=dict(
                server_uri=server_uri,
                hub_name=hub_name,
                token=token,
                namespace=namespace,
                output_path=output_path,
                force_overwrite=force_overwrite,
                readme_only=readme_only,
                heartbeat_interval=heartbeat_interval,
                request_timeout=request_timeout,
                log_dir=log_dir,
                log_level=log_level,
                process_id=i,
                stats_queue=stats_queue,
            ),
        )
        proc.start()
        processes.append(proc)
        console_logger.info(f"  ✓ Process {i:>2} spawned (PID: {proc.pid})")

        # Add startup delay to avoid thundering herd
        if i < num_clients - 1:
            time.sleep(0.8)

    console_logger.info(f"\n{'EXECUTION'}")
    console_logger.info(f"  ⏳ Waiting for {num_clients} client(s) to complete...")
    console_logger.info("  💡 Press Ctrl+C to interrupt")
    console_logger.info("")

    exit_codes = {}

    try:
        # Wait for all processes to complete
        for i, proc in enumerate(processes):
            proc.join()
            exit_codes[i] = proc.exitcode

    except KeyboardInterrupt:
        console_logger.info("\n\n" + "=" * 80)
        console_logger.info("⚠️  INTERRUPTION DETECTED - SHUTTING DOWN".center(80))
        console_logger.info("=" * 80 + "\n")
        for i, proc in enumerate(processes):
            if proc.is_alive():
                console_logger.info(f"  ⏹  Terminating process {i:>2} (PID: {proc.pid})")
                proc.terminate()
                proc.join(timeout=5.0)
                if proc.is_alive():
                    console_logger.info(f"  ⚠️  Force-killing process {i:>2} (PID: {proc.pid})")
                    proc.kill()
                    proc.join()
                exit_codes[i] = -2  # Mark as interrupted

    elapsed = time.time() - start_time

    # Collect statistics from queue
    process_stats = {}
    try:
        while not stats_queue.empty():
            stats = stats_queue.get_nowait()
            process_stats[stats["process_id"]] = stats
    except Exception:
        pass

    # Calculate aggregated task statistics
    total_tasks_processed = sum(s.get("tasks_processed", 0) for s in process_stats.values())
    total_tasks_succeeded = sum(s.get("tasks_succeeded", 0) for s in process_stats.values())
    total_tasks_failed = sum(s.get("tasks_failed", 0) for s in process_stats.values())

    # Calculate process statistics
    process_success_count = sum(1 for code in exit_codes.values() if code == 0)
    process_fail_count = sum(
        1 for code in exit_codes.values() if code not in (0, None) and code is not None
    )

    # Format elapsed time
    if elapsed < 60:
        time_str = f"{elapsed:.1f}s"
    elif elapsed < 3600:
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        time_str = f"{minutes}m {seconds}s"
    else:
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        time_str = f"{hours}h {minutes}m"

    # Summary header
    console_logger.info("\n" + "=" * 80)
    console_logger.info("📊 MULTI-CLIENT HUB UPLOAD SUMMARY".center(80))
    console_logger.info("=" * 80)

    # Configuration section
    console_logger.info(f"\n{'CONFIGURATION'}")
    console_logger.info(f"  Clients spawned    : {num_clients}")
    console_logger.info(f"  Elapsed time       : {time_str}")

    # Task results section
    console_logger.info(f"\n{'TASK RESULTS'}")
    console_logger.info(f"  Total processed    : {total_tasks_processed}")
    console_logger.info(f"  ✅ Succeeded       : {total_tasks_succeeded}")
    console_logger.info(f"  ❌ Failed          : {total_tasks_failed}")

    # Process status section
    console_logger.info(f"\n{'PROCESS STATUS'}")
    console_logger.info(f"  ✅ Completed       : {process_success_count}")
    console_logger.info(f"  ❌ Failed          : {process_fail_count}")

    # Per-process details table
    if exit_codes:
        console_logger.info(f"\n{'PROCESS DETAILS'}")
        console_logger.info(f"  {'ID':<6} {'Status':<18} {'Tasks':<10}")
        console_logger.info(f"  {'-'*6} {'-'*18} {'-'*10}")

        for proc_id in sorted(exit_codes.keys()):
            code = exit_codes[proc_id]
            stats = process_stats.get(proc_id, {})
            tasks_processed = stats.get("tasks_processed", 0)
            error_msg = stats.get("error")

            if code == 0:
                status = "✅ SUCCESS"
            elif code == -2:
                status = "⚠️  INTERRUPTED"
            elif code is None:
                status = "❓ UNKNOWN"
            else:
                status = f"❌ FAILED (exit {code})"

            console_logger.info(f"  {proc_id:<6} {status:<18} {tasks_processed:<10}")

            # Show error message if present
            if error_msg:
                console_logger.info(f"         Error: {error_msg}")

    # Show any error details
    errors_found = [s for s in process_stats.values() if s.get("error")]
    if errors_found:
        console_logger.info(f"\n{'ERROR DETAILS'}")
        for stats in errors_found:
            proc_id = stats["process_id"]
            error = stats["error"]
            console_logger.info(f"  Process {proc_id}: {error}")

    console_logger.info("\n" + "=" * 80 + "\n")

    return 0 if total_tasks_failed == 0 else 1


__all__ = [
    "HubUploadClient",
    "run_one_client_async",
    "run_one_client_process_main",
    "run_multi_clients",
    "TASK_CATEGORY",
]
