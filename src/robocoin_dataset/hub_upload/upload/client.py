"""Client component for distributed hub upload tasks.

This module provides the client-side execution for:
- Connecting to the server and requesting tasks
- Running upload on assigned datasets
- Multi-client process management

Dependencies:
    - asyncio: For asynchronous task processing
    - logging: For audit logging
    - multiprocessing: For multi-client process management
    - robocoin_dataset.distribution_computation.task_client: Base TaskClient class

Usage:
    Single client:
        config = UploadConfig(...)
        stats = asyncio.run(run_one_client_async(config, logger))

    Multi-client:
        config = UploadConfig(...)
        exit_code = run_multi_clients(config, num_clients=4, ...)
"""

import asyncio
import logging
import multiprocessing as mp
import os
import time
import traceback
from functools import cached_property
from pathlib import Path

from robocoin_dataset.distribution_computation.constant import (
    CLIENT_ID,
    MSG_CONTENT,
    MSG_TYPE,
    TASK_ID,
    TASK_RESULT,
)
from robocoin_dataset.distribution_computation.task_client import TaskClient
from .utils import (
    UploadConfig,
    UploadUtil,
)

TASK_CATEGORY = "hub_upload"

# ANSI color codes for terminal output
# Red for WARNING and ERR
ANSI_RED = "\033[91m"
# Green for SUCCESS
ANSI_GREEN = "\033[92m"
# Blue for URLs and arguments
ANSI_BLUE = "\033[94m"
# Reset color
ANSI_RESET = "\033[0m"


def _colorize(text: str, color: str, use_color: bool = True) -> str:
    """
    Apply ANSI color code to text if terminal supports colors.

    Args:
        text: Text to colorize
        color: ANSI color code (e.g., ANSI_RED, ANSI_GREEN, ANSI_BLUE)
        use_color: Whether to apply color (default: True, auto-detected if None)

    Returns:
        Colorized text string
    """
    if use_color is False:
        return text
    # Auto-detect if use_color is None
    if use_color is True:
        # Check if terminal supports colors
        use_color = os.getenv("TERM") not in (None, "dumb") and os.getenv("NO_COLOR") is None
    if use_color:
        return f"{color}{text}{ANSI_RESET}"
    return text


def _log_success(logger: logging.Logger, message: str) -> None:
    """
    Log success message with green color.

    Args:
        logger: Logger instance
        message: Success message to log
    """
    logger.info(_colorize(message, ANSI_GREEN))


def _log_error(logger: logging.Logger, message: str) -> None:
    """
    Log error message with red color.

    Args:
        logger: Logger instance
        message: Error message to log
    """
    logger.error(_colorize(message, ANSI_RED))


def _log_warning(logger: logging.Logger, message: str) -> None:
    """
    Log warning message with red color.

    Args:
        logger: Logger instance
        message: Warning message to log
    """
    logger.warning(_colorize(message, ANSI_RED))


def _log_url(logger: logging.Logger, message: str, level: int = logging.INFO) -> None:
    """
    Log URL or argument with blue color.

    Args:
        logger: Logger instance
        message: URL or argument message to log
        level: Logging level (default: INFO)
    """
    logger.log(level, _colorize(message, ANSI_BLUE))


class HubUploadClient(TaskClient):
    """Client for distributed hub upload tasks."""

    def __init__(
        self,
        config: UploadConfig,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize hub upload client with UploadConfig.

        Args:
            config: UploadConfig object containing all client configuration.
                   Expected fields: client_host (str), client_port (int),
                   client_heartbeat_interval (float), request_task_timeout (float | None).
            logger: Logger instance (optional). If None, uses default logger.

        Returns:
            None. Initializes HubUploadClient instance.

        Usage:
            Used internally by run_one_client_async() to create client instances.
        """
        # Build server_uri from config
        server_uri = f"ws://{config.client_host}:{config.client_port}"
        self.config = config
        # Get request_task_timeout from config, default to None if not set
        request_task_timeout = getattr(config, "request_task_timeout", None)
        if request_task_timeout is not None and request_task_timeout <= 0:
            request_task_timeout = None
        super().__init__(
            server_uri=server_uri,
            heartbeat_interval=config.client_heartbeat_interval,
            request_task_timeout=request_task_timeout,
            logger=logger,
        )

    def get_task_category(self) -> str:
        return TASK_CATEGORY

    def generate_task_request_desc(self) -> dict:
        return {}

    @cached_property
    def upload_util(self) -> UploadUtil:
        """
        Get upload utility instance with lazy initialization.

        This property uses @cached_property so that the uploader is created only
        once per client process, based on configuration that is constant during
        the whole lifecycle (hub_name, token, namespace, etc.).
        Per-task dynamic parameters (e.g. dataset metadata) are passed directly
        into the upload call and are NOT baked into this config.

        Input:
            None (uses self.config from instance)

        Returns:
            UploadUtil: Upload utility instance (cached or newly created)

        Usage:
            Called automatically when first accessing self.upload_util.
            Subsequent accesses return the cached instance.
        """
        hub_name = getattr(self.config, "hub_name", "unknown")
        namespace = getattr(self.config, "ms_namespace", None) or getattr(self.config, "hf_namespace", "unknown")
        self.logger.debug(
            "[HubUploadClient.upload_util] Initializing upload utility "
            f"| hub={hub_name} | namespace={namespace}"
        )
        return UploadUtil(self.config)

    def _sync_process_task(self, task_content: dict) -> dict:
        """
        Process a single upload task synchronously (implements abstract method from TaskClient).

        Input:
            task_content (dict): Task content dictionary with keys:
                - dataset_uuid (str): Unique identifier for the dataset
                - leformat_path (str): Path to the hardlink directory
                - client_config (dict, optional): Server-provided client configuration

        Returns:
            dict: Result dictionary with keys:
                - success (bool): True if upload succeeded, False otherwise
                - error_message (str, optional): Error message if success is False

        Usage:
            Called by TaskClient framework when a task is assigned to this client.
        """
        # Extract task-specific parameters
        dataset_uuid = task_content.get("dataset_uuid", "unknown")
        hardlink_path = task_content.get("leformat_path")

        # Extract client configuration from task content (server provides all necessary config)
        client_config = task_content.get("client_config", {})

        # Use server-provided config with fallback to client defaults only for optional params
        namespace = client_config.get("namespace") or getattr(self.config, "ms_namespace", None) or getattr(self.config, "hf_namespace", "unknown")
        hub_name = client_config.get("hub_name") or getattr(self.config, "hub_name", "unknown")

        self.logger.debug(f"[HubUploadClient._sync_process_task] Client config received: {client_config}")

        if not hardlink_path:
            error_msg = "No hardlink path provided in task content"
            self.logger.error(f"[HubUploadClient._sync_process_task] {error_msg}")
            return {
                "success": False,
                "error_message": error_msg
            }
        hardlink_path = Path(hardlink_path)
        dataset_name = hardlink_path.name.removesuffix("_qced_hardlink").removesuffix("_hardlink")

        self.logger.info(f"[HubUploadClient._sync_process_task] Processing task | UUID: {dataset_uuid} | Dataset: {dataset_name}")
        _log_url(self.logger, f"[HubUploadClient._sync_process_task] Task details | UUID: {dataset_uuid} | Path: {hardlink_path} | Hub: {hub_name} | Namespace: {namespace}", logging.DEBUG)

        try:
            upload_success, upload_error = self.upload_util._upload(hardlink_path)
            if upload_success:
                _log_success(self.logger, f"[HubUploadClient._sync_process_task] Task completed | UUID: {dataset_uuid} | Success: True")
                return {
                    "success": True
                }
            _log_error(self.logger, f"[HubUploadClient._sync_process_task] Task failed | UUID: {dataset_uuid} | Error: {upload_error}")
            return {
                "success": False,
                "error_message": upload_error
            }
        except Exception as e:
            tb = traceback.format_exc()
            error_msg = f"Unexpected error during upload: {e}\n\nFull traceback:\n{tb}"
            _log_error(self.logger, f"[HubUploadClient._sync_process_task] Task exception | UUID: {dataset_uuid} | Error: {e}")
            self.logger.debug(f"[HubUploadClient._sync_process_task] Full traceback:\n{tb}")
            return {
                "success": False,
                "error_message": error_msg
            }


# ===== Client entry points =====


async def run_one_client_async(
    config: UploadConfig,
    logger: logging.Logger | None = None,
) -> dict:
    """Run a single client that connects to server and processes tasks until none remain.

    Input:
        config (UploadConfig): UploadConfig object containing all client configuration.
                               Expected fields: client_host (str), client_port (int),
                               hub_name (str), ms_namespace (str), hf_namespace (str),
                               client_heartbeat_interval (float).
        logger (logging.Logger | None): Logger instance. If None, creates default logger.

    Returns:
        dict: Statistics dictionary with keys:
            - tasks_processed (int): Total number of tasks processed
            - tasks_succeeded (int): Number of tasks that succeeded
            - tasks_failed (int): Number of tasks that failed

    Usage:
        Called by run_one_client_process_main() to execute client in async context.
        Processes tasks until server indicates no more tasks available.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Extract hub info for logging
    hub_name_str = config.hub_name.lower()
    if hub_name_str in ("modelscope", "ms"):
        namespace = config.ms_namespace
        hub_display = "modelscope"
    elif hub_name_str in ("huggingface", "hf"):
        namespace = config.hf_namespace
        hub_display = "huggingface"
    else:
        namespace = getattr(config, "hf_namespace", "unknown")
        hub_display = "unknown"
    
    server_uri = f"ws://{config.client_host}:{config.client_port}"
    logger.info(f"[run_one_client_async] Client starting | Hub: {hub_display} | Namespace: {namespace}")
    _log_url(logger, f"[run_one_client_async] Server: {server_uri}")
    logger.debug(f"[run_one_client_async] Configuration | Heartbeat: {config.client_heartbeat_interval}s")

    client = HubUploadClient(
        config=config,
        logger=logger,
    )

    # Track task counts
    tasks_processed = 0
    tasks_succeeded = 0
    tasks_failed = 0

    # Connect to server
    try:
        if not client.connected:
            logger.debug(f"[run_one_client_async] Connecting to server...")
            try:
                await client.connect_to_server()
            except ConnectionError as e:
                _log_error(logger, f"[run_one_client_async] Connection failed: {e}")
                return {
                    "tasks_processed": 0,
                    "tasks_succeeded": 0,
                    "tasks_failed": 0,
                }
            except Exception as e:
                _log_error(logger, f"[run_one_client_async] Connection error: {e}")
                logger.debug(f"[run_one_client_async] Connection error traceback:", exc_info=True)
                return {
                    "tasks_processed": 0,
                    "tasks_succeeded": 0,
                    "tasks_failed": 0,
                }

        logger.debug("[run_one_client_async] Starting message receiver...")
        client._receiver_task = asyncio.create_task(client._message_receiver())

        registration_success = await client.register()
        if not registration_success or not client.client_id:
            _log_error(logger, "[run_one_client_async] Registration failed - no client_id received")
            return {
                "tasks_processed": 0,
                "tasks_succeeded": 0,
                "tasks_failed": 0,
            }

        await client._start_heartbeat()
        _log_success(logger, f"[run_one_client_async] Client ready | ID: {client.client_id}")

        # Process tasks until none remain
        while True:
            logger.debug("[run_one_client_async] Requesting task...")
            task = await client.request_task()
            if task is None:
                logger.info("[run_one_client_async] No more tasks available")
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

            logger.debug(f"[run_one_client_async] Submitting result for task {task.get(TASK_ID)}...")
            await client.submit_result(result)

    except KeyboardInterrupt:
        logger.info("[run_one_client_async] Interrupted by user")
    except Exception as e:
        _log_error(logger, f"[run_one_client_async] Client runtime exception: {e}")
        logger.debug(f"[run_one_client_async] Runtime exception traceback:", exc_info=True)
    finally:
        logger.debug("[run_one_client_async] Cleaning up and disconnecting...")
        await client._cleanup()
        logger.info("[run_one_client_async] Client shutdown complete")
        logger.info(f"[run_one_client_async] Summary | Processed: {tasks_processed} | Succeeded: {tasks_succeeded} | Failed: {tasks_failed}")

    return {
        "tasks_processed": tasks_processed,
        "tasks_succeeded": tasks_succeeded,
        "tasks_failed": tasks_failed,
    }


def run_one_client_process_main(
    config: UploadConfig,
    request_timeout: float,
    log_dir: str | Path,
    log_level: str,
    process_id: int,
    stats_queue: "mp.Queue | None" = None,
) -> int:
    """Entry point for each client process in multi-client mode.

    Input:
        config (UploadConfig): UploadConfig object containing all client configuration
        request_timeout (float): Request task timeout in seconds (unused, kept for compatibility)
        log_dir (str | Path): Directory path for log files
        log_level (str): Logging level string (e.g. "INFO", "DEBUG")
        process_id (int): Unique process identifier (0-based)
        stats_queue (mp.Queue | None): Queue for sending statistics to parent process

    Returns:
        int: Exit code (0 for success, 1 for failure)

    Usage:
        Called by run_multi_clients() as target for each multiprocessing.Process.
        Creates per-process logger and runs async client loop.
    """
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

    server_uri = f"ws://{config.client_host}:{config.client_port}"
    logger.info(f"[run_one_client_process_main] Hub upload client process {process_id} started")
    _log_url(logger, f"[run_one_client_process_main] Connecting to: {server_uri}")

    # Run async client
    try:
        stats = asyncio.run(
            run_one_client_async(
                config=config,
                logger=logger,
            )
        )
        # Send statistics back to parent process
        if stats_queue is not None:
            stats_queue.put({"process_id": process_id, **stats})
        return 0 if stats["tasks_failed"] == 0 else 1
    except Exception as e:
        # Always show critical errors to console, regardless of log level
        error_msg = f"[run_one_client_process_main] Hub upload client process {process_id} failed: {e}"
        _log_error(logger, error_msg)
        if log_level == "DEBUG":
            logger.debug(f"[run_one_client_process_main] Full traceback:", exc_info=True)

        # Always print critical errors to stderr so user sees them
        print(f"\n{_colorize(error_msg, ANSI_RED)}", file=sys.stderr)
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
    config: UploadConfig,
    num_clients: int,
    request_timeout: float,
    log_dir: str | Path,
    log_level: str,
) -> int:
    """Spawn multiple client processes for distributed upload.

    Input:
        config (UploadConfig): UploadConfig object containing all client configuration
        num_clients (int): Number of client processes to spawn (must be > 0)
        request_timeout (float): Request task timeout in seconds (unused, kept for compatibility)
        log_dir (str | Path): Directory path for log files
        log_level (str): Logging level string (e.g. "INFO", "DEBUG")

    Returns:
        int: Exit code (0 if all tasks succeeded, 1 if any task failed)

    Usage:
        Main entry point for multi-client upload mode.
        Spawns multiple processes, each running a client that connects to server.
        Waits for all processes to complete and aggregates statistics.
    """
    # Create console logger for user-facing messages
    console_logger = logging.getLogger("multi_client_console")
    console_logger.setLevel(logging.INFO)
    if not console_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        console_logger.addHandler(handler)

    # Extract hub info for logging
    hub_name_str = config.hub_name.lower()
    if hub_name_str in ("modelscope", "ms"):
        namespace = config.ms_namespace
        hub_display = "modelscope"
    else:
        namespace = config.hf_namespace
        hub_display = "huggingface"
    
    server_uri = f"ws://{config.client_host}:{config.client_port}"

    console_logger.info("\n" + "=" * 80)
    console_logger.info("[run_multi_clients] STARTING MULTI-CLIENT HUB UPLOAD".center(80))
    console_logger.info("=" * 80)
    console_logger.info(f"\n[run_multi_clients] CONFIGURATION")
    console_logger.info(f"  Clients            : {num_clients}")
    _log_url(console_logger, f"  Server URI         : {server_uri}")
    console_logger.info(f"  Hub                : {hub_display}")
    console_logger.info(f"  Namespace          : {namespace}")
    console_logger.info(f"  Heartbeat interval : {config.client_heartbeat_interval}s")
    console_logger.info(f"  Log directory      : {log_dir}")
    console_logger.info(f"\n[run_multi_clients] SPAWNING PROCESSES")

    # Create queue for collecting statistics from child processes
    stats_queue = mp.Queue()

    processes = []
    start_time = time.time()

    for i in range(num_clients):
        proc = mp.Process(
            target=run_one_client_process_main,
            kwargs=dict(
                config=config,
                request_timeout=request_timeout,
                log_dir=log_dir,
                log_level=log_level,
                process_id=i,
                stats_queue=stats_queue,
            ),
        )
        proc.start()
        processes.append(proc)
        _log_success(console_logger, f"[run_multi_clients] Process {i:>2} spawned (PID: {proc.pid})")

        # Add startup delay to avoid thundering herd
        if i < num_clients - 1:
            time.sleep(0.8)

    console_logger.info(f"\n[run_multi_clients] EXECUTION")
    console_logger.info(f"  Waiting for {num_clients} client(s) to complete...")
    console_logger.info("  Press Ctrl+C to interrupt")
    console_logger.info("")

    exit_codes = {}

    try:
        # Wait for all processes to complete
        for i, proc in enumerate(processes):
            proc.join()
            exit_codes[i] = proc.exitcode

    except KeyboardInterrupt:
        console_logger.info("\n\n" + "=" * 80)
        _log_warning(console_logger, "[run_multi_clients] INTERRUPTION DETECTED - SHUTTING DOWN".center(80))
        console_logger.info("=" * 80 + "\n")
        for i, proc in enumerate(processes):
            if proc.is_alive():
                console_logger.info(f"[run_multi_clients] Terminating process {i:>2} (PID: {proc.pid})")
                proc.terminate()
                proc.join(timeout=5.0)
                if proc.is_alive():
                    _log_warning(console_logger, f"[run_multi_clients] Force-killing process {i:>2} (PID: {proc.pid})")
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
    console_logger.info("[run_multi_clients] MULTI-CLIENT HUB UPLOAD SUMMARY".center(80))
    console_logger.info("=" * 80)

    # Configuration section
    console_logger.info(f"\n[run_multi_clients] CONFIGURATION")
    console_logger.info(f"  Clients spawned    : {num_clients}")
    console_logger.info(f"  Elapsed time       : {time_str}")

    # Task results section
    console_logger.info(f"\n[run_multi_clients] TASK RESULTS")
    console_logger.info(f"  Total processed    : {total_tasks_processed}")
    _log_success(console_logger, f"  Succeeded          : {total_tasks_succeeded}")
    _log_error(console_logger, f"  Failed             : {total_tasks_failed}")

    # Process status section
    console_logger.info(f"\n[run_multi_clients] PROCESS STATUS")
    _log_success(console_logger, f"  Completed          : {process_success_count}")
    _log_error(console_logger, f"  Failed             : {process_fail_count}")

    # Per-process details table
    if exit_codes:
        console_logger.info(f"\n[run_multi_clients] PROCESS DETAILS")
        console_logger.info(f"  {'ID':<6} {'Status':<18} {'Tasks':<10}")
        console_logger.info(f"  {'-'*6} {'-'*18} {'-'*10}")

        for proc_id in sorted(exit_codes.keys()):
            code = exit_codes[proc_id]
            stats = process_stats.get(proc_id, {})
            tasks_processed = stats.get("tasks_processed", 0)
            error_msg = stats.get("error")

            if code == 0:
                status = _colorize("SUCCESS", ANSI_GREEN)
            elif code == -2:
                status = _colorize("INTERRUPTED", ANSI_RED)
            elif code is None:
                status = "UNKNOWN"
            else:
                status = _colorize(f"FAILED (exit {code})", ANSI_RED)

            console_logger.info(f"  {proc_id:<6} {status:<18} {tasks_processed:<10}")

            # Show error message if present
            if error_msg:
                _log_error(console_logger, f"         Error: {error_msg}")

    # Show any error details
    errors_found = [s for s in process_stats.values() if s.get("error")]
    if errors_found:
        console_logger.info(f"\n[run_multi_clients] ERROR DETAILS")
        for stats in errors_found:
            proc_id = stats["process_id"]
            error = stats["error"]
            _log_error(console_logger, f"  Process {proc_id}: {error}")

    console_logger.info("\n" + "=" * 80 + "\n")

    return 0 if total_tasks_failed == 0 else 1


__all__ = [
    "HubUploadClient",
    "run_one_client_async",
    "run_one_client_process_main",
    "run_multi_clients",
    "TASK_CATEGORY",
]
