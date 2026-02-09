"""
RoboCoin Datasets Uploader - Main CLI Entry Point

This script uploads datasets to remote hubs (HuggingFace/ModelScope) using a database-driven strategy.
It supports three modes: local, server, and client.

MODES:
    1. Local mode (default, --local): Single machine upload with direct database access
    2. Server mode (--server): Starts a task distribution server for distributed upload
    3. Client mode (--client): Connects to server and processes tasks

WORKFLOW (Local Mode):
    For each dataset in the database:
    1. Sync upload status in database (mark as PENDING if needed)
    2. Generate upload task (mark as PROCESSING)
    3. Execute upload to hub
    4. Update database status (COMPLETED or FAILED)

Dependencies:
    - robocoin_dataset.hub_upload.upload: Core upload modules
    - robocoin_dataset.database: Database connection and models
    - asyncio: For async server/client operations
    - logging: For audit logging

Usage:
    # Local upload mode (single machine, default)
    python scripts/hub_upload/upload2hub.py \\
        --config configs/upload.yaml

    # Server mode (distribute tasks to clients)
    python scripts/hub_upload/upload2hub.py --server \\
        --config configs/upload.yaml \\
        --host 0.0.0.0 \\
        --port 2100

    # Client mode (connect to server and process tasks)
    python scripts/hub_upload/upload2hub.py --client \\
        --host 127.0.0.1 \\
        --port 2100 \\
        --num-clients 4 \\
        --config configs/upload.yaml
"""

import argparse
import asyncio
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from robocoin_dataset.database.database import DatasetDatabase
from robocoin_dataset.hub_upload.upload.client import run_multi_clients
from robocoin_dataset.hub_upload.upload.local import UploadLocal
from robocoin_dataset.hub_upload.upload.server import UploadServer
from robocoin_dataset.hub_upload.upload.task import (
    _gen_one_upload_task,
    _get_hardlink_path_by_uuid,
    _mark_upload_failed,
    _sync_upload_status,
)
from robocoin_dataset.hub_upload.upload.utils import UploadConfig, create_config

# ANSI color codes for terminal output
# Red for WARNING and ERR
ANSI_RED = "\033[91m"
# Green for SUCCESS
ANSI_GREEN = "\033[92m"
# Blue for URLs and arguments
ANSI_BLUE = "\033[94m"
# Reset color
ANSI_RESET = "\033[0m"

# Default platform name (can be overridden by config)
DEFAULT_PLATFORM_NAME = "RoboCOIN"


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
    # Auto-detect if use_color is True
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


def _normalize_hub_name(value: str | None) -> str:
    """
    Normalize hub name to standard format.

    Args:
        value: Hub name string (can be "huggingface", "hf", "modelscope", "ms")

    Returns:
        Normalized hub name string ("huggingface" or "modelscope")
    """
    if value is None:
        return "huggingface"
    normalized = value.strip().lower()
    if normalized in ("hf", "huggingface"):
        return "huggingface"
    elif normalized in ("ms", "modelscope"):
        return "modelscope"
    else:
        raise ValueError(
            f"Invalid hub_name '{value}'. Must be one of: huggingface, hf, modelscope, ms"
        )


def _generate_log_folder_name(
    mode: str,
    hub_name: str,
    namespace: str,
    timestamp: str | None = None,
) -> str:
    """
    Generate a log folder name with timestamp and key parameters.

    Args:
        mode: Mode prefix ('local' or 'dist')
        hub_name: Hub platform name (huggingface/modelscope)
        namespace: Namespace/username
        timestamp: Timestamp string in YYYYMMDD_HHMMSS format. If None, generates current timestamp.

    Returns:
        Folder name string (e.g., 'local_20240115_143022_huggingface_robocoin')
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Sanitize namespace (remove special characters that might cause issues in folder names)
    namespace_safe = namespace.replace("/", "_").replace("\\", "_").replace(" ", "_")

    return f"{mode}_{timestamp}_{hub_name}_{namespace_safe}"


def setup_logging(
    log_level: str = "INFO",
    log_folder: Path | str | None = None,
    log_filename: str = "local.log",
) -> tuple[logging.Logger, Path]:
    """
    Set up logging configuration for CLI.

    File: Contains all detailed logs at DEBUG level
    Console: Only shows INFO level and above (reduced output)

    Args:
        log_level: Logging level for console output (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_folder: Optional folder path for logs. If None, uses default logs/hub_upload
        log_filename: Name of the log file within the folder (default: local.log)

    Returns:
        Tuple of (Logger instance, log_file_path)
    """
    # Determine log directory
    if log_folder is None:
        base_log_dir = Path("logs/hub_upload")
        log_dir = base_log_dir
    else:
        log_dir = Path(log_folder)

    log_dir.mkdir(parents=True, exist_ok=True)

    # Generate log file path
    log_file = log_dir / log_filename

    # Configure root logger
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Remove any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.DEBUG)  # Set root to DEBUG to capture all

    # File handler - detailed logs at DEBUG level
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)  # File gets all DEBUG logs
    file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    root_logger.addHandler(file_handler)

    # Console handler - INFO level (reduced output)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))  # Console uses specified level
    console_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    root_logger.addHandler(console_handler)

    # Reduce verbosity of HTTP request logs
    logging.getLogger("httpx").setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.info(f"[SETUP] Logging to: {log_file}")

    return logger, log_file


def _setup_mode_logging(
    mode: str,
    hub_name: str,
    namespace: str,
    args: argparse.Namespace,
    log_filename: str
) -> tuple[logging.Logger, Path]:
    """
    Common logging setup for all modes.

    Args:
        mode: Mode name ("local", "dist")
        hub_name: Hub platform name
        namespace: Namespace/username
        args: Parsed command line arguments
        log_filename: Log filename

    Returns:
        Tuple of (logger, log_folder_path)
    """
    folder_name = _generate_log_folder_name(mode, hub_name, namespace)
    log_folder = Path("logs/hub_upload") / folder_name

    logger, _ = setup_logging(
        log_level=args.log_level,
        log_folder=log_folder,
        log_filename=log_filename
    )

    return logger, log_folder


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Upload RoboCoin datasets to remote hubs (HuggingFace/ModelScope).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
    # Local upload mode (single machine, default)
    python scripts/hub_upload/upload2hub.py \\
        --config configs/upload.yaml

    # Server mode (start task distribution server)
    python scripts/hub_upload/upload2hub.py --server \\
        --config configs/upload.yaml \\
        --host 0.0.0.0 \\
        --port 2100

    # Client mode (connect to server and process tasks)
    python scripts/hub_upload/upload2hub.py --client \\
        --config configs/upload.yaml \\
        --host 127.0.0.1 \\
        --port 2100 \\
        --num-clients 4

    # All options for local mode
    python scripts/hub_upload/upload2hub.py \\
        --config configs/upload.yaml \\
        --log-level DEBUG \\
        --force \\
        --readme-only
        """
    )

    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite existing repositories without prompting"
    )

    parser.add_argument(
        "--readme-only",
        action="store_true",
        help="Only update README files without uploading dataset files"
    )

    parser.add_argument(
        "--local",
        action="store_true",
        help="Use local upload mode (single machine, no server/client architecture)"
    )

    parser.add_argument(
        "--server",
        action="store_true",
        help="Start server mode for distributed upload (requires --host and --port)"
    )

    parser.add_argument(
        "--client",
        action="store_true",
        help="Start client mode to connect to upload server (requires --host and --port)"
    )

    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address for server/client mode (default: 0.0.0.0 for server, 127.0.0.1 for client)"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=2100,
        help="Port number for server/client mode (default: 2100)"
    )

    parser.add_argument(
        "--num-clients",
        type=int,
        default=1,
        help="Number of client processes to spawn in client mode (default: 1)"
    )

    parser.add_argument(
        "--heartbeat-interval",
        type=float,
        default=30.0,
        help="Heartbeat interval in seconds for server/client mode (default: 30.0)"
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=90.0,
        help="Timeout in seconds for server/client heartbeat (default: 90.0)"
    )

    parser.add_argument(
        "--request-timeout",
        type=float,
        default=-1,
        help=(
            "Timeout in seconds when waiting for a task from server in client mode. "
            "Use <= 0 to wait indefinitely for tasks (default: -1)."
        ),
    )

    return parser.parse_args()


def run_local_mode(config: UploadConfig, args: argparse.Namespace, logger: logging.Logger) -> None:
    """
    Run the upload process in local single-machine mode.

    This function processes upload tasks directly from the database, one by one,
    using the UploadLocal class which handles database status updates.

    Args:
        config: UploadConfig object containing all configuration parameters
        args: Parsed command line arguments
        logger: Logger instance for audit logging

    Expected Behavior:
        - Processes all PENDING upload tasks from database
        - Updates database status for each task (PROCESSING -> COMPLETED/FAILED)
        - Logs all operations with [stage] identifiers
        - Handles errors gracefully and continues with next task
    """
    # Validate required configuration
    if not config.pg_cfg_path:
        raise ValueError("pg_cfg_path is required for local mode")
    
    hub_name = _normalize_hub_name(config.hub_name)
    
    # Get namespace based on hub
    if hub_name == "modelscope":
        namespace = config.ms_namespace or DEFAULT_PLATFORM_NAME
    else:
        namespace = config.hf_namespace or DEFAULT_PLATFORM_NAME

    # Setup logging with timestamped folder for local mode
    logger, log_folder = _setup_mode_logging("local", hub_name, namespace, args, "local.log")

    logger.info("=" * 80)
    logger.info("[LOCAL_MODE] Starting local upload process")
    logger.info("=" * 80)
    logger.info(f"[LOCAL_MODE] Hub: {hub_name}")
    logger.info(f"[LOCAL_MODE] Namespace: {namespace}")
    logger.info(f"[LOCAL_MODE] Database config: {config.pg_cfg_path}")
    logger.info(f"[LOCAL_MODE] Force overwrite: {config.force_overwrite}")
    logger.info(f"[LOCAL_MODE] Readme only: {config.readme_only}")
    logger.info("=" * 80)

    # Initialize database connection
    try:
        database = DatasetDatabase(config.pg_cfg_path)
        logger.info(f"[LOCAL_MODE] Database connection established")
    except Exception as e:
        error_msg = f"[LOCAL_MODE] Failed to connect to database: {e}"
        _log_error(logger, error_msg)
        raise

    # Initialize upload utility
    try:
        uploader = UploadLocal(config)
        logger.info(f"[LOCAL_MODE] Upload utility initialized")
    except Exception as e:
        error_msg = f"[LOCAL_MODE] Failed to initialize upload utility: {e}"
        _log_error(logger, error_msg)
        raise

    # Statistics tracking
    tasks_processed = 0
    tasks_succeeded = 0
    tasks_failed = 0

    # Initial sync: Mark all eligible datasets as PENDING
    logger.info("[LOCAL_MODE] Syncing upload status in database...")
    try:
        with database.with_session() as session:
            _sync_upload_status(
                session=session,
                specific_uuid=None,
                hub_name=hub_name,
                logger=logger,
            )
        logger.info("[LOCAL_MODE] Status sync completed")
    except Exception as e:
        error_msg = f"[LOCAL_MODE] Failed to sync upload status: {e}"
        _log_error(logger, error_msg)
        raise

    # Main processing loop
    logger.info("[LOCAL_MODE] Starting task processing loop...")

    try:
        while True:
            dataset_uuid = None
            try:
                with database.with_session() as session:
                    # Get next task from database
                    dataset_uuid = _gen_one_upload_task(
                        session=session,
                        specific_uuid=None,
                        hub_name=hub_name,
                        logger=logger,
                    )

                    if dataset_uuid is None:
                        # No more tasks available
                        logger.info("[LOCAL_MODE] No more tasks available")
                        break

                    # Get hardlink path
                    hardlink_path = _get_hardlink_path_by_uuid(
                        session=session,
                        dataset_uuid=dataset_uuid,
                        logger=logger,
                    )

                    if hardlink_path is None:
                        error_msg = f"[LOCAL_MODE] Hardlink path not found for dataset {dataset_uuid}"
                        _log_error(logger, error_msg)
                        _mark_upload_failed(
                            session=session,
                            dataset_uuid=dataset_uuid,
                            error_msg=error_msg,
                            hub_name=hub_name,
                            logger=logger,
                        )
                        tasks_processed += 1
                        tasks_failed += 1
                        continue

                    if not hardlink_path.exists():
                        error_msg = f"[LOCAL_MODE] Hardlink path does not exist: {hardlink_path}"
                        _log_error(logger, error_msg)
                        _mark_upload_failed(
                            session=session,
                            dataset_uuid=dataset_uuid,
                            error_msg=error_msg,
                            hub_name=hub_name,
                            logger=logger,
                        )
                        tasks_processed += 1
                        tasks_failed += 1
                        continue

                    # Process upload task
                    logger.info(f"[LOCAL_MODE] Processing task | UUID: {dataset_uuid} | Path: {hardlink_path}")

                    # Upload using UploadLocal (handles database status updates internally)
                    # UploadLocal.upload() will:
                    # 1. Sync status (mark as PENDING if needed)
                    # 2. Generate task (mark as PROCESSING)
                    # 3. Execute upload
                    # 4. Update status (COMPLETED or FAILED)
                    uploader.upload(
                        hardlink_path=hardlink_path,
                        dataset_uuid=dataset_uuid,
                        hub_name=hub_name,
                        pg_session=session,
                    )

                    # Upload succeeded (status already updated by uploader.upload)
                    tasks_processed += 1
                    tasks_succeeded += 1
                    _log_success(logger, f"[LOCAL_MODE] Task completed | UUID: {dataset_uuid}")

            except KeyboardInterrupt:
                logger.info("[LOCAL_MODE] Interrupted by user")
                break
            except Exception as e:
                tasks_processed += 1
                tasks_failed += 1
                error_msg = f"[LOCAL_MODE] Task failed | UUID: {dataset_uuid or 'unknown'} | Error: {e}"
                _log_error(logger, error_msg)
                logger.debug(f"[LOCAL_MODE] Full traceback:", exc_info=True)
                
                # Try to mark as failed in database if we have UUID
                if dataset_uuid:
                    try:
                        with database.with_session() as session:
                            _mark_upload_failed(
                                session=session,
                                dataset_uuid=dataset_uuid,
                                error_msg=str(e),
                                hub_name=hub_name,
                                logger=logger,
                            )
                    except Exception as db_error:
                        logger.error(f"[LOCAL_MODE] Failed to update database status: {db_error}")

                # Continue with next task
                if config.skip_errors:
                    logger.info("[LOCAL_MODE] Continuing with next task (skip_errors=True)")
                    continue
                else:
                    # Stop on error if skip_errors is False
                    raise

    finally:
        # Print summary
        logger.info("=" * 80)
        logger.info("[LOCAL_MODE] SUMMARY")
        logger.info("=" * 80)
        logger.info(f"[LOCAL_MODE] Tasks processed: {tasks_processed}")
        _log_success(logger, f"[LOCAL_MODE] Tasks succeeded: {tasks_succeeded}")
        _log_error(logger, f"[LOCAL_MODE] Tasks failed: {tasks_failed}")
        logger.info("=" * 80)


def run_server_mode(config: UploadConfig, args: argparse.Namespace, logger: logging.Logger) -> None:
    """
    Run the upload server that distributes tasks to clients.

    Args:
        config: UploadConfig object containing all configuration parameters
        args: Parsed command line arguments
        logger: Logger instance for audit logging

    Expected Behavior:
        - Starts WebSocket server for task distribution
        - Distributes upload tasks to connected clients
        - Updates database status based on client results
        - Handles one hub platform per server instance
    """
    # Validate required configuration
    if not config.pg_cfg_path:
        raise ValueError("pg_cfg_path is required for server mode")

    hub_name = _normalize_hub_name(config.hub_name)
    
    # Get namespace based on hub
    if hub_name == "modelscope":
        namespace = config.ms_namespace or DEFAULT_PLATFORM_NAME
    else:
        namespace = config.hf_namespace or DEFAULT_PLATFORM_NAME

    # Setup logging with timestamped folder for distributed mode
    logger, log_folder = _setup_mode_logging("dist", hub_name, namespace, args, "server.log")

    logger.info("=" * 80)
    logger.info("[SERVER_MODE] Starting hub upload server")
    logger.info("=" * 80)
    logger.info(f"[SERVER_MODE] Host: {args.host}")
    logger.info(f"[SERVER_MODE] Port: {args.port}")
    logger.info(f"[SERVER_MODE] Database config: {config.pg_cfg_path}")
    logger.info(f"[SERVER_MODE] Hub: {hub_name}")
    logger.info(f"[SERVER_MODE] Namespace: {namespace}")
    logger.info(f"[SERVER_MODE] Heartbeat interval: {args.heartbeat_interval}s")
    logger.info(f"[SERVER_MODE] Timeout: {args.timeout}s")
    logger.info("=" * 80)

    # Update config with server network settings
    config.server_host = args.host
    config.server_port = args.port
    config.server_heartbeat_interval = args.heartbeat_interval
    config.server_timeout = args.timeout

    # Create and run server
    server = UploadServer(
        cfg=config,
        logger=logger,
    )

    try:
        logger.info("[SERVER_MODE] Server starting...")
        asyncio.run(server.start())
    except KeyboardInterrupt:
        logger.info("[SERVER_MODE] Server interrupted by user")
    finally:
        stats = server.get_statistics()
        logger.info("=" * 80)
        logger.info("[SERVER_MODE] SUMMARY")
        logger.info("=" * 80)
        _log_success(logger, f"[SERVER_MODE] Datasets succeeded: {stats['datasets_succeeded']}")
        _log_error(logger, f"[SERVER_MODE] Datasets failed: {stats['datasets_failed']}")
        logger.info("=" * 80)


def run_client_mode(config: UploadConfig, args: argparse.Namespace, logger: logging.Logger) -> None:
    """
    Run the upload client(s) that connect to the server and process tasks.

    Args:
        config: UploadConfig object containing all configuration parameters
        args: Parsed command line arguments
        logger: Logger instance for audit logging

    Expected Behavior:
        - Connects to server via WebSocket
        - Requests and processes upload tasks
        - Reports results back to server
        - Supports multiple client processes for parallel processing
    """
    hub_name = _normalize_hub_name(config.hub_name)
    
    # Get namespace based on hub
    if hub_name == "modelscope":
        namespace = config.ms_namespace or DEFAULT_PLATFORM_NAME
    else:
        namespace = config.hf_namespace or DEFAULT_PLATFORM_NAME

    # Update config with client network settings
    config.client_host = args.host
    config.client_port = args.port
    config.client_heartbeat_interval = args.heartbeat_interval
    config.client_timeout = args.timeout
    config.request_task_timeout = args.request_timeout if args.request_timeout > 0 else None

    # Setup logging with timestamped folder for distributed mode (shared by all clients)
    logger, log_folder = _setup_mode_logging("dist", hub_name, namespace, args, "client_main.log")

    server_uri = f"ws://{args.host}:{args.port}"

    logger.info("=" * 80)
    logger.info("[CLIENT_MODE] Starting hub upload client(s)")
    logger.info("=" * 80)
    _log_url(logger, f"[CLIENT_MODE] Server URI: {server_uri}")
    logger.info(f"[CLIENT_MODE] Number of clients: {args.num_clients}")
    logger.info(f"[CLIENT_MODE] Hub: {hub_name}")
    logger.info(f"[CLIENT_MODE] Namespace: {namespace}")
    logger.info(f"[CLIENT_MODE] Heartbeat interval: {args.heartbeat_interval}s")
    logger.info(f"[CLIENT_MODE] Log folder: {log_folder}")
    logger.info("=" * 80)

    # Run client(s) - all clients will use the same log folder
    exit_code = run_multi_clients(
        config=config,
        num_clients=args.num_clients,
        request_timeout=args.request_timeout,
        log_dir=log_folder,
        log_level=args.log_level,
    )

    if exit_code != 0:
        _log_error(logger, "[CLIENT_MODE] Client(s) completed with errors")
        sys.exit(exit_code)
    else:
        _log_success(logger, "[CLIENT_MODE] Client(s) completed successfully")


def main() -> None:
    """
    Main entry point for the hub upload CLI.
    """
    # Start timing
    script_start_time = time.time()

    # Parse arguments
    args = parse_arguments()

    # Setup initial logging - will be reconfigured per mode with timestamped folders
    logger, _ = setup_logging(args.log_level)

    try:
        # Validate mode selection
        modes_selected = sum([args.server, args.client, args.local])
        if modes_selected > 1:
            _log_error(logger, "[MAIN] Cannot specify more than one mode: --server, --client, or --local")
            sys.exit(1)

        # Load configuration from YAML file
        if not args.config:
            _log_error(logger, "[MAIN] --config/-c is required")
            sys.exit(1)

        logger.info(f"[MAIN] Loading configuration from: {args.config}")
        config = create_config(args.config)

        # Override config with command line arguments
        if args.force:
            config.force_overwrite = True
        if args.readme_only:
            config.readme_only = True

        # Handle different modes
        if args.server:
            run_server_mode(config, args, logger)
        elif args.client:
            run_client_mode(config, args, logger)
        else:
            # local mode (default)
            run_local_mode(config, args, logger)

        # Calculate total script execution time
        script_elapsed = time.time() - script_start_time
        hours, remainder = divmod(int(script_elapsed), 3600)
        minutes, seconds = divmod(remainder, 60)

        if hours > 0:
            time_str = f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            time_str = f"{minutes}m {seconds}s"
        else:
            time_str = f"{seconds}s"

        logger.info("=" * 80)
        _log_success(logger, f"[MAIN] Script completed successfully in {time_str}")
        logger.info(f"[MAIN] Total execution time: {script_elapsed:.2f}s")
        logger.info("=" * 80)

    except FileNotFoundError as e:
        script_elapsed = time.time() - script_start_time
        _log_error(logger, f"[MAIN] File not found: {e} (after {script_elapsed:.2f}s)")
        sys.exit(1)
    except ValueError as e:
        script_elapsed = time.time() - script_start_time
        _log_error(logger, f"[MAIN] Configuration error: {e} (after {script_elapsed:.2f}s)")
        sys.exit(1)
    except KeyboardInterrupt:
        script_elapsed = time.time() - script_start_time
        logger.warning(f"[MAIN] Upload interrupted by user (after {script_elapsed:.2f}s)")
        sys.exit(1)
    except Exception as e:
        script_elapsed = time.time() - script_start_time
        _log_error(logger, f"[MAIN] Unexpected error: {e} (after {script_elapsed:.2f}s)")
        logger.debug(f"[MAIN] Full traceback:", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
