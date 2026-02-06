"""
RoboCoin Datasets Uploader - Main CLI Entry Point

This script uploads datasets to the hub using a database-driven strategy.
It generates dataset info YAML files and README.md files ON-DEMAND for each dataset
right before uploading, using the hardlink paths from the database.

KEY FEATURES:
- On-demand generation of dataset info YAML files from metadata (per dataset)
- On-demand generation of README.md files from templates (per dataset)
- Upload datasets to HuggingFace or ModelScope
- Database-driven upload tracking
- Works with hardlinks at any location (not restricted to single root_path)
- Supports three modes: local, server, and client

MODES:
    1. Local mode (default, --local): Single machine upload
    2. Server mode (--server): Starts a task distribution server
    3. Client mode (--client): Connects to server and processes tasks

WORKFLOW:
    For each dataset in the database:
    1. Generate dataset_info.yml file from metadata
    2. Generate README.md file from template
    3. Upload dataset to hub (unless --readme-only is specified)

Usage:
    # Local upload mode (single machine, default)
    python scripts/hub_upload/upload2hub.py \
        --config configs/upload.yaml \
        --token YOUR_TOKEN

    # Server mode (distribute tasks to clients)
    python scripts/hub_upload/upload2hub.py --server \
        --config configs/upload.yaml \
        --db-file-path /path/to/datasets.db \
        --token YOUR_TOKEN \
        --name-space YourUsername \
        --host 0.0.0.0 \
        --port 2140

    # Client mode (connect to server and process tasks)
    python scripts/hub_upload/upload2hub.py --client \
        --host 127.0.0.1 \
        --port 2140 \
        --num-clients 4 \
        --config configs/upload.yaml \
        --token YOUR_TOKEN \
        --name-space YourUsername

    # With custom database path
    python scripts/hub_upload/upload2hub.py \
        --config configs/upload.yaml \
        --db-file-path /path/to/datasets_new.db \
        --token YOUR_TOKEN

    # README-only mode (only update README files, no dataset upload)
    python scripts/hub_upload/upload2hub.py \
        --config configs/upload.yaml \
        --readme-only \
        --token YOUR_TOKEN
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

from robocoin_dataset.hub_upload.config.constant import (
    DS_PLATFORM_NAME,
    DatasetsHubEnum,
)
from robocoin_dataset.hub_upload.lerobot.hub_upload_local import (
    upload_datasets_main as upload_datasets_main_local,
)
from robocoin_dataset.hub_upload.lerobot.hub_upload_util import (
    create_upload_config,
    load_config_from_yaml,
)


def _ensure_required_config_fields(config: dict, required_fields: list[str]) -> None:
    """
    Raise ValueError if any required field is missing or empty-ish in config dict.
    """
    missing: list[str] = []
    for field in required_fields:
        value = config.get(field)
        if value is None:
            missing.append(field)
            continue
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"", "null", "none", "default"}:
                missing.append(field)
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Missing required config field(s): {missing_str}")


def _resolve_required_path(
    value: str | Path | None,
    field_name: str,
    *,
    must_be_dir: bool,
) -> Path:
    """
    Resolve and validate a user-provided path.
    """
    if value is None:
        raise ValueError(f"{field_name} is required")

    resolved = Path(value).expanduser().absolute()
    if not resolved.exists():
        raise ValueError(f"{field_name} does not exist: {resolved}")
    if must_be_dir and not resolved.is_dir():
        raise ValueError(f"{field_name} must be a directory: {resolved}")
    if not must_be_dir and not resolved.is_file():
        raise ValueError(f"{field_name} must be a file: {resolved}")
    return resolved


def _normalize_hub_name(value: str | DatasetsHubEnum | None) -> DatasetsHubEnum:
    """
    Convert CLI/config hub_name into a DatasetsHubEnum.
    """
    if isinstance(value, DatasetsHubEnum):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        try:
            return DatasetsHubEnum[normalized]
        except KeyError as exc:
            raise ValueError(
                f"Invalid hub_name '{value}'. Must be one of: "
                f"{', '.join(member.name for member in DatasetsHubEnum)}"
            ) from exc
    if value is None:
        return DatasetsHubEnum.huggingface
    raise ValueError(f"Unsupported hub_name type: {type(value)!r}")


def _prepare_upload_config_dict(config: dict) -> dict:
    """
    Normalize and validate configuration values before constructing the dataclass.
    Note: root_path is now optional for Local mode (not used).
    """
    prepared: dict = dict(config)

    # root_path is optional for Local mode and not needed for readme-only mode
    if prepared.get("root_path") and not prepared.get("readme_only", False):
        root_path = _resolve_required_path(
            prepared.get("root_path"),
            "root_path",
            must_be_dir=True,
        )
        prepared["root_path"] = str(root_path)

    db_file_path = _resolve_required_path(
        prepared.get("db_file_path"),
        "db_file_path",
        must_be_dir=False,
    )
    prepared["db_file_path"] = str(db_file_path)

    namespace = prepared.get("namespace")
    namespace = namespace.strip() if isinstance(namespace, str) else ""
    prepared["namespace"] = namespace or DS_PLATFORM_NAME

    output_path = prepared.get("output_path") or "./dataset_info"
    prepared["output_path"] = str(Path(output_path).expanduser().absolute())

    prepared["skip_missing"] = bool(prepared.get("skip_missing", False))
    prepared["force_overwrite"] = bool(prepared.get("force_overwrite", False))
    prepared["readme_only"] = bool(prepared.get("readme_only", False))

    prepared["hub_name"] = _normalize_hub_name(prepared.get("hub_name"))

    return prepared


def _generate_log_folder_name(
    mode: str,
    hub_name: DatasetsHubEnum | str,
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

    # Normalize hub_name to string
    if isinstance(hub_name, DatasetsHubEnum):
        hub_str = hub_name.value
    else:
        hub_str = str(hub_name)

    # Sanitize namespace (remove special characters that might cause issues in folder names)
    namespace_safe = namespace.replace("/", "_").replace("\\", "_").replace(" ", "_")

    return f"{mode}_{timestamp}_{hub_str}_{namespace_safe}"


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
    # Log to file, print to console using tqdm.write
    logger.info(f"📝 Logging to: {log_file}")
    from tqdm import tqdm
    tqdm.write(f"📝 Logging to: {log_file}")

    return logger, log_file


def _setup_mode_logging(
    mode: str,
    hub_name: DatasetsHubEnum | str,
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
        description="Upload RoboCoin datasets to remote hubs (HuggingFace/ModelScope). "
                    "Always generates info YAML and README files before uploading.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Local upload mode (single machine, default)
  python scripts/hub_upload/upload2hub.py \\
      --config configs/upload.yaml \\
      --token YOUR_TOKEN

  # Server mode (start task distribution server)
  python scripts/hub_upload/upload2hub.py --server \\
      --config configs/upload.yaml \\
      --db-file-path /path/to/datasets.db \\
      --token YOUR_TOKEN \\
      --name-space YourUsername \\
      --host 0.0.0.0 \\
      --port 2140

  # Client mode (connect to server and process tasks)
  python scripts/hub_upload/upload2hub.py --client \\
      --config configs/upload.yaml \\
      --token YOUR_TOKEN \\
      --name-space YourUsername \\
      --host 127.0.0.1 \\
      --port 2140 \\
      --num-clients 4

  # All options for local mode
  python scripts/hub_upload/upload2hub.py \\
      --config configs/upload.yaml \\
      --info-output-path ./outputs/infos \\
      --token YOUR_TOKEN \\
      --name-space YourUsername \\
      --db-file-path /path/to/db.db \\
      --log-level DEBUG \\
      --skip-missing \\
      --force

  # README-only mode (only update README files, no dataset upload)
  python scripts/hub_upload/upload2hub.py \\
      --config configs/upload.yaml \\
      --readme-only \\
      --token YOUR_TOKEN
        """
    )

    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to YAML configuration file"
    )

    parser.add_argument(
        "--token",
        type=str,
        help="Authentication token for the hub platform (if not provided, will prompt)"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)"
    )

    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip datasets with missing hardlinks instead of aborting"
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
        "--db-file-path",
        type=str,
        help="Override database file path from config"
    )

    parser.add_argument(
        "--info-output-path",
        type=str,
        default=None,
        help="Output path for generated dataset info files (default: ./dataset_info)"
    )

    parser.add_argument(
        "--name-space",
        type=str,
        help="Namespace (username) on the hub platform where datasets will be uploaded. "
             "If not provided, uses default value from constant.yml (DS_PLATFORM_NAME)"
    )

    parser.add_argument(
        "--hub",
        type=str,
        choices=["huggingface", "modelscope"],
        default="huggingface",
        help="Target hub platform for client mode (default: huggingface)"
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
        default=2140,
        help="Port number for server/client mode (default: 2100)"
    )

    parser.add_argument(
        "--num-clients",
        type=int,
        default=1,
        help="Number of client processes to spawn in client mode (default: 1)"
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of worker threads per client (default: 1, currently not used)"
    )

    parser.add_argument(
        "--heartbeat-interval",
        type=float,
        default=30.0,
        help="Heartbeat interval in seconds for server/client mode (default: 30.0)"
    )

    parser.add_argument(
        "--request-timeout",
        type=float,
        default=-1,
        help=(
            "Timeout in seconds when waiting for a task from server in client mode. "
            "Use <= 0 to wait indefinitely for tasks (default: 15.0)."
        ),
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=90.0,
        help="Timeout in seconds for server/client heartbeat (default: 300.0)"
    )

    return parser.parse_args()


def run_server_mode(config: dict, args: argparse.Namespace, logger: logging.Logger) -> None:
    """
    Run the upload server that distributes tasks to clients.

    Args:
        config: Configuration dictionary from YAML
        args: Parsed command line arguments
        logger: Logger instance
    """
    from tqdm import tqdm

    from robocoin_dataset.hub_upload.lerobot.hub_upload_server import HubUploadServer
    from robocoin_dataset.utils.logger import setup_logger

    # Ensure required config fields for server mode
    _ensure_required_config_fields(config, ["db_file_path", "token"])

    hub_name = _normalize_hub_name(config.get("hub_name"))
    db_file_path = _resolve_required_path(
        config.get("db_file_path"),
        "db_file_path",
        must_be_dir=False,
    )

    # Extract client configuration parameters to be sent with tasks
    token = config.get("token", "")
    namespace = config.get("namespace", DS_PLATFORM_NAME)
    output_path = config.get("output_path", "./dataset_info")
    force_overwrite = config.get("force_overwrite", False)
    readme_only = config.get("readme_only", False)

    # Setup logging with timestamped folder for distributed mode
    logger, log_folder = _setup_mode_logging("dist", hub_name, namespace, args, "server.log")

    # Create summary logger for server status updates in the same folder
    summary_logger = setup_logger(
        name="hub_upload_server_summary",
        log_dir=log_folder,
        level=logging.DEBUG,  # File gets DEBUG level
        console_output=False,
    )

    logger.info("=" * 80)
    logger.info("🖥️  STARTING HUB UPLOAD SERVER")
    logger.info("=" * 80)
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info(f"Database: {db_file_path}")
    logger.info(f"Hub: {hub_name.value}")
    logger.info(f"Token: {'***' + token[-4:] if len(token) > 4 else 'Not set'}")
    logger.info(f"Namespace: {namespace}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Force overwrite: {force_overwrite}")
    logger.info(f"Heartbeat interval: {args.heartbeat_interval}s")
    logger.info(f"Timeout: {args.timeout}s")
    logger.info("=" * 80)
    tqdm.write("=" * 80)
    tqdm.write("🖥️  STARTING HUB UPLOAD SERVER")
    tqdm.write("=" * 80)
    tqdm.write(f"Host: {args.host}")
    tqdm.write(f"Port: {args.port}")
    tqdm.write(f"Database: {db_file_path}")
    tqdm.write(f"Hub: {hub_name.value}")
    tqdm.write(f"Namespace: {namespace}")
    tqdm.write("=" * 80)

    # Create and run server with client configuration
    server = HubUploadServer(
        db_file_path=db_file_path,
        summary_logger=summary_logger,
        hub_name=hub_name,
        token=token,
        namespace=namespace,
        output_path=output_path,
        force_overwrite=force_overwrite,
        readme_only=readme_only,
        host=args.host,
        port=args.port,
        heartbeat_interval=args.heartbeat_interval,
        timeout=args.timeout,
        logger=logger,
    )

    try:
        logger.info("🚀 Server starting...")
        tqdm.write("🚀 Server starting...")
        asyncio.run(server.start())
    except KeyboardInterrupt:
        logger.info("\n⚠️  Server interrupted by user")
        tqdm.write("\n⚠️  Server interrupted by user")
    finally:
        stats = server.get_statistics()
        logger.info("=" * 80)
        logger.info("📊 SERVER SUMMARY")
        logger.info("=" * 80)
        logger.info(f"✅ Datasets succeeded: {stats['datasets_succeeded']}")
        logger.info(f"❌ Datasets failed: {stats['datasets_failed']}")
        logger.info("=" * 80)
        tqdm.write("=" * 80)
        tqdm.write("📊 SERVER SUMMARY")
        tqdm.write("=" * 80)
        tqdm.write(f"✅ Datasets succeeded: {stats['datasets_succeeded']}")
        tqdm.write(f"❌ Datasets failed: {stats['datasets_failed']}")
        tqdm.write("=" * 80)


def run_client_mode(config: dict, args: argparse.Namespace, logger: logging.Logger) -> None:
    """
    Run the upload client(s) that connect to the server and process tasks.

    Args:
        config: Configuration dictionary from YAML
        args: Parsed command line arguments
        logger: Logger instance
    """
    from tqdm import tqdm

    from robocoin_dataset.hub_upload.lerobot.hub_upload_client import run_multi_clients

    hub_name = _normalize_hub_name(config.get("hub_name"))
    token = config.get("token", "")
    namespace = config.get("namespace", DS_PLATFORM_NAME)
    output_path = config.get("output_path", "./dataset_info")
    force_overwrite = config.get("force_overwrite", False)
    readme_only = config.get("readme_only", False)

    server_uri = f"ws://{args.host}:{args.port}"

    # Setup logging with timestamped folder for distributed mode (shared by all clients)
    logger, log_folder = _setup_mode_logging("dist", hub_name, namespace, args, "client_main.log")

    logger.info("=" * 80)
    logger.info("🔌 STARTING HUB UPLOAD CLIENT(S)")
    logger.info("=" * 80)
    logger.info(f"Server URI: {server_uri}")
    logger.info(f"Number of clients: {args.num_clients}")
    logger.info(f"Hub: {hub_name.value}")
    logger.info(f"Namespace: {namespace}")
    logger.info(f"Heartbeat interval: {args.heartbeat_interval}s")
    logger.info(f"Log folder: {log_folder}")
    logger.info("=" * 80)
    tqdm.write("=" * 80)
    tqdm.write("🔌 STARTING HUB UPLOAD CLIENT(S)")
    tqdm.write("=" * 80)
    tqdm.write(f"Server URI: {server_uri}")
    tqdm.write(f"Number of clients: {args.num_clients}")
    tqdm.write(f"Hub: {hub_name.value}")
    tqdm.write(f"Namespace: {namespace}")
    tqdm.write(f"Log folder: {log_folder}")
    tqdm.write("=" * 80)

    # Run client(s) - all clients will use the same log folder
    exit_code = run_multi_clients(
        server_uri=server_uri,
        num_clients=args.num_clients,
        hub_name=hub_name,
        token=token,
        namespace=namespace,
        output_path=output_path,
        force_overwrite=force_overwrite,
        readme_only=readme_only,
        heartbeat_interval=args.heartbeat_interval,
        request_timeout=args.request_timeout,
        log_dir=log_folder,
        log_level=args.log_level,
    )

    if exit_code != 0:
        logger.error("❌ Client(s) completed with errors")
        tqdm.write("❌ Client(s) completed with errors")
        sys.exit(exit_code)
    else:
        logger.info("✅ Client(s) completed successfully")
        tqdm.write("✅ Client(s) completed successfully")


def run_local_mode(config: dict, args: argparse.Namespace, logger: logging.Logger) -> None:
    """
    Run the upload process in local single-machine mode.

    Args:
        config: Configuration dictionary from YAML
        args: Parsed command line arguments
        logger: Logger instance
    """
    from tqdm import tqdm

    prepared_config_dict = _prepare_upload_config_dict(config)

    # Ensure required config fields (skip token check for readme-only mode)
    required_fields = ["db_file_path"]
    if not prepared_config_dict.get("readme_only", False):
        required_fields.append("token")
    _ensure_required_config_fields(config, required_fields)

    # Setup logging with timestamped folder for local mode
    hub_name = prepared_config_dict.get("hub_name", DatasetsHubEnum.huggingface)
    namespace = prepared_config_dict.get("namespace", DS_PLATFORM_NAME)
    logger, log_folder = _setup_mode_logging("local", hub_name, namespace, args, "local.log")

    # Create upload config
    config_obj = create_upload_config(prepared_config_dict)

    # Note: root_path is now optional for Local mode (not used)

    # Check if we're in readme-only mode
    if prepared_config_dict.get("readme_only", False):
        # Log information about readme-only mode
        logger.info("=" * 80)
        logger.info("📝 README-ONLY MODE: Only updating README files (no dataset upload)")
        logger.info("   YAML and README files will be generated on-demand for each dataset")
        logger.info("=" * 80)
        tqdm.write("=" * 80)
        tqdm.write("📝 README-ONLY MODE: Only updating README files (no dataset upload)")
        tqdm.write("   YAML and README files will be generated on-demand for each dataset")
        tqdm.write("=" * 80)

        # Start readme-only process
        logger.info("=" * 80)
        logger.info("🚀 Starting README-ONLY process with on-demand file generation")
        tqdm.write("🚀 Starting README-ONLY process with on-demand file generation")
        logger.info("=" * 80)
        tqdm.write("=" * 80)
    else:
        # Log information about full upload mode
        logger.info("=" * 80)
        logger.info("📝 YAML and README files will be generated on-demand for each dataset")
        logger.info("=" * 80)
        tqdm.write("=" * 80)
        tqdm.write("📝 YAML and README files will be generated on-demand for each dataset")
        tqdm.write("=" * 80)

        # Start upload process
        logger.info("=" * 80)
        logger.info("🚀 Starting LOCAL upload process with on-demand file generation")
        tqdm.write("🚀 Starting LOCAL upload process with on-demand file generation")
        logger.info("=" * 80)
        tqdm.write("=" * 80)

    # Upload datasets
    upload_datasets_main_local(config_obj, logger)


def main() -> None:
    """
    Main entry point for the hub upload CLI.
    """
    import time

    from tqdm import tqdm

    # Start timing
    script_start_time = time.time()

    # Parse arguments
    args = parse_arguments()

    # Setup logging - will be reconfigured per mode with timestamped folders
    logger, _ = setup_logging(args.log_level)

    try:
        # Validate mode selection
        modes_selected = sum([args.server, args.client, args.local])
        if modes_selected > 1:
            logger.error("❌ Cannot specify more than one mode: --server, --client, or --local")
            tqdm.write("❌ Cannot specify more than one mode: --server, --client, or --local")
            sys.exit(1)

        # Load configuration
        config_dict = {}
        if args.config:
            logger.info(f"Loading configuration from: {args.config}")
            config_dict = load_config_from_yaml(args.config)
        elif args.client:
            #### client mode don't need to load configuration file #####
            #### cause server will send the configuration to client ####
            logger.info("Using default configuration for client mode")
            config_dict.setdefault("hub_name", args.hub)
            config_dict.setdefault("namespace", DS_PLATFORM_NAME)
            config_dict.setdefault("output_path", "./dataset_info")
            config_dict.setdefault("force_overwrite", False)
        else:
            logger.error("❌ --config/-c is required for server and local modes")
            tqdm.write("❌ --config/-c is required for server and local modes")
            sys.exit(1)

        # Override config
        if args.skip_missing:
            config_dict["skip_missing"] = True
        if args.force:
            config_dict["force_overwrite"] = True
        if args.readme_only:
            config_dict["readme_only"] = True
        if args.db_file_path:
            config_dict["db_file_path"] = args.db_file_path
        if args.name_space:
            config_dict["namespace"] = args.name_space
        if args.info_output_path:
            config_dict["output_path"] = args.info_output_path
        if args.token:
            config_dict["token"] = args.token
        if hasattr(args, 'hub') and args.client:
            # Only set hub_name for client mode from command line
            config_dict["hub_name"] = args.hub

        # Handle different modes
        if args.server:
            run_server_mode(config_dict, args, logger)
        elif args.client:
            run_client_mode(config_dict, args, logger)
        else:
            # local mode (default)
            run_local_mode(config_dict, args, logger)

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
        logger.info(f"✅ Script completed successfully in {time_str}")
        logger.info(f"Total execution time: {script_elapsed:.2f}s")
        logger.info("=" * 80)
        tqdm.write("=" * 80)
        tqdm.write(f"✅ Script completed successfully in {time_str}")
        tqdm.write("=" * 80)

    except FileNotFoundError as e:
        script_elapsed = time.time() - script_start_time
        logger.error(f"❌ File not found: {e} (after {script_elapsed:.2f}s)")
        tqdm.write(f"❌ File not found: {e}")
        sys.exit(1)
    except ValueError as e:
        script_elapsed = time.time() - script_start_time
        logger.error(f"❌ Configuration error: {e} (after {script_elapsed:.2f}s)")
        tqdm.write(f"❌ Configuration error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        script_elapsed = time.time() - script_start_time
        logger.warning(f"\n⚠️  Upload interrupted by user (after {script_elapsed:.2f}s)")
        tqdm.write("\n⚠️  Upload interrupted by user")
        sys.exit(1)
    except Exception as e:
        script_elapsed = time.time() - script_start_time
        logger.error(f"❌ Unexpected error: {e} (after {script_elapsed:.2f}s)", exc_info=True)
        tqdm.write(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
