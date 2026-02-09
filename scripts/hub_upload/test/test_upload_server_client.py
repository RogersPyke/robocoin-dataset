"""
Test script for hub upload server-client and local modes.

This script creates fake database entries, generates minimal empty files,
and tests both server-client mode and local mode upload for both MS and HF platforms.
After testing, it restores the database to its initial state.

Purpose:
    - Test the server-client mode upload functionality
    - Test the local mode upload functionality
    - Verify database integration with upload operations
    - Ensure proper cleanup after testing
    - Print commands being executed for debugging

Dependencies:
    - robocoin_dataset.database: Database connection and models
    - robocoin_dataset.hub_upload.upload: Upload modules
    - subprocess: For running server and client processes
    - tempfile: For creating temporary test files
    - logging: For audit logging

Usage:
    # Set environment variables for tokens
    export HF_TOKEN="your_huggingface_token"
    export MS_TOKEN="your_modelscope_token"
    
    # Run the test
    python scripts/hub_upload/test/test_upload_server_client.py \\
        --pg-config db/postgresql_config.yaml \\
        --hf-config src/robocoin_dataset/hub_upload/config/hf.yaml \\
        --ms-config src/robocoin_dataset/hub_upload/config/ms.yaml

Expected Input:
    - pg_config_path: Path to PostgreSQL configuration YAML file
    - hf_config_path: Path to HuggingFace upload configuration YAML file
    - ms_config_path: Path to ModelScope upload configuration YAML file
    - num_test_datasets: Number of fake datasets to create (default: 3)
    - test_timeout: Timeout in seconds for server-client test (default: 300)

Expected Output:
    - Test results logged to log directory
    - Database restored to initial state after testing
    - Exit code 0 on success, non-zero on failure
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
import yaml
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

# ANSI color codes for terminal output
ANSI_RED = "\033[91m"
ANSI_GREEN = "\033[92m"
ANSI_BLUE = "\033[94m"
ANSI_YELLOW = "\033[93m"
ANSI_RESET = "\033[0m"

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / "src"))

from robocoin_dataset.database.database import DatasetDatabase
from robocoin_dataset.database.models import (
    DatasetDB,
    DatasetHardLinkDB,
    TaskStatus,
)


def _colorize(text: str, color: str, use_color: bool = True) -> str:
    """
    Apply ANSI color code to text if terminal supports colors.

    Args:
        text: Text to colorize
        color: ANSI color code (e.g., ANSI_RED, ANSI_GREEN, ANSI_BLUE)
        use_color: Whether to apply color (default: True)

    Returns:
        Colorized text string
    """
    if not use_color:
        return text
    if os.getenv("TERM") not in (None, "dumb") and os.getenv("NO_COLOR") is None:
        return f"{color}{text}{ANSI_RESET}"
    return text


def _log_success(logger: logging.Logger, message: str) -> None:
    """Log success message with green color."""
    logger.info(_colorize(message, ANSI_GREEN))


def _log_error(logger: logging.Logger, message: str) -> None:
    """Log error message with red color."""
    logger.error(_colorize(message, ANSI_RED))


def _log_warning(logger: logging.Logger, message: str) -> None:
    """Log warning message with yellow color."""
    logger.warning(_colorize(message, ANSI_YELLOW))


def _log_info(logger: logging.Logger, message: str) -> None:
    """Log info message with blue color."""
    logger.info(_colorize(message, ANSI_BLUE))


def setup_logging(log_level: str = "INFO") -> Tuple[logging.Logger, Path]:
    """
    Setup logging with timestamped log file.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Tuple of (logger instance, log directory path)
    """
    # Create log directory
    log_dir = project_root / "logs" / "hub_upload_test"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Generate timestamp in UTC+8 format (YYYYMMDDHHMMSS)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    script_name = Path(__file__).stem
    log_file = log_dir / f"{script_name}_{timestamp}.log"

    # Configure logger
    logger = logging.getLogger("test_upload_server_client")
    logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers
    logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    logger.info(f"[SETUP] Log file: {log_file}")
    return logger, log_dir


def create_test_files(
    base_dir: Path, num_files: int, logger: logging.Logger, timestamp: str
) -> Path:
    """
    Create a test folder with multiple files inside.

    Args:
        base_dir: Base directory to create test folder in
        num_files: Number of files to create inside the folder
        logger: Logger instance
        timestamp: Timestamp string for folder naming

    Returns:
        Path to the created test folder
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Create folder with name test_timestamp
    test_folder_name = f"test_{timestamp}"
    test_folder = base_dir / test_folder_name
    test_folder.mkdir(parents=True, exist_ok=True)

    logger.info(f"[CREATE_TEST_FILES] Creating {num_files} test files in folder {test_folder}")

    # Create multiple files inside the folder
    for i in range(num_files):
        # Create a minimal file (just a few bytes)
        file_path = test_folder / f"test_file_{i:03d}.txt"
        file_path.write_text(f"Test file {i}\n", encoding="utf-8")

    # Create a subdirectory with more files
    subdir = test_folder / "subdir"
    subdir.mkdir(exist_ok=True)
    for i in range(num_files // 2):
        file_path = subdir / f"sub_file_{i:03d}.txt"
        file_path.write_text(f"Sub file {i}\n", encoding="utf-8")

    _log_success(
        logger,
        f"[CREATE_TEST_FILES] Created folder {test_folder} with {num_files + num_files // 2} files",
    )
    return test_folder


def create_fake_datasets(
    session,
    num_datasets: int,
    hardlink_paths: List[Path],
    logger: logging.Logger,
) -> List[Tuple[str, Path]]:
    """
    Create fake dataset entries in database.

    Args:
        session: SQLAlchemy session instance
        num_datasets: Number of fake datasets to create
        hardlink_paths: List of hardlink paths to assign to datasets
        logger: Logger instance

    Returns:
        List of tuples (dataset_uuid, hardlink_path) for created datasets
    """
    created_datasets = []

    logger.info(f"[CREATE_FAKE_DATASETS] Creating {num_datasets} fake dataset entries")

    for i in range(num_datasets):
        # Generate unique UUID for dataset
        dataset_uuid = f"test-{uuid.uuid4().hex[:12]}"
        dataset_name = f"test_dataset_{i:03d}"

        # Get hardlink path (cycle through if more datasets than paths)
        hardlink_path = hardlink_paths[i % len(hardlink_paths)]

        # Create DatasetDB entry
        dataset = DatasetDB(
            dataset_name=dataset_name,
            dataset_uuid=dataset_uuid,
            device_model="test_device",
            device_model_version="v1.0",
            end_effector_type="test_effector",
            data_path=str(hardlink_path),
            convert_path=str(hardlink_path),
            # Set pre-stage status to COMPLETED to allow upload
            data_loader_detection_status=TaskStatus.COMPLETED,
            data_loader_detection_version=1,
            # Set upload status to PENDING
            ms_upload_status=TaskStatus.PENDING,
            ms_upload_version=0,
            ms_upload_version_ps=0,
            huggingface_upload_status=TaskStatus.PENDING,
            huggingface_upload_version=0,
            huggingface_upload_version_ps=0,
        )
        session.add(dataset)

        # Create DatasetHardLinkDB entry
        hardlink_entry = DatasetHardLinkDB(
            dataset_uuid=dataset_uuid,
            hard_link_path=str(hardlink_path),
        )
        session.add(hardlink_entry)

        created_datasets.append((dataset_uuid, hardlink_path))
        logger.debug(
            f"[CREATE_FAKE_DATASETS] Created dataset {dataset_uuid} with path {hardlink_path}"
        )

    session.commit()
    _log_success(
        logger, f"[CREATE_FAKE_DATASETS] Created {len(created_datasets)} dataset entries"
    )
    return created_datasets


def save_database_state(
    session, logger: logging.Logger
) -> List[Tuple[str, dict]]:
    """
    Save current database state for restoration.

    Args:
        session: SQLAlchemy session instance
        logger: Logger instance

    Returns:
        List of tuples (dataset_uuid, dataset_dict) for all test datasets
    """
    logger.info("[SAVE_DATABASE_STATE] Saving database state")

    # Query all test datasets (those starting with "test-")
    test_datasets = (
        session.query(DatasetDB)
        .filter(DatasetDB.dataset_uuid.like("test-%"))
        .all()
    )

    saved_state = []
    for dataset in test_datasets:
        dataset_dict = {
            "dataset_name": dataset.dataset_name,
            "dataset_uuid": dataset.dataset_uuid,
            "device_model": dataset.device_model,
            "device_model_version": dataset.device_model_version,
            "end_effector_type": dataset.end_effector_type,
            "data_path": dataset.data_path,
            "convert_path": dataset.convert_path,
            "data_loader_detection_status": dataset.data_loader_detection_status,
            "data_loader_detection_version": dataset.data_loader_detection_version,
            "ms_upload_status": dataset.ms_upload_status,
            "ms_upload_version": dataset.ms_upload_version,
            "ms_upload_version_ps": dataset.ms_upload_version_ps,
            "huggingface_upload_status": dataset.huggingface_upload_status,
            "huggingface_upload_version": dataset.huggingface_upload_version,
            "huggingface_upload_version_ps": dataset.huggingface_upload_version_ps,
        }
        saved_state.append((dataset.dataset_uuid, dataset_dict))

    logger.info(f"[SAVE_DATABASE_STATE] Saved state for {len(saved_state)} datasets")
    return saved_state


def restore_database_state(
    session, saved_state: List[Tuple[str, dict]], logger: logging.Logger
) -> None:
    """
    Restore database to initial state by deleting test entries.

    Args:
        session: SQLAlchemy session instance
        saved_state: List of tuples (dataset_uuid, dataset_dict) from save_database_state
        logger: Logger instance
    """
    logger.info("[RESTORE_DATABASE_STATE] Restoring database to initial state")

    # Delete all test datasets and their hardlink entries
    test_uuids = [uuid for uuid, _ in saved_state]

    # Delete DatasetHardLinkDB entries
    from robocoin_dataset.database.models import DatasetHardLinkDB

    hardlink_deleted = (
        session.query(DatasetHardLinkDB)
        .filter(DatasetHardLinkDB.dataset_uuid.in_(test_uuids))
        .delete(synchronize_session=False)
    )

    # Delete DatasetDB entries
    datasets_deleted = (
        session.query(DatasetDB)
        .filter(DatasetDB.dataset_uuid.in_(test_uuids))
        .delete(synchronize_session=False)
    )

    session.commit()

    _log_success(
        logger,
        f"[RESTORE_DATABASE_STATE] Deleted {datasets_deleted} datasets and {hardlink_deleted} hardlink entries",
    )


def run_server_client_test(
    config_path: Path,
    hub_name: str,
    test_timeout: int,
    logger: logging.Logger,
) -> bool:
    """
    Run server-client mode upload test.

    Args:
        config_path: Path to upload configuration YAML file
        hub_name: Hub name ("huggingface" or "modelscope")
        test_timeout: Timeout in seconds for the test
        logger: Logger instance

    Returns:
        True if test succeeded, False otherwise
    """
    logger.info(f"[RUN_SERVER_CLIENT_TEST] Starting {hub_name} hub test")

    # Initialize variables
    temp_config_path = None
    original_config_path = config_path

    # Load configuration and modify namespace
    try:
        with open(original_config_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        
        # Force set namespace based on hub
        if hub_name in ("huggingface", "hf"):
            config_dict["hf_namespace"] = "RogersPyke"
        elif hub_name in ("modelscope", "ms"):
            config_dict["ms_namespace"] = "rogerspyke"
        
        server_port = config_dict.get("server_port", 2100)
        client_port = config_dict.get("client_port", server_port)
        
        # Write modified config to temporary file
        temp_config_path = original_config_path.parent / f"{original_config_path.stem}_test_{hub_name}.yaml"
        with open(temp_config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
        config_path = temp_config_path
        logger.debug(f"[RUN_SERVER_CLIENT_TEST] Created temp config: {config_path}")
        
    except Exception as e:
        _log_error(logger, f"[RUN_SERVER_CLIENT_TEST] Config load failed: {e}")
        logger.debug(f"[RUN_SERVER_CLIENT_TEST] Traceback:\n{traceback.format_exc()}")
        server_port = 2100 if hub_name in ("huggingface", "hf") else 2101
        client_port = server_port

    # Get token from environment variable
    token_env_var = "HF_TOKEN" if hub_name in ("huggingface", "hf") else "MS_TOKEN"
    token = os.getenv(token_env_var)
    if not token:
        _log_error(logger, f"[RUN_SERVER_CLIENT_TEST] {token_env_var} not set")
        return False

    # Script path
    upload_script = project_root / "scripts" / "hub_upload" / "upload2hub.py"

    # Start server process
    logger.info(f"[RUN_SERVER_CLIENT_TEST] Starting server (port {server_port})")
    server_cmd = [
        sys.executable,
        str(upload_script),
        "--server",
        "--config",
        str(config_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(server_port),
        "--log-level",
        "INFO",
    ]

    server_process = subprocess.Popen(
        server_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, token_env_var: token},
    )

    # Wait a bit for server to start
    time.sleep(3)

    # Check if server is still running
    if server_process.poll() is not None:
        stdout, stderr = server_process.communicate()
        server_stdout = stdout.decode("utf-8", errors="ignore") if stdout else ""
        server_stderr = stderr.decode("utf-8", errors="ignore") if stderr else ""
        _log_error(logger, f"[RUN_SERVER_CLIENT_TEST] Server failed (exit {server_process.returncode})")
        if server_stderr:
            logger.error(f"[RUN_SERVER_CLIENT_TEST] Server error:\n{server_stderr}")
        return False

    # Start client process
    logger.info(f"[RUN_SERVER_CLIENT_TEST] Starting client (port {client_port})")
    client_cmd = [
        sys.executable,
        str(upload_script),
        "--client",
        "--config",
        str(config_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(client_port),
        "--num-clients",
        "1",
        "--log-level",
        "INFO",
    ]

    client_process = subprocess.Popen(
        client_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, token_env_var: token},
    )

    # Wait for client to complete or timeout
    try:
        stdout, stderr = client_process.communicate(timeout=test_timeout)
        client_exit_code = client_process.returncode
        client_stdout = stdout.decode("utf-8", errors="ignore") if stdout else ""
        client_stderr = stderr.decode("utf-8", errors="ignore") if stderr else ""
        
        if client_exit_code == 0:
            _log_success(logger, "[RUN_SERVER_CLIENT_TEST] Client completed")
        else:
            _log_error(logger, f"[RUN_SERVER_CLIENT_TEST] Client failed (exit {client_exit_code})")
            if client_stderr:
                logger.error(f"[RUN_SERVER_CLIENT_TEST] Error:\n{client_stderr}")

    except subprocess.TimeoutExpired:
        _log_warning(logger, f"[RUN_SERVER_CLIENT_TEST] Timeout after {test_timeout}s")
        client_process.kill()
        stdout, stderr = client_process.communicate()
        client_stderr = stderr.decode("utf-8", errors="ignore") if stderr else ""
        client_exit_code = 1
        if client_stderr:
            logger.error(f"[RUN_SERVER_CLIENT_TEST] Error before timeout:\n{client_stderr}")

    # Stop server
    try:
        server_process.terminate()
        server_process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        server_process.kill()
        server_process.wait()
    except Exception as e:
        logger.debug(f"[RUN_SERVER_CLIENT_TEST] Server stop error: {e}")

    # Cleanup temporary config file
    if temp_config_path and temp_config_path.exists():
        try:
            temp_config_path.unlink()
        except Exception:
            pass

    return client_exit_code == 0


def run_local_test(
    config_path: Path,
    hub_name: str,
    test_timeout: int,
    logger: logging.Logger,
) -> bool:
    """
    Run local mode upload test.

    Args:
        config_path: Path to upload configuration YAML file
        hub_name: Hub name ("huggingface" or "modelscope")
        test_timeout: Timeout in seconds for the test
        logger: Logger instance

    Returns:
        True if test succeeded, False otherwise
    """
    logger.info(f"[RUN_LOCAL_TEST] Starting {hub_name} hub test")

    # Initialize variables
    temp_config_path = None
    original_config_path = config_path

    # Load configuration and modify namespace
    try:
        with open(original_config_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        
        # Force set namespace based on hub
        if hub_name in ("huggingface", "hf"):
            config_dict["hf_namespace"] = "RogersPyke"
        elif hub_name in ("modelscope", "ms"):
            config_dict["ms_namespace"] = "rogerspyke"
        
        # Write modified config to temporary file
        temp_config_path = original_config_path.parent / f"{original_config_path.stem}_test_{hub_name}.yaml"
        with open(temp_config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
        config_path = temp_config_path
        logger.debug(f"[RUN_LOCAL_TEST] Created temp config: {config_path}")
        
    except Exception as e:
        _log_error(logger, f"[RUN_LOCAL_TEST] Config load failed: {e}")
        logger.debug(f"[RUN_LOCAL_TEST] Traceback:\n{traceback.format_exc()}")
        return False

    # Get token from environment variable
    token_env_var = "HF_TOKEN" if hub_name in ("huggingface", "hf") else "MS_TOKEN"
    token = os.getenv(token_env_var)
    if not token:
        _log_error(logger, f"[RUN_LOCAL_TEST] {token_env_var} not set")
        return False

    # Script path
    upload_script = project_root / "scripts" / "hub_upload" / "upload2hub.py"

    # Build local mode command
    local_cmd = [
        sys.executable,
        str(upload_script),
        "--local",
        "--config",
        str(config_path),
        "--log-level",
        "INFO",
    ]
    local_process = subprocess.Popen(
        local_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, token_env_var: token},
    )

    # Wait for process to complete or timeout
    try:
        stdout, stderr = local_process.communicate(timeout=test_timeout)
        local_exit_code = local_process.returncode
        local_stderr = stderr.decode("utf-8", errors="ignore") if stderr else ""
        
        if local_exit_code == 0:
            _log_success(logger, "[RUN_LOCAL_TEST] Completed")
        else:
            _log_error(logger, f"[RUN_LOCAL_TEST] Failed (exit {local_exit_code})")
            if local_stderr:
                logger.error(f"[RUN_LOCAL_TEST] Error:\n{local_stderr}")

    except subprocess.TimeoutExpired:
        _log_warning(logger, f"[RUN_LOCAL_TEST] Timeout after {test_timeout}s")
        local_process.kill()
        stdout, stderr = local_process.communicate()
        local_stderr = stderr.decode("utf-8", errors="ignore") if stderr else ""
        local_exit_code = 1
        if local_stderr:
            logger.error(f"[RUN_LOCAL_TEST] Error before timeout:\n{local_stderr}")

    # Cleanup temporary config file
    if temp_config_path and temp_config_path.exists():
        try:
            temp_config_path.unlink()
        except Exception:
            pass

    return local_exit_code == 0


def main() -> int:
    """
    Main entry point for the test script.

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    parser = argparse.ArgumentParser(
        description="Test hub upload server-client mode with fake datasets"
    )
    parser.add_argument(
        "--pg-config",
        type=str,
        required=True,
        help="Path to PostgreSQL configuration YAML file",
    )
    parser.add_argument(
        "--hf-config",
        type=str,
        required=True,
        help="Path to HuggingFace upload configuration YAML file",
    )
    parser.add_argument(
        "--ms-config",
        type=str,
        required=True,
        help="Path to ModelScope upload configuration YAML file",
    )
    parser.add_argument(
        "--num-test-datasets",
        type=int,
        default=3,
        help="Number of fake datasets to create (default: 3)",
    )
    parser.add_argument(
        "--test-timeout",
        type=int,
        default=300,
        help="Timeout in seconds for server-client test (default: 300)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Setup logging
    logger, log_dir = setup_logging(args.log_level)

    logger.info("[MAIN] Starting hub upload tests")

    # Validate configuration files
    pg_config_path = Path(args.pg_config)
    hf_config_path = Path(args.hf_config)
    ms_config_path = Path(args.ms_config)

    if not pg_config_path.exists():
        _log_error(logger, f"[MAIN] PostgreSQL config not found: {pg_config_path}")
        return 1

    if not hf_config_path.exists():
        _log_error(logger, f"[MAIN] HuggingFace config not found: {hf_config_path}")
        return 1

    if not ms_config_path.exists():
        _log_error(logger, f"[MAIN] ModelScope config not found: {ms_config_path}")
        return 1

    # Validate environment variables
    if not os.getenv("HF_TOKEN"):
        _log_error(logger, "[MAIN] HF_TOKEN environment variable not set")
        return 1

    if not os.getenv("MS_TOKEN"):
        _log_error(logger, "[MAIN] MS_TOKEN environment variable not set")
        return 1

    # Initialize database
    try:
        logger.info(f"[MAIN] Connecting to database using {pg_config_path}")
        db = DatasetDatabase(pg_config_path)
    except Exception as e:
        _log_error(logger, f"[MAIN] Failed to connect to database: {e}")
        return 1

    # Create temporary directory for test files
    test_base_dir = Path(tempfile.mkdtemp(prefix="hub_upload_test_"))
    logger.debug(f"[MAIN] Test directory: {test_base_dir}")

    # Generate timestamp for folder naming
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    saved_state = None
    created_datasets = None

    try:
        with db.with_session() as session:
            # Save initial database state
            saved_state = save_database_state(session, logger)

            # Create test folders (one per dataset)
            test_folders = []
            for i in range(args.num_test_datasets):
                # Create a unique timestamp for each folder
                folder_timestamp = f"{timestamp}_{i:03d}"
                test_folder = create_test_files(
                    test_base_dir, num_files=5, logger=logger, timestamp=folder_timestamp
                )
                test_folders.append(test_folder)

            # Create fake datasets
            created_datasets = create_fake_datasets(
                session, args.num_test_datasets, test_folders, logger
            )

            logger.info(
                f"[MAIN] Created {len(created_datasets)} test datasets for testing"
            )

        # Run tests
        test_results = {}

        # Test HuggingFace - Server-Client mode
        logger.info("[MAIN] Testing HuggingFace (server-client)")
        hf_sc_result = run_server_client_test(
            hf_config_path,
            "huggingface",
            test_timeout=args.test_timeout,
            logger=logger,
        )
        test_results["huggingface_server_client"] = hf_sc_result

        # Test ModelScope - Server-Client mode
        logger.info("[MAIN] Testing ModelScope (server-client)")
        ms_sc_result = run_server_client_test(
            ms_config_path,
            "modelscope",
            test_timeout=args.test_timeout,
            logger=logger,
        )
        test_results["modelscope_server_client"] = ms_sc_result

        # Test HuggingFace - Local mode
        logger.info("[MAIN] Testing HuggingFace (local)")
        hf_local_result = run_local_test(
            hf_config_path,
            "huggingface",
            test_timeout=args.test_timeout,
            logger=logger,
        )
        test_results["huggingface_local"] = hf_local_result

        # Test ModelScope - Local mode
        logger.info("[MAIN] Testing ModelScope (local)")
        ms_local_result = run_local_test(
            ms_config_path,
            "modelscope",
            test_timeout=args.test_timeout,
            logger=logger,
        )
        test_results["modelscope_local"] = ms_local_result

        # Print test summary
        logger.info("[MAIN] Test summary:")
        for hub_name, result in test_results.items():
            if result:
                _log_success(logger, f"[MAIN] {hub_name}: PASSED")
            else:
                _log_error(logger, f"[MAIN] {hub_name}: FAILED")

        # Restore database state
        logger.info("[MAIN] Restoring database state")
        with db.with_session() as session:
            restore_database_state(session, saved_state, logger)

        # Cleanup test files
        logger.info(f"[MAIN] Cleaning up test directory: {test_base_dir}")
        shutil.rmtree(test_base_dir, ignore_errors=True)

        # Determine exit code
        all_passed = all(test_results.values())
        if all_passed:
            _log_success(logger, "[MAIN] All tests passed")
            return 0
        else:
            _log_error(logger, "[MAIN] Some tests failed")
            return 1

    except Exception as e:
        _log_error(logger, f"[MAIN] Test failed: {e}")
        logger.error(f"[MAIN] Exception: {type(e).__name__}: {str(e)}")
        logger.debug(f"[MAIN] Traceback:\n{traceback.format_exc()}")

        # Try to restore database state even on error
        if saved_state:
            try:
                with db.with_session() as session:
                    restore_database_state(session, saved_state, logger)
            except Exception as restore_error:
                _log_error(logger, f"[MAIN] Database restore failed: {restore_error}")
                logger.debug(f"[MAIN] Restore traceback:\n{traceback.format_exc()}")

        # Cleanup test files
        if test_base_dir.exists():
            shutil.rmtree(test_base_dir, ignore_errors=True)

        return 1


if __name__ == "__main__":
    sys.exit(main())

