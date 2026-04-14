"""
RoboCoin Datasets Upload Utilities

This module provides utility classes and functions for uploading datasets to remote hubs.
It contains the business logic for dataset upload operations.

Do only upload, no other logic, no checking.
"""

import logging
import os
import random
import re
import shutil
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import yaml
from tqdm import tqdm

from robocoin_dataset.hub_upload.check.val_readme import get_readme_path

#===== Local var =====

UPLOAD_LOGGER_NAME = "UploadUtil"
UPLOAD_REPO_NAME_LOGGER_NAME = "UploadReponameUtil"
MAX_RETRIES = 3

######## CONFIGURATION ########

@dataclass
class UploadConfig():
    """
    Configuration class for dataset uploading.

    This configuration supports both local and distributed upload modes.
    For distributed mode with dedicated servers per hub, each server instance
    should be initialized with a specific hub_name parameter.

    Args:
        hub_name: Specify which hub to upload to. Can be "huggingface" or "modelscope".
                  For server mode, this determines which hub the server handles.
        pg_cfg_path: Path to the postgresql configuration file.
        skip_errors: If skip errors and continue uploading.
        upload_force_overwrite: If force overwrite existing repositories without prompting.
        upload_readme_only: If only update README files without uploading dataset files.
        max_retries: Maximum number of retries for failed uploads.
        
        # HuggingFace configuration
        hf_token: HuggingFace authentication token
        hf_namespace: HuggingFace namespace (username)
        
        # ModelScope configuration
        ms_token: ModelScope authentication token
        ms_namespace: ModelScope namespace (username)
        
        # Common configuration
        force_overwrite: Force overwrite existing repositories
        readme_only: Only update README files without uploading dataset files
        
        # Server network configuration
        server_host: Server host address (default: 0.0.0.0)
        server_port: Server port number (default: 2100 for HF, 2101 for MS recommended)
        server_heartbeat_interval: Heartbeat interval in seconds
        server_timeout: Connection timeout in seconds
        
        # Client network configuration
        client_host: Client host address (default: 127.0.0.1)
        client_port: Client port number (default: 2140)
        client_heartbeat_interval: Heartbeat interval in seconds
        client_timeout: Connection timeout in seconds
    """
    # ===== Global configuration =====
    hub_name: str = "huggingface"  # Renamed from 'hub' for clarity
    pg_cfg_path: str = ""
    skip_errors: bool = False
    upload_force_overwrite: bool = False
    upload_readme_only: bool = False
    max_retries: int = MAX_RETRIES

    # ===== Hub-specific configuration =====
    # HuggingFace configuration
    hf_token: str = ""
    hf_namespace: str = ""
    # ModelScope configuration
    ms_token: str = ""
    ms_namespace: str = ""
    # Common configuration
    force_overwrite: bool = False
    readme_only: bool = False
    
    # ==== Server network configuration =====
    server_host: str = "0.0.0.0"
    server_port: int = 2100
    server_heartbeat_interval: float = 30.0
    server_timeout: float = 90.0

    # ==== Client network configuration =====
    # NOTE: These params are not used in the server mode, only in the client mode when activating clients.
    # Never be passed while Server is genetarating and passing tasks to clients.
    # Only used when activating clients.(Init stage)
    client_host: str = "127.0.0.1"
    client_port: int = 2140
    client_heartbeat_interval: float = 30.0
    client_timeout: float = 90.0
    request_task_timeout: float | None = 90.0

    # ===== helpers to build cfg =====

def _load_config_from_yaml(config_path: str | Path) -> dict:
    """
    Load configuration from YAML file.

    Returns:
        Dictionary containing configuration parameters
    """
    config_file_path = Path(config_path)
    if not config_file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_file_path, encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    if not config_dict:
        raise ValueError(f"Empty or invalid configuration file: {config_path}")
    return config_dict


def _resolve_token_from_env(token: str | None, env_vars: list[str]) -> str:
    """
    Resolve token from configuration or environment variables.
    
    If token is provided and non-empty, use it. Otherwise, try to read from
    environment variables in order of preference.
    
    Args:
        token: Token from configuration file (can be None or empty string)
        env_vars: List of environment variable names to try (in order)
        
    Returns:
        Token string (may be empty if not found)
    """
    # If token is provided and non-empty, use it
    if token and token.strip():
        return token.strip()
    
    # Try environment variables in order
    for env_var in env_vars:
        env_token = os.getenv(env_var)
        if env_token and env_token.strip():
            return env_token.strip()
    
    # Return empty string if not found
    return ""


def _create_upload_config(config_dict: dict) -> UploadConfig:
    """
    Create UploadConfig from configuration dictionary.
    
    This function maps YAML configuration to UploadConfig dataclass.
    Supports both 'hub' and 'hub_name' keys for backward compatibility.
    
    Token resolution priority:
    1. Token from YAML config file (if provided and non-empty)
    2. Environment variable (HF_TOKEN or HUGGINGFACE_TOKEN for HF, MS_TOKEN or MODELSCOPE_TOKEN for MS)
    
    Args:
        config_dict: Dictionary containing configuration parameters from YAML
        
    Returns:
        UploadConfig: Configuration object for upload operations
    """
    # Resolve HuggingFace token from config or environment
    hf_token = _resolve_token_from_env(
        config_dict.get("hf_token"),
        ["HF_TOKEN", "HUGGINGFACE_TOKEN"]
    )
    
    # Resolve ModelScope token from config or environment
    ms_token = _resolve_token_from_env(
        config_dict.get("ms_token"),
        ["MS_TOKEN", "MODELSCOPE_TOKEN"]
    )
    
    return UploadConfig(
        hub_name=config_dict.get("hub_name", config_dict.get("hub", "huggingface")),
        pg_cfg_path=config_dict.get("pg_cfg_path", ""),
        skip_errors=config_dict.get("skip_errors", False),
        upload_force_overwrite=config_dict.get("upload_force_overwrite", False),
        upload_readme_only=config_dict.get("upload_readme_only", False),
        # HuggingFace configuration
        hf_token=hf_token,
        hf_namespace=config_dict.get("hf_namespace", ""),
        # ModelScope configuration
        ms_token=ms_token,
        ms_namespace=config_dict.get("ms_namespace", ""),
        # Common configuration (use upload_* if available, otherwise use direct keys)
        force_overwrite=config_dict.get("force_overwrite", config_dict.get("upload_force_overwrite", False)),
        readme_only=config_dict.get("readme_only", config_dict.get("upload_readme_only", False)),
        # Server network configuration
        server_host=config_dict.get("server_host", config_dict.get("host", "0.0.0.0")),
        server_port=config_dict.get("server_port", config_dict.get("port", 2100)),
        server_heartbeat_interval=config_dict.get("server_heartbeat_interval", config_dict.get("heartbeat_interval", 30.0)),
        server_timeout=config_dict.get("server_timeout", config_dict.get("timeout", 90.0)),
        # Client network configuration
        client_host=config_dict.get("client_host", "127.0.0.1"),
        client_port=config_dict.get("client_port", 2140),
        client_heartbeat_interval=config_dict.get("client_heartbeat_interval", 30.0),
        client_timeout=config_dict.get("client_timeout", 90.0),
        request_task_timeout=config_dict.get("request_task_timeout", 90.0),
    )
    
    # ----- Calling entrypoint -----

def create_config(config_path: str | Path) -> UploadConfig:
    """
    Create UploadConfig from configuration file.
    """
    config_dict = _load_config_from_yaml(config_path)
    return _create_upload_config(config_dict)

######## UPLOAD UTILITY CLASS ########

class UploadUtil():
    """
    Utility class for uploading local datasets to remote hubs.
    Do only upload, no other logic.
    """

    def __init__(self, config: UploadConfig) -> None:
        """
        Initialize the uploader with configuration.
        
        Args:
            config: UploadConfig containing hub_name and authentication tokens
            
        Raises:
            ValueError: If hub_name is not supported
        """
        self.config = config
        hub_name = getattr(config, 'hub_name')
        
        if hub_name == "modelscope" or hub_name == "ms":
            from ..hubs.ms_hub import ModelscopeUploadHub
            self.hub = ModelscopeUploadHub(self.config.ms_token)
        elif hub_name == "huggingface" or hub_name == "hf":
            from ..hubs.hf_hub import HuggingfaceUploadHub
            self.hub = HuggingfaceUploadHub(self.config.hf_token)
        else:
            raise ValueError(f"hub {hub_name} is not supported or illegal.")

        self.logger = logging.getLogger(UPLOAD_LOGGER_NAME)

    def upload(self, hardlink_path: Path, dataset_name: str) -> tuple[bool, str]:
        """
        Upload a single, local dataset folder to the remote hub.
        """
        if not hardlink_path.exists():
            raise FileNotFoundError(f"Dataset path {hardlink_path} does not exist")

        # Extract canonical dataset name from validated info.yaml value.
        dataset_name = self._folder_name_to_repo_name(dataset_name)
        self.logger.debug(f"Uploading dataset named: {dataset_name} from path: {hardlink_path}")

        # Get the correct namespace based on hub_name
        hub_name = getattr(self.config, 'hub_name', 'huggingface')
        if hub_name == "modelscope" or hub_name == "ms":
            namespace = self.config.ms_namespace
        elif hub_name == "huggingface" or hub_name == "hf":
            namespace = self.config.hf_namespace

        # Repository ID uses clean name (without _hardlink suffix)
        repo_id = f"{namespace}/{dataset_name}"

        # Generate commit message
        commit_msg = (
            f"Upload dataset {dataset_name}"
            if not self.config.upload_readme_only
            else f"Update README for {dataset_name}, from README-only mode"
        )

        # Retry logic with random delays
        for attempt in range(1, self.config.max_retries + 1):
            try:  # noqa: PERF203 - retry logic requires try-except in loop
                if not self.hub.repo_exists(repo_id=repo_id):
                    self.logger.info(f"Repo does not exist, creating repo: {repo_id}")
                    self.hub.create_repo(repo_id=repo_id)

                #==== UPLOAD: CALL API ====
                commit_url = self.hub.upload_repo(
                    folder_path=hardlink_path,
                    repo_id=repo_id,
                    commit_msg=commit_msg,
                    logger=self.logger,
                )

                self.logger.debug(f"{dataset_name}: {commit_url}")
                return True, ""

            except Exception as e:  # noqa: PERF203
                # Retry logic: if failed, retry with random delay.
                if attempt < self.config.max_retries:
                    # Random delay before retry
                    delay = random.uniform(2, 10)
                    self.logger.debug(  
                        f"Failed! Retrying... {dataset_name}: Retry {attempt}/{self.config.max_retries} in {delay:.1f}s: {e}"
                    )
                    time.sleep(delay)
                else:
                    tb = traceback.format_exc()
                    error_msg = f"Failed after {self.config.max_retries} attempts: {e}\n\nFull traceback:\n{tb}"
                    self.logger.debug(f"{dataset_name}: {error_msg}")
                    return False, error_msg
        return False, "Upload failed: max retries exceeded without success"

    def upload_readme_only(
        self,
        hardlink_path: Path,
        dataset_name: str,
    ) -> tuple[bool, str]:
        """
        Upload only the README.md file for a dataset by staging it in a temporary folder.
        Uses case-insensitive lookup for README.md (e.g. readme.md, README.md).
        """
        readme_path = get_readme_path(hardlink_path)
        if readme_path is None:
            error_msg = f"README.md not found for {dataset_name}"
            self.logger.error(error_msg)
            return False, error_msg

        with tempfile.TemporaryDirectory(prefix="robo-readme-upload-") as tmpdir:
            staging_dir = Path(tmpdir)
            staging_readme = staging_dir / "README.md"
            shutil.copy2(readme_path, staging_readme)
            self.logger.debug(f"{dataset_name}: Staging README for upload at {staging_readme}")
            return self.upload(
                hardlink_path=staging_dir,
                dataset_name=dataset_name,
            )

    # ---- inner helpers -----

    def _folder_name_to_repo_name(self, folder_name: str) -> str:
        """
        Normalize a folder name into a valid repository name by stripping upload suffixes,
        removing duplicate robot names, and sanitizing invalid characters.
        NOTE: Only process EXPECTED names, do no checking!

        Args:
            folder_name: Original folder name (may contain suffixes and invalid chars)

        Returns:
            Sanitized repository name that meets Hub requirements
        """
        dataset_name = folder_name.removesuffix("_qced_hardlink").removesuffix("_hardlink")

        # Sanitize invalid characters to comply with Hub requirements
        sanitized_name = self._sanitize(dataset_name)
        if sanitized_name != dataset_name:
            self.logger.info(
                "Sanitized repository name '%s' -> '%s'",
                dataset_name,
                sanitized_name,
            )
        return sanitized_name

    def _sanitize(self, repo_name: str) -> str:
        """
        Check if repository name is legal:
        - Only alphanumeric chars, '-', '_', or '.' are allowed
        - Cannot start or end with '-' or '.'
        - Maximum length is 96 characters
        """
        if not repo_name:
            return repo_name

        # Replace invalid characters with underscore
        # Keep only alphanumeric, '-', '_', and '.'
        sanitized = re.sub(r'[^a-zA-Z0-9._-]', '_', repo_name)

        # Remove leading/trailing '-' and '.'
        sanitized = sanitized.strip('-.')

        # Collapse multiple consecutive underscores/dots/dashes into single underscore
        sanitized = re.sub(r'[._-]+', '_', sanitized)

        # Remove leading/trailing separators again after collapsing
        sanitized = sanitized.strip('-.')

        # Truncate to maximum length of 96 characters
        if len(sanitized) > 96:
            sanitized = sanitized[:96].rstrip('-.')

        # Ensure we don't end up with an empty string
        if not sanitized:
            # Fallback: use a default name if sanitization results in empty string
            sanitized = "dataset"
        return sanitized

if __name__ == "__main__":
    pass