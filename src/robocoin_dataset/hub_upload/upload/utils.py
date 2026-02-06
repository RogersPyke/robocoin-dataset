"""
RoboCoin Datasets Upload Utilities

This module provides utility classes and functions for uploading datasets to remote hubs.
It contains the business logic for dataset upload operations.

Do only upload, no other logic, no checking.
"""

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

from robocoin_dataset.hub_upload.gen_readme.gen_readme import gen_readme
from robocoin_dataset.prepare_metadata.metadata_service import MetadataSyncService
from robocoin_dataset.prepare_metadata.unified_metadata_def import UnifiedMetadata

#===== Local var =====

UPLOAD_LOGGER_NAME = "UploadUtil"
UPLOAD_REPO_NAME_LOGGER_NAME = "UploadReponameUtil"
MAX_RETRIES = 3

######## CONFIGURATION ########

@dataclass
class UploadConfig():
    """
    Configuration class for local dataset uploading.

    Args:
        hub_name: specify which hub to upload to. can be "huggingface" or "modelscope".
        token: token for authentication with the hub.
        namespace: namespace (username) on the hub platform.
        pg_cfg_path: path to the postgresql configuration file.
        skip_errors: if skip errors and continue uploading.
        upload_force_overwrite: if force overwrite existing repositories without prompting.
        upload_readme_only: if only update README files without uploading dataset files.
    """

    hub_name: str = "huggingface"
    token: str = ""
    namespace: str = ""
    pg_cfg_path: str = ""
    skip_errors: bool = False
    upload_force_overwrite: bool = False
    upload_readme_only: bool = False
    max_retries: int = MAX_RETRIES


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


def _create_upload_config(config_dict: dict) -> UploadConfig:
    """
    Create UploadConfig from configuration dictionary.
    should pass yaml cfg dict into this function.
    """
    return UploadConfig(
        hub_name=config_dict.get("hub_name", "huggingface"),
        token=config_dict.get("token", ""),
        namespace=config_dict.get("namespace", ""),
        pg_cfg_path=config_dict.get("pg_cfg_path", ""),
        skip_errors=config_dict.get("skip_errors", False),
        upload_force_overwrite=config_dict.get("upload_force_overwrite", False),
        upload_readme_only=config_dict.get("upload_readme_only", False),
    )


######## UPLOAD UTILITY CLASS ########


class UploadUtil():
    """
    Utility class for uploading local datasets to remote hubs.
    Do only upload, no other logic.
    """

    def __init__(self, config: UploadConfig) -> None:
        """
        Initialize the uploader with configuration.
        """
        self.config = config
        if config.hub_name == "modelscope":
            from ..hubs.ms_hub import ModelscopeUploadHub
            self.hub = ModelscopeUploadHub(self.config.token)
        elif config.hub_name == "huggingface":
            from ..hubs.hf_hub import HuggingfaceUploadHub
            self.hub = HuggingfaceUploadHub(self.config.token)
        else:
            raise ValueError(f"hub {config.hub_name} is not supported or illegal.")

        self.logger = self.setup_logger(logger_name=UPLOAD_LOGGER_NAME)

    def _upload(self, hardlink_path: Path) -> tuple[bool]:
        """
        Upload a single, local dataset folder to the remote hub.
        """
        if not hardlink_path.exists():
            raise FileNotFoundError(f"Dataset path {hardlink_path} does not exist")

        # Extract dataset name from hardlink path
        dataset_name = self._folder_name_to_repo_name(hardlink_path.name)
        self.logger.debug(f"Uploading dataset named: {dataset_name} from path: {hardlink_path}")

        # Repository ID uses clean name (without _hardlink suffix)
        repo_id = f"{self.config.namespace}/{dataset_name}"

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
                return True

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
        return False

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
    
    def _upload_readme_only(
        self,
        hardlink_path: Path,
        dataset_name: str,
    ) -> tuple[bool, str]:
        """
        Upload only the README.md file for a dataset by staging it in a temporary folder.
        """
        readme_path = hardlink_path / "README.md"
        if not readme_path.exists():
            error_msg = f"README.md not found for {dataset_name}"
            self.logger.error(error_msg)
            return False, error_msg

        with tempfile.TemporaryDirectory(prefix="robo-readme-upload-") as tmpdir:
            staging_dir = Path(tmpdir)
            staging_readme = staging_dir / "README.md"
            shutil.copy2(readme_path, staging_readme)
            self.logger.debug(f"{dataset_name}: Staging README for upload at {staging_readme}")
            return self._upload(
                hardlink_path=hardlink_path,
                upload_path=staging_dir,
            )

if __name__ == "__main__":
    pass