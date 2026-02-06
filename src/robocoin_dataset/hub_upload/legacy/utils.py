"""
RoboCoin Datasets Upload Utilities

This module provides utility classes and functions for uploading datasets to remote hubs.
It contains the business logic for dataset upload operations.
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

from .local_datasets_util import LocalDsConfig, LocalDsUtil

ROBOT_NAME_SEPARATORS = frozenset({"_", "-", "."})


_ROBOT_NAMES_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "robot_names.yml"
)
_ROBOT_NAMES_CACHE: list[str] | None = None


def _load_robot_names_from_config() -> list[str]:
    """
    Load the list of robot name identifiers from the shared configuration file.
    """
    global _ROBOT_NAMES_CACHE

    if _ROBOT_NAMES_CACHE is not None:
        return _ROBOT_NAMES_CACHE

    try:
        with open(_ROBOT_NAMES_CONFIG_PATH, encoding="utf-8") as config_file:
            raw_names = yaml.safe_load(config_file)
    except (FileNotFoundError, yaml.YAMLError):
        _ROBOT_NAMES_CACHE = []
        return _ROBOT_NAMES_CACHE

    if isinstance(raw_names, list):
        cleaned_names: list[str] = []
        for entry in raw_names:
            name = str(entry).strip()
            if name:
                cleaned_names.append(name)
        _ROBOT_NAMES_CACHE = cleaned_names
    else:
        _ROBOT_NAMES_CACHE = []

    return _ROBOT_NAMES_CACHE

######## CONFIGURATION ########


@dataclass
class LocalDsUploadConfig(LocalDsConfig):
    """
    Configuration class for local dataset uploading.

    Attributes:
        hub_name (DatasetsHubEnum): Target hub platform for uploading datasets.
            Defaults to DatasetsHubEnum.HUGGINGFACE.
        token (str): Authentication token for the target hub platform. Defaults to empty string.
        namespace (str): Username/namespace on the target hub platform.
        output_path (str): Path to the output directory for commit history files. Defaults to empty string.
        db_file_path (str): Path to the database file for dataset tracking. Defaults to empty string.
        skip_missing (bool): Skip datasets with missing paths instead of aborting.
        force_overwrite (bool): Force overwrite existing repositories without prompting. Defaults to False.
        readme_only (bool): Only update README files without uploading dataset files. Defaults to False.
    """

    hub_name: DatasetsHubEnum = DatasetsHubEnum.huggingface
    token: str = ""
    namespace: str = ""
    output_path: str = ""
    db_file_path: str = ""
    skip_missing: bool = True
    force_overwrite: bool = False
    readme_only: bool = False


def load_config_from_yaml(config_path: str | Path) -> dict:
    """
    Load configuration from YAML file.

    Returns:
        Dictionary containing configuration parameters
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file, encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    if not config_dict:
        raise ValueError(f"Empty or invalid configuration file: {config_path}")

    return config_dict


def create_upload_config(config_dict: dict) -> LocalDsUploadConfig:
    """
    Create LocalDsUploadConfig from configuration dictionary.

    Args:
        config_dict: Dictionary containing configuration parameters

    Returns:
        LocalDsUploadConfig instance

    Raises:
        ValueError: If required configuration parameters are missing
    """
    # Convert hub_name string to enum if needed
    hub_name = config_dict.get("hub_name", "huggingface")
    if isinstance(hub_name, str):
        try:
            hub_name = DatasetsHubEnum[hub_name.lower()]
        except KeyError:
            raise ValueError(f"Invalid hub_name: {hub_name}. Must be 'huggingface' or 'modelscope'")

    return LocalDsUploadConfig(
        hub_name=hub_name,
        token=config_dict.get("token", ""),
        namespace=config_dict.get("namespace", ""),
        output_path=config_dict.get("output_path", ""),
        db_file_path=config_dict.get("db_file_path", ""),
        skip_missing=config_dict.get("skip_missing", True),
        force_overwrite=config_dict.get("force_overwrite", False),
        readme_only=config_dict.get("readme_only", False),
    )


######## UPLOAD UTILITY CLASS ########


class LocalDsUploadUtil(LocalDsUtil):
    """
    Utility class for uploading local datasets to remote hubs.

    This class handles the process of validating local datasets, checking commit history,
    and uploading datasets to either Hugging Face or ModelScope platforms.

    Attributes:
        config (LocalDsUploadConfig): Configuration object for the uploader.
        hub: Hub-specific upload implementation (HuggingfaceUploadHub or ModelscopeUploadHub).
        logger: Logger instance for the uploader.
    """

    def __init__(self, config: LocalDsUploadConfig) -> None:
        """
        Initialize the uploader with configuration.

        Args:
            config (LocalDsUploadConfig): Configuration object for the uploader.

        Raises:
            ValueError: If the specified hub platform is not supported.
        """
        super().__init__(config)
        self.config = config
        self.namespace = config.namespace
        if config.hub_name == DatasetsHubEnum.modelscope:
            from ..hubs.ms_hub import ModelscopeUploadHub

            self.hub = ModelscopeUploadHub(self.config.token)
        elif config.hub_name == DatasetsHubEnum.huggingface:
            from ..hubs.hf_hub import HuggingfaceUploadHub

            self.hub = HuggingfaceUploadHub(self.config.token)
        else:
            raise ValueError(f"hub {config.hub_name} is not supported.")
        pass

        self.logger = self.setup_logger(logger_name="UPLOAD_DATASETS")
        # README 生成与页面同步共用的统一元数据服务。
        # 分布式客户端没有本地数据库时不创建实例，改为使用服务器传入的元数据。
        self.metadata_service: MetadataSyncService | None = None
        if self.config.db_file_path:
            try:
                self.metadata_service = MetadataSyncService(
                    db_file_path=self.config.db_file_path,
                    logger=self.logger,
                )
            except FileNotFoundError as exc:
                self.logger.error("初始化元数据服务失败: %s", exc)
                raise


    def _do_upload(
        self,
        hardlink_path: Path,
        commit_msg: str | None = None,
        upload_path: Path | None = None,
        max_retries: int = 3,
    ) -> tuple[bool, str]:
        """
        Upload a single dataset to the remote hub.
        local function without db: just upload the dataset to the remote hub.

        Args:
            hardlink_path (Path): Path to the hardlink folder to upload.
            commit_msg (str): Commit message for the upload. Defaults to auto-generated message.
            max_retries (int): Maximum number of retry attempts. Defaults to 3.
        Returns: tuple[bool, str]: (success status, error message if failed)
        """

        # Extract dataset name from hardlink path
        dataset_name = self._process_repo_name(hardlink_path.name)

        # Validate dataset structure using shared validation from LocalDsUtil
        # TODO: we no longer use root_path but keep it for compatibility.
        # here we just skip the validation of the dataset structure.
        # ORI code:
        # try:
        #     self.check_dataset_dir_valid(
        #         ds_name=hardlink_path.name,
        #         additional_check_list=[README_FILE],  # Upload requires README.md
        #     )
        # except Exception as e:
        #     tb = traceback.format_exc()
        #     error_msg = f"Validation failed: {e}\n\nFull traceback:\n{tb}"
        #     self.logger.debug(f"{dataset_name}: {error_msg}")
        #     return False, error_msg
        # if not hardlink_path.exists():
        #     raise FileNotFoundError(f"dataset path {hardlink_path} does not exist")

        effective_upload_path = upload_path or hardlink_path
        self.logger.debug(f"{dataset_name}: Using {effective_upload_path}")

        # Repository ID uses clean name (without _hardlink suffix)
        repo_id = f"{self.namespace}/{dataset_name}"

        # Generate commit message if not provided
        if not commit_msg:
            commit_msg = (
                f"Upload dataset {dataset_name}"
                if not self.config.readme_only
                else f"Update README for {dataset_name}"
            )

        # Retry logic with random delays
        for attempt in range(1, max_retries + 1):
            try:  # noqa: PERF203 - retry logic requires try-except in loop
                if not self.hub.repo_exists(repo_id=repo_id):
                    self.logger.debug(f"{dataset_name}: repo not exists, creating repo {repo_id}")
                    self.hub.create_repo(repo_id=repo_id)

                ### UPLOAD: CALL API ###
                commit_url = self.hub.upload_repo(
                    folder_path=effective_upload_path,
                    repo_id=repo_id,
                    commit_msg=commit_msg,
                    logger=self.logger,
                )

                self.logger.debug(f"{dataset_name}: {commit_url}")
                return True, ""

            except Exception as e:  # noqa: PERF203
                if attempt < max_retries:
                    # Random delay before retry
                    delay = random.uniform(2, 10)
                    self.logger.debug(
                        f"{dataset_name}: Retry {attempt}/{max_retries} in {delay:.1f}s: {e}"
                    )
                    # Countdown display
                    for remaining in range(int(delay), 0, -1):
                        print(f"\r  ⏳ {remaining}s...   ", end="", flush=True)
                        time.sleep(1)
                    # Sleep remaining fractional seconds
                    time.sleep(delay - int(delay))
                    print("\r" + " " * 20 + "\r", end="", flush=True)  # Clear the countdown line
                else:
                    tb = traceback.format_exc()
                    error_msg = f"Failed after {max_retries} attempts: {e}\n\nFull traceback:\n{tb}"
                    self.logger.debug(f"{dataset_name}: {error_msg}")
                    return False, error_msg

        return False, "Upload failed with unknown error"

    def _sanitize_repo_name(self, repo_name: str) -> str:
        """
        Sanitize repository name to comply with Hugging Face validation rules:
        - Only alphanumeric chars, '-', '_', or '.' are allowed
        - Cannot start or end with '-' or '.'
        - Maximum length is 96 characters

        Args:
            repo_name: Original repository name

        Returns:
            Sanitized repository name that meets Hugging Face requirements
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

    def _process_repo_name(self, folder_name: str) -> str:
        """
        Normalize a folder name into a valid repository name by stripping upload suffixes,
        removing duplicate robot names, and sanitizing invalid characters.

        Args:
            folder_name: Original folder name (may contain suffixes and invalid chars)

        Returns:
            Sanitized repository name that meets Hugging Face requirements
        """
        dataset_name = folder_name.removesuffix("_qced_hardlink").removesuffix("_hardlink")
        robot_names = self._get_robot_name_list()
        normalized_name = self._remove_duplicate_robot_names(dataset_name, robot_names)
        if normalized_name != dataset_name:
            self.logger.debug(
                "Trimmed duplicate robot names in '%s' -> '%s'",
                dataset_name,
                normalized_name,
            )

        # Sanitize invalid characters to comply with Hugging Face validation rules
        sanitized_name = self._sanitize_repo_name(normalized_name)
        if sanitized_name != normalized_name:
            self.logger.debug(
                "Sanitized repository name '%s' -> '%s'",
                normalized_name,
                sanitized_name,
            )

        return sanitized_name

    def _get_robot_name_list(self) -> list[str]:
        """
        Retrieve robot names from the shared configuration.
        """
        return _load_robot_names_from_config()

    def _remove_duplicate_robot_names(
        self,
        repo_name: str,
        robot_names: list[str],
    ) -> str:
        """
        Ensure each robot name appears at most once in the repository name.
        """
        if not robot_names:
            return repo_name

        sanitized = repo_name
        for robot_name in robot_names:
            if not robot_name:
                continue

            first_index = sanitized.find(robot_name)
            if first_index == -1:
                continue

            search_start = first_index + len(robot_name)
            while True:
                duplicate_index = sanitized.find(robot_name, search_start)
                if duplicate_index == -1:
                    break

                remove_start = self._locate_duplicate_start(sanitized, duplicate_index)
                sanitized = (
                    sanitized[:remove_start]
                    + sanitized[duplicate_index + len(robot_name) :]
                )
                search_start = remove_start

        return sanitized

    def _locate_duplicate_start(self, repo_name: str, duplicate_start: int) -> int:
        """
        Walk backwards to remove surrounding separators before the duplicate entry.
        """
        start = duplicate_start
        while start > 0 and repo_name[start - 1] in ROBOT_NAME_SEPARATORS:
            start -= 1
        return start

    def _upload_one_dataset(
        self,
        hardlink_path: Path,
        metadata: UnifiedMetadata | dict | None = None,
    ) -> tuple[bool, str]:
        """
        Upload a dataset to the hub with YAML and README generation.
        Enhanced version of _upload_one_dataset in LocalDsUtil.

        Args:
            hardlink_path: Path to the hardlink directory.
            metadata: Optional pre-built UnifiedMetadata instance or dict.
                - If provided, it will be used directly (DISTRIBUTED mode).
                - If None, metadata will be collected locally from DB/files
                  using create_unified_metadata (LOCAL mode).

        Returns:
            tuple[bool, str]: (success status, error message if failed or empty string if success)
        """
        # Start timing for this dataset
        dataset_start_time = time.time()

        dataset_name = self._process_repo_name(hardlink_path.name)

        # Define output path for intermediate YAML file
        output_path = Path(self.config.output_path or "./dataset_info").expanduser().absolute()
        output_path.mkdir(parents=True, exist_ok=True)

        # Step 2: Build aggregated metadata from database + local files
        # NOTE:
        #   - LOCAL mode: metadata is collected here using db_file_path.
        #   - DISTRIBUTED mode: metadata is pre-packaged on the server and
        #     passed in via the `metadata` argument so the client never touches DB.
        tqdm.write("    🧩 Collecting unified metadata...")
        self.logger.info(f"{dataset_name}: Collecting unified metadata...")
        try:
            if metadata is None:
                # Local mode: collect metadata using shared service
                if not self.metadata_service:
                    raise ValueError(
                        "metadata_service is not initialized; db_file_path is required for collection"
                    )
                metadata = self.metadata_service.collect_unified_metadata(
                    hardlink_path=hardlink_path,
                    dataset_uuid=None,
                )
            elif isinstance(metadata, dict):
                # Convert dict back to UnifiedMetadata when coming from server
                metadata = UnifiedMetadata.from_dict(metadata)
        except Exception as e:  # noqa: PERF203
            tb = traceback.format_exc()
            error_msg = f"Unified metadata collection failed: {e}\n\nFull traceback:\n{tb}"
            tqdm.write(f"      ❌ Unified metadata collection failed: {e}")
            self.logger.error(f"{dataset_name}: {error_msg}")
            return False, error_msg

        # Step 3: Generate README file for this dataset using unified metadata
        tqdm.write("    📝 Generating README from unified metadata...")
        self.logger.info(f"{dataset_name}: Generating README from unified metadata...")
        readme_success, readme_error = self._generate_readme_for_dataset(
            hardlink_path,
            output_path,
            metadata,
        )
        if not readme_success:
            return False, readme_error

        # Check if we're in readme-only mode
        if self.config.readme_only:
            tqdm.write("    📝 README-only mode: Uploading README.md to hub...")
            self.logger.info(f"{dataset_name}: README-only mode - pushing README.md only")
            result = self._upload_readme_only(hardlink_path, dataset_name)
            dataset_elapsed = time.time() - dataset_start_time
            if result[0]:
                tqdm.write(f"    ✅ README uploaded successfully (took {dataset_elapsed:.1f}s)")
            else:
                tqdm.write(f"    ❌ README upload failed: {result[1] or 'unknown error'}")
            return result

        # Step 4: Execute upload
        tqdm.write("    ⬆️  Uploading to hub (this may take several minutes for large datasets)...")
        self.logger.info(f"{dataset_name}: Uploading to hub...")
        result = self._do_upload(hardlink_path)

        # Calculate elapsed time for this dataset
        dataset_elapsed = time.time() - dataset_start_time

        if result[0]:  # success
            tqdm.write(f"    ✅ Upload completed successfully! (took {dataset_elapsed:.1f}s)")
            self.logger.info(f"{dataset_name}: Upload completed in {dataset_elapsed:.2f}s")
        else:
            self.logger.info(f"{dataset_name}: Upload failed after {dataset_elapsed:.2f}s")

        return result

    def _generate_readme_for_dataset(
        self,
        hardlink_path: Path,
        dataset_info_root_path: Path,
        metadata: UnifiedMetadata,
    ) -> tuple[bool, str]:
        return gen_readme(hardlink_path, dataset_info_root_path, self.logger, metadata=metadata)

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
            return self._do_upload(
                hardlink_path=hardlink_path,
                upload_path=staging_dir,
            )



if __name__ == "__main__":
    pass
