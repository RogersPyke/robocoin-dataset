import logging
import warnings
from pathlib import Path

from modelscope.hub.api import HubApi  # noqa: E402

# from robocoin_dataset.hub_upload.config.constant import (  # noqa: E402
#     DEFAULT_UPLOAD_ALLOW_PATTERNS,
#     DEFAULT_UPLOAD_IGNORE_PATTERNS,
#     MODELSCOPE_BUG_EXCEPTON_MSG,
# )

DEFAULT_UPLOAD_ALLOW_PATTERNS=None
DEFAULT_UPLOAD_IGNORE_PATTERNS=None
MODELSCOPE_BUG_EXCEPTON_MSG="Expecting value: line 1 column 1 (char 0)"

from .abstract_hub import (  # noqa: E402
    AbstractUploadHub,
)

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")


class ModelscopeUploadHub(AbstractUploadHub):
    """
    Implementation of AbstractUploadHub for ModelScope dataset uploads.

    This class provides functionality to upload datasets to ModelScope Hub,
    including repository creation and file uploading with commit messages.
    Handles a known bug in ModelScope API by catching specific exception messages.

    Attributes:
        hub (HubApi): ModelScope API client instance.
    """

    from modelscope.hub.api import HubApi

    def __init__(self, token: str) -> None:
        """
        Initialize the ModelScope upload hub with authentication token.

        Args:
            token (str): Authentication token for ModelScope API.
        """
        super().__init__(token)
        self.hub = HubApi()

    def repo_exists(self, repo_id: str) -> bool:
        """
        Check if a dataset repository exists on ModelScope Hub.

        Args:
            repo_id (str): Identifier of the repository to check.

        Returns:
            bool: True if repository exists, False otherwise.
        """
        try:
            return self.hub.repo_exists(repo_id=repo_id, token=self.token, repo_type="dataset")
        except Exception as e:
            print(f"[ModelscopeUploadHub.repo_exists] Warning: Could not check if repo {repo_id} exists: {e}")
            return False

    def create_repo(self, repo_id: str) -> None:
        """
        Create a new dataset repository on ModelScope Hub.

        If the repository already exists, this method does nothing due to exist_ok=True.

        Args:
            repo_id (str): Identifier for the new repository.
        """
        if not self.repo_exists(repo_id=repo_id):
            self.hub.create_repo(
                repo_id=repo_id,
                token=self.token,
                repo_type="dataset",
                exist_ok=True,
            )

    def upload_repo(self, folder_path: Path, repo_id: str, commit_msg: str, logger: logging.Logger = None) -> str:
        """
        Upload a local folder to a ModelScope dataset repository.

        Uses batched upload to split large folders into manageable chunks for reliability.

        Args:
            folder_path (Path): Path to the local folder to upload.
            repo_id (str): Identifier of the target repository.
            commit_msg (str): Commit message for the upload.
            logger (logging.Logger): Logger for progress messages (optional).

        Returns:
            str: URL of the commit on ModelScope Hub, or success message if known bug occurs.

        Raises:
            Exception: If upload fails for any reason other than the known ModelScope bug.
        """
        return self.upload_repo_batched(
            folder_path=folder_path,
            repo_id=repo_id,
            commit_msg=commit_msg,
            logger=logger,
        )

    def upload_repo_batched(
        self,
        folder_path: Path,
        repo_id: str,
        commit_msg: str,
        max_batch_size_mb: int = 500,
        max_files_per_batch: int = 1000,
        allow_patterns: list[str] = None,
        ignore_patterns: list[str] = None,
        logger: logging.Logger = None
    ) -> str:
        """
        Upload a folder using ModelScope's upload_folder() method.

        Uses ModelScope's native upload_folder() which handles uploads efficiently
        in a single commit or automatic batching.

        Args:
            folder_path: Path to folder to upload
            repo_id: Repository identifier
            commit_msg: Base commit message
            max_batch_size_mb: Maximum size per batch in MB (ignored - ModelScope handles batching)
            max_files_per_batch: Maximum files per batch (ignored - ModelScope handles batching)
            allow_patterns: Glob patterns for files to include (uses default if None)
            ignore_patterns: Glob patterns for files to exclude (uses default if None)
            logger: Logger for progress messages (optional)

        Returns:
            Success message or URL of the commit
        """
        # Use provided logger or fall back to module logger
        if logger is None:
            logger = logging.getLogger(__name__)

        # Use default patterns if not provided
        if allow_patterns is None:
            allow_patterns = DEFAULT_UPLOAD_ALLOW_PATTERNS
        if ignore_patterns is None:
            ignore_patterns = DEFAULT_UPLOAD_IGNORE_PATTERNS

        try:
            logger.info(f"Uploading folder to {repo_id}...")

            # Use ModelScope's native upload_folder method
            commit_info = self.hub.upload_folder(
                repo_id=repo_id,
                folder_path=str(folder_path),
                token=self.token,
                repo_type="dataset",
                commit_message=commit_msg,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                max_workers=8  # Parallel upload workers
            )

            # Handle both single CommitInfo and List[CommitInfo] returns
            if isinstance(commit_info, list):
                logger.info(f"[ModelscopeUploadHub.upload_repo_batched] Uploaded in {len(commit_info)} batches")
                return f"Uploaded successfully in {len(commit_info)} batches"
            if hasattr(commit_info, 'commit_url'):
                logger.info(f"[ModelscopeUploadHub.upload_repo_batched] Upload completed: {commit_info.commit_url}")
                return commit_info.commit_url
            logger.info(f"[ModelscopeUploadHub.upload_repo_batched] Successfully uploaded to {repo_id}")
            return f"Successfully uploaded to {repo_id}"

        except Exception as e:
            if str(e) == MODELSCOPE_BUG_EXCEPTON_MSG:
                logger.warning("ModelScope API bug encountered, but upload succeeded")
                return f"Exception captured when Modelscope upload dataset {repo_id}, but the repo has been uploaded successfully."
            logger.error(f"Upload failed: {e}")
            raise e
