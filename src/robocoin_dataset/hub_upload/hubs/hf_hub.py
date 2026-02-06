import logging
from pathlib import Path

from huggingface_hub import HfApi

from robocoin_dataset.hub_upload.config.constant import (
    DEFAULT_UPLOAD_ALLOW_PATTERNS,
    DEFAULT_UPLOAD_IGNORE_PATTERNS,
)

from .abstract_hub import AbstractUploadHub


class HuggingfaceUploadHub(AbstractUploadHub):
    """
    Implementation of AbstractUploadHub for Hugging Face dataset uploads.

    This class provides functionality to upload datasets to Hugging Face Hub,
    including repository creation and file uploading with commit messages.

    Attributes:
        logger_name (str): Name of the logger used for Hugging Face API.
        hub (HfApi): Hugging Face API client instance.
    """

    def __init__(self, token: str) -> None:
        """
        Initialize the Hugging Face upload hub with authentication token.

        Args:
            token (str): Authentication token for Hugging Face API.
        """
        super().__init__(token)
        self.hub = HfApi(token=token)

    def repo_exists(self, repo_id: str) -> bool:
        """
        Check if a dataset repository exists on Hugging Face Hub.

        Args:
            repo_id (str): Identifier of the repository to check.

        Returns:
            bool: True if repository exists, False otherwise.
        """
        try:
            return self.hub.repo_exists(repo_id=repo_id, token=self.token, repo_type="dataset")
        except Exception as e:
            print(f"⚠️  Warning: Could not check if repo {repo_id} exists: {e}")
            return False

    def create_repo(self, repo_id: str) -> None:
        """
        Create a new dataset repository on Hugging Face Hub.

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
        Upload a local folder to a Hugging Face dataset repository.

        Uses upload_folder() which handles large uploads efficiently and returns
        commit information.

        Args:
            folder_path (Path): Path to the local folder to upload.
            repo_id (str): Identifier of the target repository.
            commit_msg (str): Commit message for the upload.
            logger (logging.Logger): Logger for progress messages (optional).

        Returns:
            str: URL of the commit on Hugging Face Hub.

        Raises:
            Exception: If upload fails for any reason.
        """
        # Use provided logger or fall back to module logger
        if logger is None:
            logger = logging.getLogger(__name__)

        try:
            logger.info(f"Uploading folder to {repo_id}...")
            commit_info = self.hub.upload_folder(
                repo_id=repo_id,
                folder_path=folder_path,
                token=self.token,
                repo_type="dataset",
                commit_message=commit_msg,
                allow_patterns=DEFAULT_UPLOAD_ALLOW_PATTERNS,
                ignore_patterns=DEFAULT_UPLOAD_IGNORE_PATTERNS,
            )

            # Handle both single CommitInfo object and Future[CommitInfo]
            if hasattr(commit_info, 'commit_url'):
                logger.info(f"✅ Upload completed: {commit_info.commit_url}")
                return commit_info.commit_url
            logger.info(f"✅ Successfully uploaded to {repo_id}")
            return f"Successfully uploaded to {repo_id}"
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            print(e)
            raise e
