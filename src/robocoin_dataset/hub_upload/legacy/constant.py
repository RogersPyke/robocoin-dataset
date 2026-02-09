"""
Constants used across the robocoin datasets module.

This module loads constants from constant.yml and exposes them as Python constants.
It also defines the DatasetsHubEnum enum class.
"""

from enum import Enum
from pathlib import Path

import yaml

# Load constants from YAML file
_CONFIG_FILE = Path(__file__).parent / "constant.yml"
with open(_CONFIG_FILE, encoding="utf-8") as f:
    _CONFIG = yaml.safe_load(f)

# Platform configuration
DS_PLATFORM_NAME = _CONFIG["ds_platform_name"]
"""str: Name of the dataset platform."""

# Upload configuration
DEFAULT_UPLOAD_ALLOW_PATTERNS = _CONFIG["default_upload_allow_patterns"]
"""Optional[list[str]]: Default file patterns to allow during upload (None means all files)."""

DEFAULT_UPLOAD_IGNORE_PATTERNS = _CONFIG["default_upload_ignore_patterns"]
"""list[str]: Default file patterns to ignore during upload."""

# Commit and versioning
COMMIT_MESSAGE_FILE = Path(_CONFIG["commit_message_file"])
"""Path: Filename for storing commit messages."""

COMMIT_MSG_LABEL = _CONFIG["commit_msg_label"]
"""str: Label for commit message in metadata."""

COMMIT_UUID_LABEL = _CONFIG["commit_uuid_label"]
"""str: Label for local commit UUID in metadata."""

REMOTE_COMMIT_UUID_LABEL = _CONFIG["remote_commit_uuid_label"]
"""str: Label for remote commit UUID in metadata."""

INIT_COMMIT_MSG = _CONFIG["init_commit_msg"]
"""str: Default message for initial commits."""

MODELSCOPE_BUG_EXCEPTON_MSG = _CONFIG["modelscope_bug_excepton_msg"]
"""str: Known exception message from Modelscope API bug."""

# Dataset files
DATASET_INFO_FILE = _CONFIG["dataset_info_file"]
"""str: Dataset information file name."""

LEROBOT_META_INFO_FILE = _CONFIG["lerobot_meta_info_file"]
"""str: LeRobot format metadata file path."""

LEROBOT_META_TASKS_FILE = _CONFIG["lerobot_meta_tasks_file"]
"""str: LeRobot format tasks file path."""

LEROBOT_META_EPISODES_FILE = _CONFIG["lerobot_meta_episodes_file"]
"""str: LeRobot format episodes file path."""

LEROBOT_META_EPISODES_STATS_FILE = _CONFIG["lerobot_meta_episodes_stats_file"]
"""str: LeRobot format episodes statistics file path."""

README_FILE = _CONFIG["readme_file"]
"""str: README file name."""

# New structure directories (LeRobot v2.0+ format)
ANNOTATIONS_DIR = _CONFIG["annotations_dir"]
"""str: Annotations directory path."""

DATA_DIR = _CONFIG["data_dir"]
"""str: Data directory path."""

META_DIR = _CONFIG["meta_dir"]
"""str: Metadata directory path."""

VIDEOS_DIR = _CONFIG["videos_dir"]
"""str: Videos directory path."""

LOCAL_DATASET_CHECK_STRUCTURE = _CONFIG["local_dataset_check_structure"]
"""list[str]: Required files and directories for basic local dataset validation (LeRobot v2.0+ format)."""

GEN_README_DATASET_ADDITIONAL_CHECK_STRUCTURE = _CONFIG["gen_readme_dataset_additional_check_structure"]
"""list[str]: Additional files required for README generation."""

UPLOAD_DATASET_ADDITIONAL_CHECK_STRUCTURE = _CONFIG["upload_dataset_additional_check_structure"]
"""list[str]: Additional files required for dataset upload."""

IGNORED_SUBTASKS = _CONFIG["ignored_subtasks"]
"""list[str]: Subtasks that should be ignored during processing."""

HUB_DATASETS_INFO_FOLDER = _CONFIG["hub_datasets_info_folder"]
"""str: Folder name for storing hub dataset information."""

DATASETS_UPLOAD_LOG_FOLDER = _CONFIG["datasets_upload_log_folder"]
"""str: Folder name for upload log files."""

DATASETS_GEN_README_LOG_FOLDER = _CONFIG["datasets_gen_readme_log_folder"]
"""str: Folder name for README generation log files."""

DATASETS_GEN_INFO_LOG_FOLDER = _CONFIG["datasets_gen_info_log_folder"]
"""str: Folder name for dataset info generation log files."""

DATASET_INFO_TEMPLATE_FILE = _CONFIG["dataset_info_template_file"]
"""str: Path to the dataset info template file."""

DEFAULT_OUTPUT_LOG_PATH = Path(_CONFIG["default_output_log_path"])

DEVICE_LIST_KEY = _CONFIG["device_list_key"]

DEVICE_TO_FEATURES_FILE = _CONFIG["device_to_features_file"]

LEROBOT_FEATURES_KEY = _CONFIG["lerobot_features_key"]

LEROBOT_OBSERVATION_KEY = _CONFIG["lerobot_observation_key"]

LEROBOT_IMAGE_KEY = _CONFIG["lerobot_image_key"]

LEROBOT_STATE_KEY = _CONFIG["lerobot_state_key"]

LEROBOT_ACTION_KEY = _CONFIG["lerobot_action_key"]

FEATURE_CAM_NAME_KEY = _CONFIG["feature_cam_name_key"]

LEROBOT_FEATURE_DESCRIPTON_KEY = _CONFIG["lerobot_feature_descripton_key"]

LEROBOT_DEFAULT_IMAGE_SHAPE_NAMES = _CONFIG["lerobot_default_image_shape_names"]

FEATURE_NAME_KEY = _CONFIG["feature_name_key"]

TASK_DESCRIPTIONS_KEY = _CONFIG["task_descriptions_key"]

TASK_INDEX_KEY = _CONFIG["task_index_key"]
