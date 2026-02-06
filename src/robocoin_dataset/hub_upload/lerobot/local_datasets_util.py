"""
Local dataset util base class
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from functools import cached_property
from pathlib import Path

from .log_config import LogConfig


@dataclass
class LocalDsConfig:
    """
    Configuration class for local dataset operations.

    Attributes:
        root_path (Path | None): Root directory path for datasets. Can be None if not needed.
        process_all (bool): Flag to indicate whether to process all subdirectories. Defaults to True.
        subdirs_to_process (list[str]): List of specific subdirectories to process. Defaults to empty list.
        subdirs_to_ignore (list[str]): List of subdirectories to ignore during processing. Defaults to empty list.
        log_config (LogConfig): Logging configuration object. Defaults to empty LogConfig.
    """

    root_path: Path | None = None
    process_all: bool = True
    subdirs_to_process: list[str] = field(default_factory=list)
    subdirs_to_ignore: list[str] = field(default_factory=list)
    log_config: LogConfig = field(default_factory=LogConfig)


class LocalDsUtil:
    """
    Utility class for handling local dataset operations.

    This class provides methods for validating dataset directories,
    checking file structures, and setting up logging for dataset operations.

    Attributes:
        config (LocalDsConfig): Configuration object for local dataset operations.
    """

    def __init__(self, config: LocalDsConfig) -> None:
        """
        Initialize LocalDsUtil with configuration.

        Args:
            config (LocalDsConfig): Configuration object for local dataset operations.
        """
        self.config = config

    def setup_logger(self, logger_name: str) -> logging.Logger:
        """
        Set up and configure logger for dataset operations.

        Args:
            logger_name (str): Name for the logger instance.

        Returns:
            logging.Logger: Configured logger instance.

        Raises:
            ValueError: If log_dir is not specified in log_config.
        """
        if not self.config.log_config.log_dir:
            raise ValueError("log_dir is required in log_config")

        logger = logging.getLogger(logger_name)
        logger.setLevel(getattr(logging, self.config.log_config.log_level.upper()))

        log_dir = Path(self.config.log_config.log_dir).expanduser().absolute()
        log_dir.mkdir(parents=True, exist_ok=True)

        if logger.handlers:
            logger.handlers.clear()

        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_filename = f"{logger_name}_{timestamp}.log"

        log_filepath = log_dir / log_filename

        if self.config.log_config.log_to_console:
            ch = logging.StreamHandler()
            ch.setLevel(getattr(logging, self.config.log_config.log_level.upper()))
            ch.setFormatter(formatter)
            logger.addHandler(ch)

        fh = logging.FileHandler(log_filepath, encoding="utf-8")
        fh.setLevel(getattr(logging, self.config.log_config.log_level.upper()))
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        logger.info(f"Logging is enabled, output to: {log_filepath}")

        return logger

    @cached_property
    def root_path(self) -> Path:
        """
        Get the absolute root path for datasets with validation.

        Returns:
            Path: Absolute path to the root dataset directory.

        Raises:
            ValueError: If root_path is not set in configuration.
            FileNotFoundError: If the root path does not exist.
            NotADirectoryError: If the root path is not a directory.
        """
        if self.config.root_path is None:
            raise ValueError("root_path is not set in configuration")

        path = Path(self.config.root_path).expanduser().absolute()
        if not path.exists():
            raise FileNotFoundError(f"root_path {path} does not exists")
        if not path.is_dir():
            raise NotADirectoryError(f"root_path {path} is not a directory")
        return path

    def get_root_path_subdirs(self) -> list[str]:
        """
        Get list of subdirectories in the root path based on configuration.

        Returns:
            list[str]: List of subdirectory names in the root path.
        """
        self.check_root_path_valid()

        sub_dirs: list[str] = []

        if self.config.process_all:
            sub_dirs = [
                p.name
                for p in self.root_path.iterdir()
                if p.is_dir() and p.name not in self.config.subdirs_to_ignore
            ]
        else:
            sub_dirs = [
                p
                for p in self.config.subdirs_to_process
                if (self.root_path / p).is_dir() and p not in self.config.subdirs_to_ignore
            ]
        return sub_dirs

    def check_dataset_dir_valid(self, ds_name: str, additional_check_list: list[str] = []) -> None:
        """
        Check if a dataset directory is valid and contains required files.

        Args:
            ds_name (str): Name of the dataset directory to check.
            additional_check_list (list[str], optional): Additional files to check for. Defaults to [].

        Raises:
            FileNotFoundError: If dataset directory doesn't exist or required files are missing.
            NotADirectoryError: If the dataset path is not a directory.
        """
        from ..config.constant import LOCAL_DATASET_CHECK_STRUCTURE

        ds_path = self.root_path / ds_name
        if not ds_path.exists():
            raise FileNotFoundError(f"dataset {ds_name} does not exists in {self.root_path}")
        if not ds_path.is_dir():
            raise NotADirectoryError(f"dataset {ds_name} is not a directory in {self.root_path}")

        # Check required files and directories
        missing_items: list[Path] = []
        for item in LOCAL_DATASET_CHECK_STRUCTURE + additional_check_list:
            item_path = self.root_path.joinpath(ds_name).joinpath(item)
            if not item_path.exists():
                missing_items.append(item_path)

        if missing_items:
            missing_items_str = "\n  ".join(map(str, missing_items))
            raise FileNotFoundError(
                f"Dataset {ds_name} missing required files/directories:\n  {missing_items_str}"
            )

    def check_root_path_valid(self) -> None:
        """
        Validate that the root path exists and is a directory.

        Raises:
            ValueError: If root_path is not set.
            FileNotFoundError: If the root path does not exist.
            NotADirectoryError: If the root path is not a directory.
        """
        if self.config.root_path is None:
            raise ValueError("root_path is not set in configuration")

        if not self.root_path.exists():
            raise FileNotFoundError(f"root_path {self.root_path} does not exists")
        if not self.root_path.is_dir():
            raise NotADirectoryError(f"root_path {self.root_path} is not a directory")


@dataclass
class LocalDsReadmeConfig(LocalDsConfig):
  """
  Configuration class for local dataset README generation.

  Attributes:
      dataset_info_root_path (str): Root path containing dataset info files. Defaults to empty string.
  """

  dataset_info_root_path: str = ""


class LocalDsReadmeUtil(LocalDsUtil):
  """
  Utility class for generating dataset README files from Jinja2 templates.

  This class generates README.md files for datasets by combining dataset information
  with a Jinja2 template, producing formatted documentation for each dataset.

  Attributes:
      config (LocalDsReadmeConfig): Configuration object for the README generator.
      logger: Logger instance for the README generator.
  """

  def __init__(self, config: LocalDsReadmeConfig) -> None:
    """
    Initialize the README generator with configuration.

    Args:
        config (LocalDsReadmeConfig): Configuration object for the README generator.
    """
    super().__init__(config)
    self.config = config

    self.logger = self.setup_logger(logger_name="GEN_DATASET_README")

  @cached_property
  def readme_template_file(self) -> Path:
    """
    Get the path to the README template file with validation.

    Returns:
        Path: Absolute path to the README template file.

    Raises:
        FileNotFoundError: If the README template file does not exist.
    """
    path = Path(__file__).parent.parent.parent.joinpath(
        "prepare_metadata",
        "readmes",
        "templates",
        "readme.j2",
    )
    if not path.exists():
      raise FileNotFoundError(f"readme template file {path} does not exists")
    return path

  def _generate_readme(self, ds_name: str) -> None:
    """
    Generate README file for a specific dataset using Jinja2 template.

    IMPORTANT: This method ALWAYS OVERWRITES the existing README.md file.
    The file is opened in write mode ('w'), which truncates any existing content.

    Args:
        ds_name (str): Name of the dataset to generate README for.

    Raises:
        FileNotFoundError: If meta info file does not exist.
        RuntimeError: If there are errors during README generation.
    """
    from jinja2 import Environment, FileSystemLoader

    from ...prepare_metadata.metadata_collect_utils import (
        generate_folder_structure,
    )
    from ..config.constant import (
        DATASET_INFO_FILE,
        LEROBOT_META_INFO_FILE,
    )

    def get_meta_info_content() -> str:
      """
      Get the content of the meta info file.

      Returns:
          str: Content of the meta info file.

      Raises:
          FileNotFoundError: If meta info file does not exist.
      """
      meta_info_file = self.root_path.joinpath(ds_name, LEROBOT_META_INFO_FILE)
      if not meta_info_file.exists():
        raise FileNotFoundError(f"Meta info file {meta_info_file} does not exist.")
      return meta_info_file.read_text(encoding="utf-8")

    def get_folder_structure() -> str:
      """
      Get the folder structure tree for the dataset.

      Returns:
          str: Formatted folder structure tree showing leaf directories with first 5 files.
      """
      ds_path = self.root_path.joinpath(ds_name)
      return generate_folder_structure(ds_path, max_files_per_dir=5)

    ds_info_file = (
      Path(self.config.dataset_info_root_path)
      .joinpath(ds_name, DATASET_INFO_FILE)
      .expanduser()
      .absolute()
    )
    ds_info: dict
    try:
      import yaml
      with open(ds_info_file) as f:
        ds_info = yaml.safe_load(f)
    except Exception as e:
      raise RuntimeError(e) from e

    # Auto-generate structure if not provided in ds_info
    if "structure" not in ds_info or ds_info.get("structure") == "auto_generated":
      ds_info["structure"] = get_folder_structure()

    try:
      env = Environment(loader=FileSystemLoader(self.readme_template_file.parent))
      env.globals["get_meta_info_content"] = get_meta_info_content
      ###########################################################################
      # Remove _qced_hardlink suffix from dataset_name for display in README.md #
      display_dataset_name = ds_name.removesuffix("_qced_hardlink")
      readme_content = env.get_template(self.readme_template_file.name).render(
        dataset_name=display_dataset_name, **ds_info
      )

      ds_path = self.root_path.joinpath(ds_name)
      readme_file = ds_path.joinpath("README.md")

      # Always overwrite the README.md file
      with open(readme_file, "w", encoding="utf-8") as f:
        f.write(readme_content)
    except Exception as e:
      raise RuntimeError(e) from e

  def generate_readmes(self) -> None:
    """
    Generate README files for all valid datasets in the root path.

    This method validates datasets and generates README.md files for each one
    using the Jinja2 template and dataset information files.
    """
    from ..config.constant import README_FILE

    self.check_root_path_valid()
    ds_names = self.get_root_path_subdirs()

    for ds_name in ds_names:
      self.check_dataset_dir_valid(ds_name=ds_name)
      log_prefix = f"dataset {ds_name}:"

      try:
        self._generate_readme(ds_name=ds_name)
        self.logger.info(f"{log_prefix} generate {README_FILE} successfully")
      except Exception as e:
        self.logger.error(f"{log_prefix} generate {README_FILE} failed, {e}")
