"""
README Generator Module

This module provides the ReadmeGenerator class for generating README.md files
from YAML metadata and Jinja2 templates. It orchestrates the README generation
process by calling underlying utility functions.

Dependencies:
    - robocoin_dataset.readme modules: For implementation functions
    - yaml: For parsing configuration files
    - pathlib: For file path operations

Usage:
    Basic usage:
        generator = ReadmeGenerator(dataset_path="/path/to/dataset")
        generator.generate_readme()

    With custom configuration:
        generator = ReadmeGenerator(
            dataset_path="/path/to/dataset",
            info_yaml_path="/custom/path/info.yaml",
            template_path="/custom/path/readme.j2",
            output_path="/custom/path/README.md"
        )
        generator.generate_readme()
"""

import logging
from pathlib import Path
from typing import Optional

import yaml

from robocoin_dataset.readme.context_utils import build_readme_context_from_schema
from robocoin_dataset.readme.logging_utils import (
    setup_readme_logger,
    validate_hardlink_directory,
)
from robocoin_dataset.readme.template_utils import (
    load_jinja2_template,
    render_template,
    write_readme_file,
)
from robocoin_dataset.readme.yaml_utils import locate_yaml_file
from robocoin_dataset.utils.log_config import log_error, log_success  # noqa: F401


# ============================================================================
# Main ReadmeGenerator Class
# ============================================================================


class ReadmeGenerator:
    """
    README Generator class for generating README.md files from YAML metadata and Jinja2 templates.

    This class orchestrates the README generation process by calling underlying utility functions.
    All actual implementation is delegated to functions in utils.py.

    Attributes:
        dataset_path (Path): Path to the dataset directory
        info_yaml_path (Path): Path to info.yaml file (default: dataset_path/readme/info.yaml)
        template_path (Path): Path to readme.j2 template (default: dataset_path/readme/readme.j2)
        output_path (Path): Path for output README.md (default: dataset_path/README.md)
        log_dir (Path): Directory for log files (default: dataset_path/logs/gen_readme)
        logger (logging.Logger): Logger instance for audit logging

    Usage:
        Basic usage:
            generator = ReadmeGenerator(dataset_path="/path/to/dataset")
            generator.generate_readme()

        With custom paths:
            generator = ReadmeGenerator(
                dataset_path="/path/to/dataset",
                info_yaml_path="/custom/path/info.yaml",
                template_path="/custom/path/readme.j2",
                output_path="/custom/path/README.md"
            )
            generator.generate_readme()
    """

    def __init__(
        self,
        dataset_path: str | Path,
        info_yaml_path: Optional[str | Path] = None,
        template_path: Optional[str | Path] = None,
        output_path: Optional[str | Path] = None,
        log_dir: Optional[str | Path] = None,
        local_dataset_info_filename: str = "local_dataset_info.yaml",
    ) -> None:
        """
        Initialize ReadmeGenerator instance.

        Input:
            dataset_path (str | Path): Path to the dataset directory (hardlink directory)
            info_yaml_path (Optional[str | Path]): Custom path to info.yaml file.
                If None, defaults to dataset_path/readme/info.yaml
            template_path (Optional[str | Path]): Custom path to readme.j2 template.
                If None, defaults to dataset_path/readme/readme.j2
            output_path (Optional[str | Path]): Custom path for output README.md.
                If None, defaults to dataset_path/README.md
            log_dir (Optional[str | Path]): Directory for log files.
                If None, defaults to dataset_path/logs/gen_readme
            local_dataset_info_filename (str): Filename to search for at dataset_path first level.
                If found, used as local metadata file. (default: "local_dataset_info.yaml")

        Output:
            None

        Logic:
            1. Convert all paths to Path objects
            2. Setup logger using utility function (early to log diagnostics)
            3. Locate YAML file: search in dataset_path first level, then validate custom path
            4. Set default paths if not provided
            5. Log initialization parameters

        Usage:
            Called when creating a ReadmeGenerator instance.

        Raises:
            FileNotFoundError: If YAML file cannot be located or dataset_path is invalid
        """
        self.dataset_path = Path(dataset_path).expanduser().resolve()
        module_root = Path(__file__).resolve().parent

        # Setup log directory early (needed for logger)
        if log_dir is None:
            self.log_dir = self.dataset_path / "logs" / "gen_readme"
        else:
            self.log_dir = Path(log_dir).expanduser().resolve()

        # Setup logger using utility function (do this early for diagnostics)
        self.logger = setup_readme_logger(
            log_dir=self.log_dir,
            script_name="gen_readme",
            level=logging.INFO,
            console_output=True,
        )

        # Validate dataset_path exists
        if not self.dataset_path.exists():
            error_msg = f"[INIT] Dataset directory does not exist: {self.dataset_path}"
            log_error(self.logger, error_msg)
            raise FileNotFoundError(error_msg)

        if not self.dataset_path.is_dir():
            error_msg = f"[INIT] Dataset path is not a directory: {self.dataset_path}"
            log_error(self.logger, error_msg)
            raise ValueError(error_msg)

        # Validate that directory appears to be a hardlink directory (soft warning)
        validate_hardlink_directory(self.dataset_path, self.logger)

        # Locate YAML file using priority-based search:
        # 1. Search for local_dataset_info_filename in dataset_path's first level
        # 2. If custom path provided, validate and use it
        # 3. If not found, error and exit
        self.info_yaml_path = locate_yaml_file(
            hardlink_dir=self.dataset_path,
            yaml_filename=local_dataset_info_filename,
            custom_yaml_path=info_yaml_path,
            logger=self.logger,
        )

        # Set other default paths
        # Schema path is fixed to package assets/info.yaml.
        self.schema_yaml_path = module_root / "assets" / "info.yaml"

        if template_path is None:
            self.template_path = module_root / "assets" / "readme.j2"
        else:
            self.template_path = Path(template_path).expanduser().resolve()

        if output_path is None:
            self.output_path = self.dataset_path / "README.md"
        else:
            self.output_path = Path(output_path).expanduser().resolve()

        # Log initialization
        self.logger.info("[INIT] ReadmeGenerator initialized")
        self.logger.info(f"[INIT] Dataset path: {self.dataset_path}")
        self.logger.info(f"[INIT] Local dataset info path: {self.info_yaml_path}")
        self.logger.info(f"[INIT] Schema YAML path: {self.schema_yaml_path}")
        self.logger.info(f"[INIT] Template path: {self.template_path}")
        self.logger.info(f"[INIT] Output path: {self.output_path}")
        self.logger.info(f"[INIT] Log directory: {self.log_dir}")

    def generate_readme(self) -> Path:
        """
        Generate README.md file from YAML metadata and Jinja2 template.

        Input:
            None (uses instance attributes)

        Output:
            Path: Path to the generated README.md file

        Logic:
            1. Load YAML metadata using utility function
            2. Load Jinja2 template using utility function
            3. Render template using utility function
            4. Write README file using utility function
            5. Return output file path

        Usage:
            Main entry point for generating README files.
            Call this method after initializing ReadmeGenerator.

        Note:
            The info_yaml_path is determined during __init__ using locate_yaml_file,
            which implements priority-based search: first in dataset_path first level,
            then validates custom path if provided. FileNotFoundError is raised early
            during __init__ if no valid YAML file can be located.

        Raises:
            FileNotFoundError: If template file is missing
            yaml.YAMLError: If YAML parsing fails
            jinja2.TemplateError: If template rendering fails
            IOError: If file writing fails
        """
        try:
            self.logger.info("[GENERATE] Starting README generation process")

            # Step 1: Build context from schema + source-scoped metadata.
            self.logger.info("[GENERATE] Step 1: Building context from schema and source files")
            yaml_data, source_type_counter = build_readme_context_from_schema(
                schema_yaml_path=self.schema_yaml_path,
                dataset_path=self.dataset_path,
                local_dataset_info_path=self.info_yaml_path,
                logger=self.logger,
            )
            self.logger.info(
                f"[GENERATE] Built context with {len(yaml_data)} fields from schema"
            )
            self.logger.info(
                "[GENERATE] Source type summary: "
                + ", ".join(
                    [
                        f"{k}={v}"
                        for k, v in sorted(source_type_counter.items(), key=lambda x: x[0])
                    ]
                )
            )

            # Step 2: Load Jinja2 template
            self.logger.info("[GENERATE] Step 2: Loading Jinja2 template")
            template = load_jinja2_template(self.template_path, self.logger)

            # Step 3: Render template
            self.logger.info("[GENERATE] Step 3: Rendering template with data")
            rendered_content = render_template(template, yaml_data, self.logger)

            # Step 4: Write README file
            self.logger.info("[GENERATE] Step 4: Writing README file")
            write_readme_file(rendered_content, self.output_path, self.logger)

            # Success
            log_success(
                self.logger,
                f"[GENERATE] README generation completed successfully: {self.output_path}",
            )

            return self.output_path

        except Exception as e:
            error_msg = f"[GENERATE] README generation failed: {e}"
            log_error(self.logger, error_msg)
            self.logger.debug("[GENERATE] Exception traceback:", exc_info=True)
            raise

    # ===== helpers =====


def create_from_config_file(config_path: str | Path) -> ReadmeGenerator:
    """
    Create ReadmeGenerator instance from configuration file.

    Input:
        config_path (str | Path): Path to YAML configuration file

    Output:
        ReadmeGenerator: Initialized ReadmeGenerator instance

    Logic:
        1. Load configuration from YAML file
        2. Extract configuration parameters
        3. Create and return ReadmeGenerator instance

    Usage:
        For future extension: allows creating ReadmeGenerator from config file.
        Config file format (example):
            dataset_path: /path/to/dataset
            info_yaml_path: /custom/path/info.yaml  # optional
            template_path: /custom/path/readme.j2  # optional
            output_path: /custom/path/README.md     # optional
            log_dir: /custom/path/logs              # optional

    Raises:
        FileNotFoundError: If config file does not exist
        yaml.YAMLError: If config file parsing fails
        KeyError: If required 'dataset_path' is missing
    """
    config_path = Path(config_path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if config is None:
            raise ValueError(f"Configuration file is empty: {config_path}")

        # Extract required parameter
        if "dataset_path" not in config:
            raise KeyError("Required parameter 'dataset_path' is missing in config file")

        # Extract optional parameters
        info_yaml_path = config.get("info_yaml_path")
        template_path = config.get("template_path")
        output_path = config.get("output_path")
        log_dir = config.get("log_dir")

        # Create ReadmeGenerator instance
        return ReadmeGenerator(
            dataset_path=config["dataset_path"],
            info_yaml_path=info_yaml_path,
            template_path=template_path,
            output_path=output_path,
            log_dir=log_dir,
        )

    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse configuration file {config_path}: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Unexpected error loading configuration file {config_path}: {e}") from e


def create_from_cli_args(
    dataset_path: str | Path,
    info_yaml_path: Optional[str | Path] = None,
    template_path: Optional[str | Path] = None,
    output_path: Optional[str | Path] = None,
    log_dir: Optional[str | Path] = None,
) -> ReadmeGenerator:
    """
    Create ReadmeGenerator instance from CLI arguments.

    Input:
        dataset_path (str | Path): Path to the dataset directory (required)
        info_yaml_path (Optional[str | Path]): Custom path to info.yaml file
        template_path (Optional[str | Path]): Custom path to readme.j2 template
        output_path (Optional[str | Path]): Custom path for output README.md
        log_dir (Optional[str | Path]): Directory for log files

    Output:
        ReadmeGenerator: Initialized ReadmeGenerator instance

    Logic:
        1. Validate required parameters
        2. Create and return ReadmeGenerator instance

    Usage:
        For future extension: allows creating ReadmeGenerator from CLI arguments.
        Example:
            generator = create_from_cli_args(
                dataset_path="/path/to/dataset",
                info_yaml_path="/custom/path/info.yaml",
                output_path="/custom/path/README.md"
            )
            generator.generate_readme()

    Raises:
        ValueError: If required 'dataset_path' is None or empty
    """
    if not dataset_path:
        raise ValueError("Required parameter 'dataset_path' cannot be None or empty")

    return ReadmeGenerator(
        dataset_path=dataset_path,
        info_yaml_path=info_yaml_path,
        template_path=template_path,
        output_path=output_path,
        log_dir=log_dir,
    )
