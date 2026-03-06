"""
README Generator Module

This module provides the ReadmeGenerator class for generating README.md files
from pre-collected YAML metadata and Jinja2 templates.

Strict responsibility (Render stage only):
    ReadmeGenerator ONLY does two things:
    1. Delegate Phase 1 (Collect) to InfoCollector from metadata.collect
    2. Execute Phase 2 (Render) by loading template and rendering with collected data

    NO metadata collection/auto-fields/schema resolution logic here.
    NO input validation beyond file existence.

Generation flow (two-phase pipeline):
    Phase 1 - Collect (delegated to metadata.collect.InfoCollector):
        Reads local_dataset_info.yaml + all dataset source files,
        resolves every schema field, and writes flat info.yaml.

    Phase 2 - Render (executed by ReadmeGenerator):
        Reads the collected info.yaml, loads Jinja2 template,
        renders template, and writes README.md.

Dependencies:
    - robocoin_dataset.metadata.collect: InfoCollector for Phase 1 delegation
    - robocoin_dataset.readme._utils: load_collected_info_yaml for Phase 2
    - robocoin_dataset.readme._logging: Logger setup for Render stage
    - robocoin_dataset.readme._template_utils: Template operations for Phase 2
    - yaml: For parsing configuration files
    - pathlib: For file path operations

Usage:
    Basic usage:
        generator = ReadmeGenerator(dataset_path="/path/to/dataset")
        generator.generate_readme()

    With custom configuration:
        generator = ReadmeGenerator(
            dataset_path="/path/to/dataset",
            local_dataset_info_path="/custom/path/local_dataset_info.yaml",
            template_path="/custom/path/readme.j2",
            output_path="/custom/path/README.md"
        )
        generator.generate_readme()
"""

import logging
from pathlib import Path
from typing import Optional

import yaml

from robocoin_dataset.metadata.collect import InfoCollector
from robocoin_dataset.readme._logging import setup_readme_logger
from robocoin_dataset.readme._utils import load_collected_info_yaml
from robocoin_dataset.readme._template_utils import (
    load_jinja2_template,
    render_template,
    write_readme_file,
)
from robocoin_dataset.utils.log_config import log_error, log_success  # noqa: F401


# ============================================================================
# Main ReadmeGenerator Class
# ============================================================================


class ReadmeGenerator:
    """
    README Generator class for generating README.md files via a two-phase pipeline.

    Phase 1 - Collect (delegated to InfoCollector):
        Reads local_dataset_info.yaml + all dataset source files, resolves every
        schema field, and writes the result as a flat info.yaml inside the dataset
        directory.  This step is idempotent: running it again overwrites info.yaml
        with fresh data.

    Phase 2 - Render:
        Reads the collected info.yaml (flat key->value YAML), loads the Jinja2
        template, renders the template, and writes README.md.

    Attributes:
        dataset_path (Path): Root directory of the dataset (hardlink dir).
        local_dataset_info_path (Path): Resolved path to local_dataset_info.yaml,
            used as input to InfoCollector in Phase 1.
        collected_info_yaml_path (Path): Path where Phase 1 writes info.yaml and
            Phase 2 reads from (default: dataset_path/info.yaml).
        template_path (Path): Path to the Jinja2 readme.j2 template.
        output_path (Path): Output path for README.md.
        log_dir (Path): Directory for audit log files.
        logger (logging.Logger): Logger instance.

    Usage:
        Basic usage:
            generator = ReadmeGenerator(dataset_path="/path/to/dataset")
            generator.generate_readme()

        With custom paths:
            generator = ReadmeGenerator(
                dataset_path="/path/to/dataset",
                local_dataset_info_path="/custom/path/local_dataset_info.yaml",
                template_path="/custom/path/readme.j2",
                output_path="/custom/path/README.md",
            )
            generator.generate_readme()
    """

    def __init__(
        self,
        dataset_path: str | Path,
        local_dataset_info_path: Optional[str | Path] = None,
        template_path: Optional[str | Path] = None,
        output_path: Optional[str | Path] = None,
        log_dir: Optional[str | Path] = None,
        local_dataset_info_filename: str = "local_dataset_info.yaml",
        # Backward-compat alias kept so existing callers using info_yaml_path= still work.
        info_yaml_path: Optional[str | Path] = None,
    ) -> None:
        """
        Initialize ReadmeGenerator.

        Input:
            dataset_path (str | Path): Root directory of the dataset (hardlink dir).
            local_dataset_info_path (Optional[str | Path]): Explicit path to
                local_dataset_info.yaml (input for Phase 1 InfoCollector).
                If None, searched at dataset_path first level.
            template_path (Optional[str | Path]): Path to readme.j2 template.
                If None, defaults to package assets/readme.j2.
            output_path (Optional[str | Path]): Output path for README.md.
                If None, defaults to dataset_path/README.md.
            log_dir (Optional[str | Path]): Log file directory.
                If None, defaults to dataset_path/logs/gen_readme.
            local_dataset_info_filename (str): Filename to locate at dataset_path
                first level (default: "local_dataset_info.yaml").
            info_yaml_path (Optional[str | Path]): Backward-compat alias for
                local_dataset_info_path. Ignored if local_dataset_info_path is set.

        Output:
            None.

        Logic:
            1. Resolve dataset_path.
            2. Setup logger (early, for diagnostics).
            3. Validate dataset_path exists and is a directory.
            4. Soft-validate that directory name matches hardlink convention.
            5. Instantiate InfoCollector (which locates local_dataset_info.yaml and
               validates all inputs for Phase 1).
            6. Capture the collected info.yaml output path from InfoCollector.
            7. Set template_path and output_path defaults.

        Raises:
            FileNotFoundError: If dataset_path or local_dataset_info.yaml not found.
            ValueError: If dataset_path is not a directory.
        """
        self.dataset_path = Path(dataset_path).expanduser().resolve()
        module_root = Path(__file__).resolve().parent

        if log_dir is None:
            self.log_dir = self.dataset_path / "logs" / "gen_readme"
        else:
            self.log_dir = Path(log_dir).expanduser().resolve()

        self.logger = setup_readme_logger(
            log_dir=self.log_dir,
            script_name="gen_readme",
            level=logging.INFO,
            console_output=True,
        )

        if not self.dataset_path.exists():
            error_msg = f"[INIT] Dataset directory does not exist: {self.dataset_path}"
            log_error(self.logger, error_msg)
            raise FileNotFoundError(error_msg)

        if not self.dataset_path.is_dir():
            error_msg = f"[INIT] Dataset path is not a directory: {self.dataset_path}"
            log_error(self.logger, error_msg)
            raise ValueError(error_msg)


        # Resolve local_dataset_info_path: prefer explicit param, fall back to alias.
        _local_info_param = local_dataset_info_path if local_dataset_info_path is not None else info_yaml_path

        # Phase 1 delegate: InfoCollector owns source location and schema resolution.
        # It also locates local_dataset_info.yaml and validates the dataset path.
        self._info_collector = InfoCollector(
            dataset_path=self.dataset_path,
            local_dataset_info_path=_local_info_param,
            log_dir=self.log_dir,
            local_dataset_info_filename=local_dataset_info_filename,
        )

        # Phase 2 reads from where Phase 1 writes.
        self.collected_info_yaml_path = self._info_collector.output_info_yaml_path

        if template_path is None:
            self.template_path = module_root / "assets" / "readme.j2"
        else:
            self.template_path = Path(template_path).expanduser().resolve()

        if output_path is None:
            self.output_path = self.dataset_path / "README.md"
        else:
            self.output_path = Path(output_path).expanduser().resolve()

        self.logger.info("[INIT] ReadmeGenerator initialized")
        self.logger.info(f"[INIT] Dataset path: {self.dataset_path}")
        self.logger.info(
            f"[INIT] Local dataset info path: {self._info_collector.local_dataset_info_path}"
        )
        self.logger.info(f"[INIT] Collected info.yaml path: {self.collected_info_yaml_path}")
        self.logger.info(f"[INIT] Template path: {self.template_path}")
        self.logger.info(f"[INIT] Output path: {self.output_path}")
        self.logger.info(f"[INIT] Log directory: {self.log_dir}")

    def generate_readme(self) -> Path:
        """
        Generate README.md via two-phase pipeline: check/collect info.yaml -> render.

        Input:
            None (uses instance attributes).

        Output:
            Path: Absolute path to the generated README.md file.

        Logic:
            Phase 1 - Check/Collect info.yaml:
                1. Check if info.yaml exists at collected_info_yaml_path.
                2. If not, run InfoCollector.collect() to generate it.
                3. If yes, log that it already exists (skip regeneration).

            Phase 2 - Render:
                4. Read the info.yaml using load_collected_info_yaml().
                5. Load the Jinja2 readme.j2 template.
                6. Render the template with the loaded context.
                7. Write README.md to output_path.
                8. Return output_path.

        Usage:
            Main entry point after initializing ReadmeGenerator.

        Raises:
            FileNotFoundError: If template file or info.yaml is missing.
            yaml.YAMLError: If YAML parsing fails.
            jinja2.TemplateError: If template rendering fails.
            IOError: If file writing fails.
        """
        try:
            self.logger.info("[GENERATE] Starting README generation pipeline")

            # ---- Phase 1: Check/Collect info.yaml ----
            self.logger.info(
                "[GENERATE] Phase 1: Check/collect info.yaml"
            )
            if not self.collected_info_yaml_path.exists():
                self.logger.info(
                    f"[GENERATE] info.yaml not found at {self.collected_info_yaml_path}. "
                    "Generating via InfoCollector..."
                )
                self._info_collector.collect()
                self.logger.info(
                    f"[GENERATE] Generated info.yaml: {self.collected_info_yaml_path}"
                )
            else:
                self.logger.info(
                    f"[GENERATE] info.yaml already exists: {self.collected_info_yaml_path}. "
                    "Skipping collection."
                )

            # ---- Phase 2: Render ----
            self.logger.info("[GENERATE] Phase 2: Rendering README from info.yaml")

            self.logger.info("[GENERATE] Step 2.1: Loading context from info.yaml")
            context_data = load_collected_info_yaml(
                info_yaml_path=self.collected_info_yaml_path,
                logger=self.logger,
            )
            self.logger.info(
                f"[GENERATE] Loaded context with {len(context_data)} fields"
            )

            self.logger.info("[GENERATE] Step 2.2: Loading Jinja2 template")
            template = load_jinja2_template(self.template_path, self.logger)

            self.logger.info("[GENERATE] Step 2.3: Rendering template with context")
            rendered_content = render_template(template, context_data, self.logger)

            self.logger.info("[GENERATE] Step 2.4: Writing README.md")
            write_readme_file(rendered_content, self.output_path, self.logger)

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
    Create ReadmeGenerator instance from a YAML configuration file.

    Input:
        config_path (str | Path): Path to YAML configuration file.
            Supported keys:
                dataset_path (required): Root dataset directory.
                local_dataset_info_path (optional): Explicit path to
                    local_dataset_info.yaml.  Also accepted as the legacy key
                    info_yaml_path for backward compatibility.
                template_path (optional): Path to readme.j2 template.
                output_path (optional): Path for output README.md.
                log_dir (optional): Log file directory.

    Output:
        ReadmeGenerator: Initialized ReadmeGenerator instance.

    Logic:
        1. Load configuration from YAML file.
        2. Extract and validate parameters.
        3. Create and return ReadmeGenerator instance.

    Raises:
        FileNotFoundError: If config file does not exist.
        yaml.YAMLError: If config file parsing fails.
        KeyError: If required 'dataset_path' is missing.
    """
    config_path = Path(config_path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if config is None:
            raise ValueError(f"Configuration file is empty: {config_path}")

        if "dataset_path" not in config:
            raise KeyError("Required parameter 'dataset_path' is missing in config file")

        # Prefer canonical key; fall back to legacy alias.
        local_dataset_info_path = config.get("local_dataset_info_path") or config.get(
            "info_yaml_path"
        )
        template_path = config.get("template_path")
        output_path = config.get("output_path")
        log_dir = config.get("log_dir")

        return ReadmeGenerator(
            dataset_path=config["dataset_path"],
            local_dataset_info_path=local_dataset_info_path,
            template_path=template_path,
            output_path=output_path,
            log_dir=log_dir,
        )

    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse configuration file {config_path}: {e}") from e
    except Exception as e:
        raise RuntimeError(
            f"Unexpected error loading configuration file {config_path}: {e}"
        ) from e


def create_from_cli_args(
    dataset_path: str | Path,
    local_dataset_info_path: Optional[str | Path] = None,
    template_path: Optional[str | Path] = None,
    output_path: Optional[str | Path] = None,
    log_dir: Optional[str | Path] = None,
    # Backward-compat alias.
    info_yaml_path: Optional[str | Path] = None,
) -> ReadmeGenerator:
    """
    Create ReadmeGenerator instance from CLI arguments.

    Input:
        dataset_path (str | Path): Root dataset directory (required).
        local_dataset_info_path (Optional[str | Path]): Explicit path to
            local_dataset_info.yaml. If None, auto-located at dataset_path first level.
        template_path (Optional[str | Path]): Path to readme.j2 template.
        output_path (Optional[str | Path]): Path for output README.md.
        log_dir (Optional[str | Path]): Log file directory.
        info_yaml_path (Optional[str | Path]): Backward-compat alias for
            local_dataset_info_path.

    Output:
        ReadmeGenerator: Initialized ReadmeGenerator instance.

    Logic:
        1. Validate required parameters.
        2. Create and return ReadmeGenerator instance.

    Usage:
        generator = create_from_cli_args(
            dataset_path="/path/to/dataset",
            local_dataset_info_path="/custom/path/local_dataset_info.yaml",
        )
        generator.generate_readme()

    Raises:
        ValueError: If required 'dataset_path' is None or empty.
    """
    if not dataset_path:
        raise ValueError("Required parameter 'dataset_path' cannot be None or empty")

    _local_info = local_dataset_info_path if local_dataset_info_path is not None else info_yaml_path

    return ReadmeGenerator(
        dataset_path=dataset_path,
        local_dataset_info_path=_local_info,
        template_path=template_path,
        output_path=output_path,
        log_dir=log_dir,
    )
