"""
README Generator Module.

This module implements README rendering from an existing `info.yaml`.
It never triggers metadata collection.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import yaml

from robocoin_dataset.readme._logging import setup_readme_logger
from robocoin_dataset.readme._template_utils import (
    load_jinja2_template,
    render_template,
    write_readme_file,
)
from robocoin_dataset.readme._utils import load_collected_info_yaml
from robocoin_dataset.utils.log_config import log_error, log_success


class ReadmeGenerator:
    """
    Render README.md from an existing info.yaml and Jinja2 template.
    """

    def __init__(
        self,
        dataset_path: str | Path,
        template_path: Optional[str | Path] = None,
        output_path: Optional[str | Path] = None,
        log_dir: Optional[str | Path] = None,
    ) -> None:
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

        self.collected_info_yaml_path = self.dataset_path / "info.yaml"
        if template_path is None:
            self.template_path = module_root / "assets" / "readme.j2"
        else:
            self.template_path = Path(template_path).expanduser().resolve()

        if output_path is None:
            self.output_path = self.dataset_path / "README.md"
        else:
            self.output_path = Path(output_path).expanduser().resolve()

        self.logger.info("[INIT] ReadmeGenerator initialized")
        self.logger.info("[INIT] Dataset path: %s", self.dataset_path)
        self.logger.info("[INIT] info.yaml path: %s", self.collected_info_yaml_path)
        self.logger.info("[INIT] Template path: %s", self.template_path)
        self.logger.info("[INIT] Output path: %s", self.output_path)

    def generate_readme(self) -> Path:
        """
        Render README.md from pre-collected info.yaml.
        """
        try:
            self.logger.info("[GENERATE] Starting README render stage")
            context_data = load_collected_info_yaml(
                info_yaml_path=self.collected_info_yaml_path,
                logger=self.logger,
            )
            template = load_jinja2_template(self.template_path, self.logger)
            rendered_content = render_template(template, context_data, self.logger)
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


def create_from_config_file(config_path: str | Path) -> ReadmeGenerator:
    """
    Create ReadmeGenerator from YAML config.
    Required key: dataset_path.
    """
    config_path = Path(config_path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if config is None:
        raise ValueError(f"Configuration file is empty: {config_path}")
    if "dataset_path" not in config:
        raise KeyError("Required parameter 'dataset_path' is missing in config file")

    return ReadmeGenerator(
        dataset_path=config["dataset_path"],
        template_path=config.get("template_path"),
        output_path=config.get("output_path"),
        log_dir=config.get("log_dir"),
    )


def create_from_cli_args(
    dataset_path: str | Path,
    template_path: Optional[str | Path] = None,
    output_path: Optional[str | Path] = None,
    log_dir: Optional[str | Path] = None,
) -> ReadmeGenerator:
    """
    Create ReadmeGenerator from CLI arguments.
    """
    if not dataset_path:
        raise ValueError("Required parameter 'dataset_path' cannot be None or empty")
    return ReadmeGenerator(
        dataset_path=dataset_path,
        template_path=template_path,
        output_path=output_path,
        log_dir=log_dir,
    )
