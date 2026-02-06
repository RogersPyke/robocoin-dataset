"""
Single Dataset README Generator - For On-Demand Generation During Upload

This module provides a class for generating README.md files for individual datasets
without requiring root_path manipulation. Designed for single-file generation during upload.
"""

import logging
from functools import cached_property
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, FileSystemLoader

from robocoin_dataset.hub_upload.config.constant import (
    DATASET_INFO_FILE,
    LEROBOT_META_INFO_FILE,
    README_FILE,
)
from robocoin_dataset.prepare_metadata.metadata_collect_utils import (
    generate_folder_structure,
)
from robocoin_dataset.prepare_metadata.unified_metadata_def import UnifiedMetadata


class SingleDatasetReadmeGenerator:
    """
    Generator for creating README.md files for individual datasets.

    This class works directly with hardlink paths without requiring root_path setup,
    making it ideal for on-demand generation during upload operations.

    Attributes:
        dataset_path (Path): Direct path to the dataset directory (hardlink)
        dataset_info_root_path (Path): Directory containing the dataset info YAML files
        logger (logging.Logger | None): Optional logger instance
    """

    def __init__(
        self,
        dataset_path: Path,
        dataset_info_root_path: Path | None,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize the single dataset README generator.

        Args:
            dataset_path: Direct path to the dataset directory (hardlink path)
            dataset_info_root_path: Directory containing the dataset info YAML files
            logger: Optional logger instance
        """
        self.dataset_path = Path(dataset_path)
        self.dataset_info_root_path = (
            Path(dataset_info_root_path) if dataset_info_root_path is not None else None
        )
        self.logger = logger

        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {self.dataset_path}")
        if not self.dataset_path.is_dir():
            raise NotADirectoryError(f"Dataset path is not a directory: {self.dataset_path}")

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
            raise FileNotFoundError(f"README template file {path} does not exist")
        return path

    @property
    def dataset_name(self) -> str:
        """Get the dataset name from the path."""
        return self.dataset_path.name

    def _get_meta_info_content(self) -> str:
        """
        Get the content of the meta info file.

        Returns:
            str: Content of the meta info file.

        Raises:
            FileNotFoundError: If meta info file does not exist.
        """
        meta_info_file = self.dataset_path / LEROBOT_META_INFO_FILE
        if not meta_info_file.exists():
            raise FileNotFoundError(f"Meta info file {meta_info_file} does not exist")
        return meta_info_file.read_text(encoding="utf-8")

    def _get_folder_structure(self) -> str:
        """
        Get the folder structure tree for the dataset.

        Returns:
            str: Formatted folder structure tree showing leaf directories with first 5 files.
        """
        return generate_folder_structure(self.dataset_path, max_files_per_dir=5)

    def _render_readme(self, context: dict[str, Any]) -> str:
        """
        Render README content from provided context dict.

        The context is typically generated from a UnifiedMetadata instance.
        """
        # Setup Jinja2 environment
        env = Environment(loader=FileSystemLoader(self.readme_template_file.parent))
        env.globals["get_meta_info_content"] = self._get_meta_info_content

        # Determine display name (strip hardlink suffix, or fall back to path in metadata)
        display_dataset_name = context.get("path") or (
            self.dataset_name.removesuffix("_qced_hardlink").removesuffix("_hardlink")
        )

        # Create a copy of context without dataset_name to avoid conflict
        # since we're passing dataset_name explicitly as a keyword argument
        context_copy = context.copy()
        context_copy.pop("dataset_name", None)

        return env.get_template(self.readme_template_file.name).render(
            dataset_name=display_dataset_name,
            **context_copy,
        )

    def generate_readme(
        self,
        metadata: UnifiedMetadata | dict[str, Any] | None = None,
    ) -> tuple[bool, str]:
        """
        Generate README file for the dataset using Jinja2 template.

        IMPORTANT:
            - This method ALWAYS OVERWRITES the existing README.md file.
            - When `metadata` is provided, this method only does parameter
              filling and template rendering. All heavy data collection and
              computation should be done in `metadata_collect.py` (via
              `create_unified_metadata`), and the resulting metadata is
              passed in as the context.

        Returns:
            tuple[bool, str]: (success status, error message if failed or empty string if success)
        """
        try:
            # Prefer using aggregated metadata object if provided
            if metadata is not None:
                if isinstance(metadata, UnifiedMetadata):
                    context: dict[str, Any] = metadata.to_dict()
                else:
                    context = dict(metadata)
                # NOTE:
                #   We intentionally DO NOT perform any extra computation here
                #   (such as regenerating folder structure). All such logic
                #   should happen in `metadata_collect.py` so that this
                #   generator focuses on "fill params + render".
            else:
                # Backward-compatible path: load dataset_info.yml and build context
                if self.dataset_info_root_path is None:
                    error_msg = (
                        "dataset_info_root_path is required when metadata is not provided"
                    )
                    if self.logger:
                        self.logger.error(f"{self.dataset_name}: {error_msg}")
                    return False, error_msg

                ds_info_file = (
                    self.dataset_info_root_path / self.dataset_name / DATASET_INFO_FILE
                )

                if not ds_info_file.exists():
                    error_msg = f"Dataset info file not found: {ds_info_file}"
                    if self.logger:
                        self.logger.error(f"{self.dataset_name}: {error_msg}")
                    return False, error_msg

                with open(ds_info_file, encoding="utf-8") as f:
                    ds_info: dict[str, Any] = yaml.safe_load(f)

                # Auto-generate structure if not provided in ds_info
                if "structure" not in ds_info or ds_info.get("structure") == "auto_generated":
                    ds_info["structure"] = self._get_folder_structure()

                context = ds_info

            # Render README from template
            readme_content = self._render_readme(context)

            # Write README file
            readme_file = self.dataset_path / README_FILE
            with open(readme_file, "w", encoding="utf-8") as f:
                f.write(readme_content)

            if self.logger:
                self.logger.debug(f"{self.dataset_name}: Generated README at {readme_file}")

            return True, ""

        except Exception as e:
            error_msg = f"README generation failed: {e}"
            if self.logger:
                self.logger.error(f"{self.dataset_name}: {error_msg}")
            return False, error_msg
