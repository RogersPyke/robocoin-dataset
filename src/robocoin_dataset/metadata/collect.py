"""
Metadata information collector for dataset info.yaml generation.

Purpose:
    Collect metadata from all dataset source files (local_dataset_info.yaml,
    meta/info.json, annotations/, etc.) and solidify them into a single flat
    info.yaml under the dataset root directory.

    This is the independent "Collect" stage. It has no dependencies on readme/
    modules, ensuring clean separation between Collect (metadata) and Render
    (README generation). Its output (info.yaml) is consumed by:
      - README generation (readme/gen_readme.py)
      - Web page asset generation (future downstream consumers)

Dependencies:
    - robocoin_dataset.metadata.collect_utils: Field resolution and context building
    - robocoin_dataset.metadata.logging: Collect stage logger setup
    - robocoin_dataset.metadata.yaml_utils: YAML file locating and loading
    - robocoin_dataset.utils.log_config: log_error, log_success
    - yaml: YAML serialization
    - pathlib: Path operations

Usage example:
    collector = InfoCollector(
        dataset_path="/data/my_dataset_qced_hardlink",
        local_dataset_info_path=None,  # auto-locate local_dataset_info.yaml
    )
    output_path = collector.collect()
    # -> writes /data/my_dataset_qced_hardlink/info.yaml
    # -> returns Path to the written file
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from robocoin_dataset.metadata.collect_utils import resolve_context_from_schema
from robocoin_dataset.metadata.logging import (
    setup_collect_logger,
    validate_hardlink_directory,
)
from robocoin_dataset.metadata.yaml_utils import locate_yaml_file
from robocoin_dataset.utils.log_config import log_error, log_success

# Filename of the generated output placed in the dataset root directory.
COLLECTED_INFO_YAML_FILENAME = "info.yaml"


# ============================================================================
# Internal helper
# ============================================================================


def _write_collected_info_yaml(
    context_data: Dict[str, Any],
    output_path: Path,
    logger: logging.Logger,
) -> None:
    """
    Serialize resolved context dict to YAML and write to file.

    Input:
        context_data (Dict[str, Any]): Fully resolved, flat metadata dict.
        output_path (Path): Absolute path to write the YAML file.
        logger (logging.Logger): Logger for audit messages.

    Output:
        None. Writes file to output_path.

    Logic:
        1. Create parent directories if needed.
        2. Dump context_data as YAML (allow_unicode, no flow style, preserve key order).
        3. On failure, log error and raise IOError.

    Raises:
        IOError: If writing fails for any reason.
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(
                context_data,
                f,
                allow_unicode=True,
                default_flow_style=False,
                sort_keys=False,
            )
        logger.info(f"[YAML_WRITE] Written collected info.yaml: {output_path}")
    except Exception as e:
        error_msg = f"[YAML_WRITE] Failed to write info.yaml at {output_path}: {e}"
        log_error(logger, error_msg)
        raise IOError(error_msg) from e


# ============================================================================
# Main InfoCollector class
# ============================================================================


class InfoCollector:
    """
    Collect all dataset metadata from distributed source files and write
    a unified, flat info.yaml to the dataset root directory.

    This class is the upstream "collect" phase. All downstream consumers
    (README generator, web page generator, etc.) read from the output info.yaml
    rather than re-reading all raw source files individually.

    Attributes:
        dataset_path (Path): Root directory of the dataset (hardlink dir).
        local_dataset_info_path (Path): Resolved path to local_dataset_info.yaml.
        schema_yaml_path (Path): Path to assets/info.yaml schema template (fixed).
        output_info_yaml_path (Path): Output path for the generated info.yaml.
        log_dir (Path): Directory for audit log files.
        logger (logging.Logger): Logger instance.

    Usage:
        collector = InfoCollector(
            dataset_path="/data/my_dataset_qced_hardlink",
        )
        info_yaml_path = collector.collect()
    """

    def __init__(
        self,
        dataset_path: str | Path,
        local_dataset_info_path: Optional[str | Path] = None,
        output_info_yaml_path: Optional[str | Path] = None,
        log_dir: Optional[str | Path] = None,
        local_dataset_info_filename: str = "local_dataset_info.yaml",
    ) -> None:
        """
        Initialize InfoCollector.

        Input:
            dataset_path (str | Path): Root directory of the dataset (hardlink dir).
            local_dataset_info_path (Optional[str | Path]): Explicit path to
                local_dataset_info.yaml. If None, searched at dataset_path first level.
            output_info_yaml_path (Optional[str | Path]): Output path for info.yaml.
                If None, defaults to dataset_path/info.yaml.
            log_dir (Optional[str | Path]): Log file directory.
                If None, defaults to dataset_path/logs/collect.
            local_dataset_info_filename (str): Filename to locate at dataset_path
                first level (default: "local_dataset_info.yaml").

        Output:
            None.

        Logic:
            1. Resolve and validate dataset_path.
            2. Setup logger (early, for diagnostics).
            3. Validate that directory appears to be a hardlink directory (soft warning).
            4. Locate local_dataset_info.yaml via priority-based search.
            5. Set schema_yaml_path (fixed to package assets/info.yaml).
            6. Set output_info_yaml_path.

        Raises:
            FileNotFoundError: If dataset_path or local_dataset_info.yaml not found.
            ValueError: If dataset_path is not a directory.
        """
        self.dataset_path = Path(dataset_path).expanduser().resolve()

        # schema template is always the package-bundled assets/info.yaml
        # collect.py is at: src/robocoin_dataset/metadata/collect.py
        # readme/assets/info.yaml is at: src/robocoin_dataset/readme/assets/info.yaml
        _readme_module_root = Path(__file__).resolve().parent.parent / "readme"
        self.schema_yaml_path = _readme_module_root / "assets" / "info.yaml"

        if log_dir is None:
            self.log_dir = self.dataset_path / "logs" / "collect"
        else:
            self.log_dir = Path(log_dir).expanduser().resolve()

        self.logger = setup_collect_logger(
            log_dir=self.log_dir,
            script_name="collect",
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

        validate_hardlink_directory(self.dataset_path, self.logger)

        # Locate local_dataset_info.yaml:
        # 1. If custom path provided, validate and use it.
        # 2. Otherwise search at dataset_path first level.
        self.local_dataset_info_path = locate_yaml_file(
            hardlink_dir=self.dataset_path,
            yaml_filename=local_dataset_info_filename,
            custom_yaml_path=local_dataset_info_path,
            logger=self.logger,
        )

        if output_info_yaml_path is None:
            self.output_info_yaml_path = self.dataset_path / COLLECTED_INFO_YAML_FILENAME
        else:
            self.output_info_yaml_path = Path(output_info_yaml_path).expanduser().resolve()

        self.logger.info("[INIT] InfoCollector initialized")
        self.logger.info(f"[INIT] Dataset path: {self.dataset_path}")
        self.logger.info(f"[INIT] Local dataset info path: {self.local_dataset_info_path}")
        self.logger.info(f"[INIT] Schema YAML path: {self.schema_yaml_path}")
        self.logger.info(f"[INIT] Output info.yaml path: {self.output_info_yaml_path}")
        self.logger.info(f"[INIT] Log directory: {self.log_dir}")

    def collect(self) -> Path:
        """
        Collect metadata from all sources and write to info.yaml.

        Input:
            None (uses instance attributes).

        Output:
            Path: Absolute path to the generated info.yaml file.

        Logic:
            1. Resolve all field values via resolve_context_from_schema from
               collect_utils, which reads: local_dataset_info.yaml, meta/info.json,
               annotations/*.jsonl, and applies auto-computed fields and
               template compatibility transforms.
            2. Write the resulting flat dict to output_info_yaml_path as YAML.
            3. Return output_info_yaml_path.

        Usage:
            collector = InfoCollector(dataset_path="/data/dataset_qced_hardlink")
            info_yaml_path = collector.collect()

        Raises:
            FileNotFoundError: If schema YAML or local_dataset_info.yaml is missing.
            IOError: If writing output info.yaml fails.
        """
        try:
            self.logger.info("[COLLECT] Starting metadata collection")

            self.logger.info("[COLLECT] Step 1: Resolving context from schema and source files")
            context_data = resolve_context_from_schema(
                schema_yaml_path=self.schema_yaml_path,
                dataset_path=self.dataset_path,
                local_dataset_info_path=self.local_dataset_info_path,
                logger=self.logger,
            )
            self.logger.info(
                f"[COLLECT] Resolved {len(context_data)} fields"
            )

            self.logger.info(
                f"[COLLECT] Step 2: Writing collected info.yaml to {self.output_info_yaml_path}"
            )
            _write_collected_info_yaml(
                context_data=context_data,
                output_path=self.output_info_yaml_path,
                logger=self.logger,
            )

            log_success(
                self.logger,
                f"[COLLECT] Metadata collection completed: {self.output_info_yaml_path}",
            )
            return self.output_info_yaml_path

        except Exception as e:
            error_msg = f"[COLLECT] Metadata collection failed: {e}"
            log_error(self.logger, error_msg)
            self.logger.debug("[COLLECT] Exception traceback:", exc_info=True)
            raise
