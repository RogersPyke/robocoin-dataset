"""
RoboCoin Datasets Generate README.md for Single Datasets

This module provides a function for generating README.md files for individual
datasets during the upload process. It uses the SingleDatasetReadmeGenerator which works
directly with hardlink paths without requiring root_path manipulation.

Usage:
  Called internally during upload process, or standalone with:
  python -m robocoin.datasets.gen_readme --config configs/gen_readme.yaml
"""

import logging
import traceback
from pathlib import Path
from typing import Any

import draccus
from tqdm import tqdm

from robocoin_dataset.hub_upload.lerobot.local_datasets_util import (
    LocalDsReadmeConfig,
    LocalDsReadmeUtil,
)
from robocoin_dataset.prepare_metadata.unified_metadata_def import UnifiedMetadata

from .single_dataset_readme_generator import SingleDatasetReadmeGenerator


def gen_readme(
    hardlink_path: Path,
    dataset_info_root_path: Path | None,
    logger: logging.Logger | None = None,
    metadata: UnifiedMetadata | dict[str, Any] | None = None,
) -> tuple[bool, str]:
    """
    Generate README.md for a single dataset using direct hardlink path.

    This function uses SingleDatasetReadmeGenerator which reads directly from the
    hardlink path without requiring root_path setup, eliminating the need for
    temporary root_path manipulation.

    Args:
        hardlink_path: Path to the hardlink directory (full dataset path)
        dataset_info_root_path: Path containing the dataset info YAML files
        logger: Optional logger instance

    Returns:
        tuple[bool, str]: (success status, error message if failed or empty string if success)
    """
    try:
        # Use the new single-dataset generator (no root_path manipulation needed!)
        generator = SingleDatasetReadmeGenerator(
            dataset_path=hardlink_path,
            dataset_info_root_path=dataset_info_root_path,
            logger=logger,
        )

        # Generate README (prefer aggregated metadata when provided)
        success, error = generator.generate_readme(metadata=metadata)

        if success:
            readme_path = hardlink_path / "README.md"
            tqdm.write(f"      ✅ README: {readme_path}")
            if logger:
                logger.debug(f"{hardlink_path.name}: Generated README at {readme_path}")
        else:
            tqdm.write(f"      ❌ README generation failed: {error}")

        return success, error

    except Exception as e:
        tb = traceback.format_exc()
        error_msg = f"README generation failed: {e}\n\nFull traceback:\n{tb}"
        tqdm.write(f"      ❌ README generation failed: {e}")
        if logger:
            logger.error(f"{hardlink_path.name}: {error_msg}")
        return False, error_msg


if __name__ == "__main__":
    """
    Main entry point for the batch README generator.

    Parses command line configuration and runs the batch README generation process.
    This uses the original LocalDsReadmeUtil class for batch generation.
    """
    config = draccus.parse(LocalDsReadmeConfig)
    generator = LocalDsReadmeUtil(config)
    generator.generate_readmes()
    pass
