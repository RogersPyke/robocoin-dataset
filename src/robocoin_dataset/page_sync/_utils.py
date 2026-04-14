"""
Script utilities for page sync and dataset upload.

Page sync consumes info.yaml only. Dataset name and display fields are read
exclusively from the collected info.yaml (dataset_name); DB convert_path
is not used for naming.
"""

import logging
from pathlib import Path

######## ACTUAL OPERATION ########

# ------- DATASET NAME FROM INFO.YAML -------#


def _get_dataset_name_from_info_yaml(info_yaml_path: str, logger: logging.Logger) -> str:
    """
    Read dataset_name from a collected info.yaml file.

    Uses info.yaml key "dataset_name". No fallback is allowed.

    Input:
        info_yaml_path: Absolute path to info.yaml produced by metadata collect.
        logger: Logger for diagnostics.

    Output:
        str: Non-empty name used for asset filenames and consolidated keys.

    Raises:
        FileNotFoundError, ValueError, yaml.YAMLError: From load_collected_info_yaml.
    """
    from robocoin_dataset.readme._utils import load_collected_info_yaml

    data = load_collected_info_yaml(Path(info_yaml_path), logger)
    name = (data.get("dataset_name") or "").strip()
    if name:
        return name
    raise ValueError(f"dataset_name is missing or empty in info.yaml: {info_yaml_path}")


# ------- VALIDATION -------#


def _validate_exist(info_yaml_path: str | None, hardlink_path: str | None) -> bool:
    """
    Validate that both info_yaml_path and hardlink_path exist.

    Returns:
      bool: True if BOTH exist, False otherwise
    """
    _logger = logging.getLogger(__name__)
    # Check if both paths are provided
    if not info_yaml_path or not hardlink_path:
        _logger.debug(f"Missing paths - info_yaml_path: {info_yaml_path}, hardlink_path: {hardlink_path}")
        return False
    # Check if info_yaml_path exists
    yaml_file = Path(info_yaml_path)
    if not yaml_file.exists():
        _logger.debug(f"YAML file does not exist: {info_yaml_path}")
        return False
    # Check if hardlink_path exists
    hardlink_dir = Path(hardlink_path)
    if not hardlink_dir.exists():
        _logger.debug(f"Hardlink directory does not exist: {hardlink_path}")
        return False
    _logger.debug("Both paths validated successfully")
    return True


# ------- YAML OPERATION -------#


def _copy_info_yaml(
    src_info_yaml_path: str,
    dst_yaml_path: str,
) -> None:
    """
    Copy a pre-generated info.yaml file into page assets/dataset_info.
    """
    import shutil

    _logger = logging.getLogger(__name__)
    src_yaml = Path(src_info_yaml_path)
    dst_yaml = Path(dst_yaml_path)

    if not src_yaml.exists() or not src_yaml.is_file():
        raise FileNotFoundError(f"Source info.yaml not found: {src_yaml}")

    dst_yaml.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_yaml, dst_yaml)
    _logger.debug("Copied info.yaml from %s to %s", src_yaml, dst_yaml)


# ------- VIDEO OPERATION -------#


def _sample_one_video_path(hardlink_path: str) -> str | None:
    """
    Sample one video path from the dataset root directory (priority: high/top/head/front).

    Delegates to metadata._vid_coll.get_one_video_path so selection logic stays
    in one place. Returns absolute path to one .mp4 for page_sync compression/display.

    Input:
        hardlink_path (str): Dataset root in LeRobot format.

    Output:
        str | None: Absolute path to one sampled video, or None if none found.
    """
    _logger = logging.getLogger(__name__)
    from robocoin_dataset.metadata._vid_coll import get_one_video_path

    return get_one_video_path(
        dataset_path=hardlink_path,
        logger=_logger,
        random=True,
    )


def _compress_video_to_dst(
    selected_video_path: str,
    dst_path: str,
    crf: int =18,
    force_update: bool = False,
) -> None:
    """
    Compress a single video file from source path to destination path using CRF (Constant Rate Factor).

    CRF (Constant Rate Factor) is a quality-based encoding method that maintains consistent visual
    quality across the video. Lower CRF values mean better quality but larger file sizes.
    - CRF 18: visually lossless (very large files)
    - CRF 23: high quality (default, good balance)
    - CRF 28: acceptable quality (smaller files)

    INPUT:
    selected_video_path, -> the sampled, actual video path. point DIRECTLY at the video file.
    dst_path, -> the dst path to compress the video file.(in assets/dataset_info/videos/)
    crf, -> CRF value for video compression (default: 18, range: 0-51, lower = better quality).
    force_update, -> if True, always regenerate; if False, skip if file exists (default: False).
    OUTPUT:
    None, execute the compress and copying operation.
    """
    import subprocess

    video_file = Path(selected_video_path)
    dst_video_path = Path(dst_path) / video_file.name

    _logger = logging.getLogger(__name__)

    # Skip if file exists and force_update is False
    if not force_update and dst_video_path.exists():
        _logger.info(f"Video already exists at {dst_video_path}, skipping compression")
        return

    _logger.debug(f"Video file: {video_file}")
    _logger.debug(f"Destination: {dst_video_path}")

    # Check if source video file exists
    if not video_file.exists():
        _logger.error(f"Source video file does not exist: {selected_video_path}")
        raise FileNotFoundError(f"Source video file not found: {selected_video_path}")

    # Create destination directory if it doesn't exist
    dst_video_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Get original file size for logging
        original_size_kb = video_file.stat().st_size / 1024
        _logger.debug(f"Original video size: {original_size_kb:.2f} KB")

        # Build ffmpeg command with CRF-based encoding
        compress_cmd = [
            "ffmpeg",
            "-i",
            str(video_file),
            "-c:v",
            "libx264",  # Use H.264 codec
            "-crf",
            str(crf),  # Constant Rate Factor for quality control
            "-preset",
            "medium",  # Balanced encoding speed/quality
            "-pix_fmt",
            "yuv420p",  # Ensure compatibility
            "-movflags",
            "+faststart",  # Optimize for web playback
            "-y",  # Overwrite output file if exists
            str(dst_video_path),
        ]

        _logger.debug(f"ffmpeg command: {' '.join(compress_cmd)}")
        _logger.info(f"Starting video compression with CRF={crf} (this may take a while)...")
        subprocess.run(compress_cmd, check=True, capture_output=True, timeout=300)

        # Check output size
        if dst_video_path.exists():
            output_size_kb = dst_video_path.stat().st_size / 1024
            _logger.info(
                f"Successfully compressed {video_file.name}: "
                f"{original_size_kb:.2f}KB -> {output_size_kb:.2f}KB (CRF={crf})"
            )
        else:
            _logger.warning("Compressed file created but size check failed")
            _logger.info(f"Compression completed for {video_file.name}")

    except subprocess.TimeoutExpired:
        _logger.error(f"Video compression timed out for {video_file.name}")
        # Clean up partial output file if it exists
        if dst_video_path.exists():
            dst_video_path.unlink()
        raise RuntimeError(f"Video compression timed out for {video_file.name}")
    except subprocess.CalledProcessError as e:
        _logger.error(f"Failed to compress {video_file.name}: {e}")
        error_output = e.stderr.decode() if e.stderr else "N/A"
        _logger.error(f"ffmpeg stderr: {error_output}")
        # Clean up partial output file if it exists
        if dst_video_path.exists():
            dst_video_path.unlink()
        raise RuntimeError(f"Video compression failed for {video_file.name}: {e}")
    except Exception as e:
        _logger.error(f"Error processing {video_file.name}: {e}", exc_info=True)
        # Clean up partial output file if it exists
        if dst_video_path.exists():
            dst_video_path.unlink()
        raise


def _align_video_name_with_yaml(yaml_path: str, video_path: str, dataset_name: str) -> None:
    """
    Align YAML and video filenames to match dataset name for page script compatibility.
    Step 1: Check if YAML is named dataset_name.yaml, if not, rename it.
    Step 2: Check if video is named dataset_name.mp4, if not, rename it.
    """

    _logger = logging.getLogger(__name__)

    # Step 1: Check and rename YAML file if necessary
    src_yaml = Path(yaml_path)
    if not src_yaml.exists():
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")

    expected_yaml_name = f"{dataset_name}.yaml"
    if src_yaml.name != expected_yaml_name:
        dst_yaml = src_yaml.parent / expected_yaml_name
        src_yaml.rename(dst_yaml)
        _logger.debug(f"Renamed YAML from {src_yaml.name} to {dst_yaml.name}")
    else:
        _logger.debug(f"YAML already named correctly: {src_yaml.name}")

    # Step 2: Check and rename video file if necessary
    src_video = Path(video_path)
    if not src_video.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    expected_video_name = f"{dataset_name}{src_video.suffix}"
    if src_video.name != expected_video_name:
        dst_video = src_video.parent / expected_video_name
        src_video.rename(dst_video)
        _logger.debug(f"Renamed video from {src_video.name} to {dst_video.name}")
    else:
        _logger.debug(f"Video already named correctly: {src_video.name}")


def _gen_video_thumbnail(
    video_path: str,
    thumbnail_dir: str,
    force_update: bool = False,
) -> None:
    """
    Generate a thumbnail image from a video file.
    Extracts the first frame of the video and saves it as a JPEG image.

    INPUT:
    video_path -> path to the video file
    thumbnail_dir -> directory to save the thumbnail image
    force_update -> if True, always regenerate; if False, skip if file exists (default: False)

    OUTPUT:
    None, saves thumbnail image with the same name as the video (with .jpg extension)
    """
    import subprocess

    _logger = logging.getLogger(__name__)

    video_file = Path(video_path)
    thumbnail_dir_path = Path(thumbnail_dir)
    thumbnail_dir_path.mkdir(parents=True, exist_ok=True)

    thumbnail_path = thumbnail_dir_path / f"{video_file.stem}.jpg"

    # Skip if file exists and force_update is False
    if not force_update and thumbnail_path.exists():
        _logger.info(f"Thumbnail already exists at {thumbnail_path}, skipping generation")
        return

    subprocess.run(
        ["ffmpeg", "-i", str(video_file), "-vframes", "1", "-q:v", "2", "-y", str(thumbnail_path)],
        check=True,
        capture_output=True,
        timeout=60,
    )
    _logger.debug(f"Generated thumbnail: {thumbnail_path}")


# ------- CONSOLIDATION -------#


def _gen_consolidation(dataset_info_dir: str, output_path: str) -> None:
    """
    Generate consolidated_datasets.json by reading all YAML files from dataset_info directory
    and combining their metadata into a single JSON file.

    INPUT:
    dataset_info_dir -> path to the directory containing YAML files
    output_path -> path to write the consolidated JSON file

    OUTPUT:
    None, writes consolidated_datasets.json with all metadata
    """
    import json

    import yaml

    _logger = logging.getLogger(__name__)

    dataset_info_path = Path(dataset_info_dir)
    output_file = Path(output_path)

    if not dataset_info_path.exists():
        _logger.error(f"Dataset info directory does not exist: {dataset_info_dir}")
        raise FileNotFoundError(f"Dataset info directory not found: {dataset_info_dir}")

    # Find all YAML files
    yaml_files = list(dataset_info_path.glob("*.yaml")) + list(dataset_info_path.glob("*.yml"))
    _logger.info(f"Found {len(yaml_files)} YAML files to consolidate")

    if not yaml_files:
        _logger.warning("No YAML files found to consolidate")
        consolidated_data = {}
    else:
        consolidated_data = {}

        for yaml_file in yaml_files:
            try:
                _logger.debug(f"Reading YAML file: {yaml_file}")
                with open(yaml_file, encoding="utf-8") as f:
                    data = yaml.safe_load(f)

                if not isinstance(data, dict):
                    raise ValueError(
                        f"YAML root is not a mapping in {yaml_file}: {type(data).__name__}"
                    )

                # Use the filename (without extension) as the key
                dataset_name = yaml_file.stem
                consolidated_data[dataset_name] = dict(data)
                _logger.debug(f"Added {dataset_name} to consolidated data")

            except Exception as e:  # noqa: PERF203
                _logger.error(f"Failed to read or parse {yaml_file}: {e}", exc_info=True)
                raise

    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Write consolidated data to JSON
    _logger.debug(f"Writing consolidated data to {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(consolidated_data, f, indent=2, ensure_ascii=False)

    _logger.info(f"Successfully wrote consolidated datasets to {output_file}")


def _gen_data_index(dataset_info_dir: str, output_path: str) -> None:
    """
    Generate data_index.json by listing all YAML files from dataset_info directory.

    INPUT:
    dataset_info_dir -> path to the directory containing YAML files
    output_path -> path to write the data index JSON file

    OUTPUT:
    None, writes data_index.json with list of all YAML files
    """
    import json

    _logger = logging.getLogger(__name__)

    dataset_info_path = Path(dataset_info_dir)
    output_file = Path(output_path)

    if not dataset_info_path.exists():
        _logger.error(f"Dataset info directory does not exist: {dataset_info_dir}")
        raise FileNotFoundError(f"Dataset info directory not found: {dataset_info_dir}")

    # Find all YAML files
    yaml_files = list(dataset_info_path.glob("*.yaml")) + list(dataset_info_path.glob("*.yml"))
    _logger.info(f"Found {len(yaml_files)} YAML files for indexing")

    # Create list of dataset names (filenames without extension)
    data_index = {
        "datasets": sorted([yaml_file.stem for yaml_file in yaml_files]),
        "count": len(yaml_files),
    }

    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Write index to JSON
    _logger.debug(f"Writing data index to {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data_index, f, indent=2, ensure_ascii=False)

    _logger.info(f"Successfully wrote data index to {output_file} with {len(yaml_files)} datasets")


def _copy_robot_aliases_and_exclude(info_dir: str) -> None:
    """
    Copy robot_aliases.json and exclude.json into the page info directory
    from robocoin_dataset.metadata.assets only.
    """
    import shutil
    from importlib import resources

    _logger = logging.getLogger(__name__)
    dst_dir = Path(info_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_asset_path(filename: str) -> Path:
        """Resolve asset path from metadata assets package only."""
        try:
            package_file = resources.files("robocoin_dataset.metadata.assets").joinpath(filename)
            resolved = Path(str(package_file))
            if resolved.exists():
                return resolved
        except Exception:  # noqa: BLE001
            _logger.exception("Failed to access metadata assets package while resolving %s", filename)
            raise

        expected = "robocoin_dataset.metadata.assets"
        _logger.error("%s resource missing under package %s", filename, expected)
        raise FileNotFoundError(f"Failed to locate {filename} under package {expected}")

    robot_aliases_src = _resolve_asset_path("robot_aliases.json")
    robot_aliases_dst = dst_dir / "robot_aliases.json"
    shutil.copy2(robot_aliases_src, robot_aliases_dst)
    _logger.info("Copied %s to %s", robot_aliases_src, robot_aliases_dst)

    exclude_src = _resolve_asset_path("exclude.json")
    exclude_dst = dst_dir / "exclude.json"
    shutil.copy2(exclude_src, exclude_dst)
    _logger.info("Copied %s to %s", exclude_src, exclude_dst)
