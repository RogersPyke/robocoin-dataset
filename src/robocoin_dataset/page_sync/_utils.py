"""
脚本旨在提供一些通用的工具函数，用于页面同步和数据集上传的辅助操作
"""

import logging
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

######## ACTUAL OPERATION ########

# ------- DATASET NAME GETTING -------#

def _get_dataset_name(session: Session, dataset_uuid: str) -> str | None:
    """
    Get dataset name from a dataset record using dataset_uuid.
    This is for the page script compatibility.

    Args:
        session: Database session
        dataset_uuid: UUID of the dataset to query

    Returns:
        Dataset name (basename of convert_path) or None if not found
    """
    from robocoin_dataset.database.models import DatasetDB

    _logger = logging.getLogger(__name__)

    _logger.debug(f"Querying for dataset name using dataset_uuid: {dataset_uuid}...")
    query = session.query(DatasetDB).filter(
        DatasetDB.dataset_uuid == dataset_uuid
    )
    item = query.first()

    if not item:
        _logger.warning(f"No dataset found with dataset_uuid: {dataset_uuid}")
        return None

    if not hasattr(item, 'convert_path') or not item.convert_path:
        _logger.warning(f"Dataset {dataset_uuid} found but convert_path is missing or empty")
        return None

    # Get the basename (ending) of the convert_path as dataset_name
    dataset_name = Path(item.convert_path).name
    _logger.debug(f"Retrieved dataset name: {dataset_name} for dataset_uuid: {dataset_uuid}")
    return dataset_name


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
    Sample one video path from the dataset root directory,
    identify the actual video path.

    Priority: searches folders containing "high", "top", or "head" first.
    Falls back to all observation.images.* folders if no match found.

    INPUT:
    hardlink_path, -> the dataset in lerobot foramt, sepecify to sample from where.
    OUTPUT:
    selected_video_path, -> the sampled, actual video path.

    Expects root directory structure:
    hardlink_path/
      videos/
        chunk-*/
          observation.images.*/*.mp4
    """
    import random

    _logger = logging.getLogger(__name__)
    root_path = Path(hardlink_path)

    if not root_path.exists():
        _logger.warning(f"Root directory does not exist: {hardlink_path}")
        return None

    videos_path = root_path / "videos"
    if not videos_path.exists():
        _logger.warning(f"Videos directory does not exist: {videos_path}")
        return None

    # Get all videos first
    all_videos = list(videos_path.glob("chunk-*/observation.images.*/*.mp4"))
    if not all_videos:
        _logger.warning(f"No videos found in any observation.images.* folders under {videos_path}")
        return None

    # Filter videos from priority folders (containing "high", "top", or "head")
    priority_keywords = ["high", "top", "head","front"]
    priority_videos = [
        v for v in all_videos if any(kw in str(v).lower() for kw in priority_keywords)
    ]

    # Use priority videos if found, otherwise use all videos
    video_files = priority_videos if priority_videos else all_videos
    selected_video_path = random.choice(video_files)
    _logger.info(f"Sampled video: {selected_video_path}")

    return str(selected_video_path)


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
                    _logger.warning("YAML root is not a mapping in %s, skip", yaml_file)
                    continue

                # Use the filename (without extension) as the key
                dataset_name = yaml_file.stem
                data_with_compat = dict(data)
                data_with_compat["legacy_compat"] = _build_legacy_compat_payload(
                    dataset_name=dataset_name,
                    data=data_with_compat,
                )
                consolidated_data[dataset_name] = data_with_compat
                _logger.debug(f"Added {dataset_name} to consolidated data")

            except Exception as e:  # noqa: PERF203
                _logger.error(f"Failed to read or parse {yaml_file}: {e}", exc_info=True)
                continue

    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Write consolidated data to JSON
    _logger.debug(f"Writing consolidated data to {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(consolidated_data, f, indent=2, ensure_ascii=False)

    _logger.info(f"Successfully wrote consolidated datasets to {output_file}")


def _build_legacy_compat_payload(dataset_name: str, data: dict[str, Any]) -> dict[str, Any]:
    """
    Build legacy-compatible fields for old page consumers.
    """
    path_value = data.get("dataset_name") or dataset_name
    structure_value = data.get("data_structure")
    if structure_value in (None, ""):
        structure_value = data.get("structure")

    task_instruction = data.get("task_instruction")
    if isinstance(task_instruction, list):
        tasks_value = "\n".join([str(item) for item in task_instruction if str(item).strip()])
    elif isinstance(task_instruction, str):
        tasks_value = task_instruction
    else:
        tasks_value = ""

    return {
        "path": path_value,
        "video_url": f"./assets/videos/{path_value}.mp4",
        "thumbnail_url": f"./assets/thumbnails/{path_value}.jpg",
        "robot_type": data.get("robot_name") or data.get("device_model") or "",
        "structure": structure_value or "",
        "tasks": tasks_value,
        "task_descriptions": data.get("sub_tasks") if isinstance(data.get("sub_tasks"), list) else [],
    }


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
    Copy the repository's robot_aliases.json and exclude.json into the page info directory.
    """
    import shutil

    _logger = logging.getLogger(__name__)
    assets_dir = Path(__file__).parent / "assets"

    # Copy robot_aliases.json
    robot_aliases_src = assets_dir / "robot_aliases.json"
    if not robot_aliases_src.exists():
        _logger.error("robot_aliases.json resource missing at %s", robot_aliases_src)
        raise FileNotFoundError(f"Failed to locate robot_aliases.json at {robot_aliases_src}")

    dst_dir = Path(info_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    robot_aliases_dst = dst_dir / "robot_aliases.json"
    shutil.copy2(robot_aliases_src, robot_aliases_dst)
    _logger.info("Copied %s to %s", robot_aliases_src, robot_aliases_dst)

    # Copy exclude.json
    exclude_src = assets_dir / "exclude.json"
    if not exclude_src.exists():
        _logger.error("exclude.json resource missing at %s", exclude_src)
        raise FileNotFoundError(f"Failed to locate exclude.json at {exclude_src}")

    exclude_dst = dst_dir / "exclude.json"
    shutil.copy2(exclude_src, exclude_dst)
    _logger.info("Copied %s to %s", exclude_src, exclude_dst)
