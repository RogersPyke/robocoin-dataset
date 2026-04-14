"""
Download statistics collection module for PageSync.

Purpose:
    Collect download statistics from HuggingFace and ModelScope for all datasets
    under a given organization. Generate a consolidated JSON file for web consumption.

Design:
    This module is the ONLY component in PageSync that depends on the external
    git submodule: src/download_stat/DownloadAnalyzer

    It consumes:
      - info.yaml files (from assets/dataset_info/) - REQUIRED, determines which
        datasets exist and their names for matching
      - DownloadAnalyzer submodule - fetches download counts from HF/MS APIs

    It produces:
      - assets/info/download_stats.json - aggregated download statistics

    The info.yaml files are produced by the Metadata Collection stage and consumed
    by PageSync. This module reads the dataset names from assets/dataset_info/*.yaml
    to match against the download statistics returned by DownloadAnalyzer.

    IMPLEMENTATION NOTE:
        This module calls DownloadAnalyzer via subprocess (not direct import).
        This avoids importing matplotlib/seaborn at module level, which would
        require installing visualization dependencies in the main environment.
        The submodule should have its own virtual environment with all required
        dependencies installed.

Dependencies:
    - src/download_stat/DownloadAnalyzer/run_stat.py (git submodule, called via subprocess)
    - pandas (for CSV reading)
    - Standard library: json, logging, datetime, pathlib, subprocess, sys

Usage:
    from robocoin_dataset.page_sync._download_stat import generate_download_stats_json

    generate_download_stats_json(
        dataset_info_dir="/path/to/assets/dataset_info",
        output_path="/path/to/assets/info/download_stats.json",
        hf_org_name="RoboCOIN",
        ms_org_name="RoboCOIN",
        logger=logger,
    )

Output format (download_stats.json):
    {
        "last_updated": "2026-04-14T12:00:00+08:00",
        "huggingface": {
            "org_name": "RoboCOIN",
            "total_downloads": 1234567,
            "total_likes": 5000,
            "dataset_count": 42,
            "datasets": {
                "dataset_name_1": {"downloads": 10000, "likes": 50},
                "dataset_name_2": {"downloads": 5000, "likes": 20}
            }
        },
        "modelscope": {
            "org_name": "RoboCOIN",
            "total_downloads": 234567,
            "dataset_count": 38,
            "datasets": {
                "dataset_name_1": {"downloads": 3000},
                "dataset_name_2": {"downloads": 1500}
            }
        },
        "combined_total_downloads": 1469134
    }
"""

import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Path to DownloadAnalyzer submodule
_DOWNLOAD_ANALYZER_PATH = (
    Path(__file__).resolve().parents[3] / "src" / "download_stat" / "DownloadAnalyzer"
)


def _get_existing_dataset_names(dataset_info_dir: str, logger: logging.Logger) -> set[str]:
    """
    Extract dataset names from existing info.yaml files in dataset_info directory.

    @input:
        dataset_info_dir [str]: Path to directory containing dataset YAML files.
        logger [logging.Logger]: Logger instance.

    @output:
        set[str]: Set of dataset names (filename stems without extension).

    @scenario:
        Determine which datasets exist locally for matching against download stats.
        DownloadAnalyzer returns all datasets from HF/MS org, but we only want to
        include those that exist in our local assets.
    """
    dataset_info_path = Path(dataset_info_dir)
    if not dataset_info_path.exists():
        logger.warning("Dataset info directory does not exist: %s", dataset_info_dir)
        return set()

    yaml_files = list(dataset_info_path.glob("*.yaml")) + list(dataset_info_path.glob("*.yml"))
    dataset_names = {yaml_file.stem for yaml_file in yaml_files}
    logger.info("Found %d dataset names from info.yaml files", len(dataset_names))
    return dataset_names


def _normalize_dataset_name(full_name: str) -> str:
    """
    Normalize dataset name by stripping organization prefix.

    @input:
        full_name [str]: Full dataset name, possibly with org prefix (e.g., "RoboCOIN/dataset_name").

    @output:
        str: Normalized dataset name without org prefix.

    @scenario:
        HF/MS APIs return dataset names as "org/dataset_name", but local info.yaml
        uses only "dataset_name". Strip the org prefix for matching.
    """
    if "/" in full_name:
        return full_name.split("/")[-1]
    return full_name


def _fetch_download_stats_from_submodule(
    hf_org_name: str,
    ms_org_name: str,
    temp_output_dir: str,
    logger: logging.Logger,
) -> dict[str, Any]:
    """
    Call DownloadAnalyzer submodule to fetch download statistics.

    @input:
        hf_org_name [str]: HuggingFace organization name.
        ms_org_name [str]: ModelScope organization name.
        temp_output_dir [str]: Temporary directory for DownloadAnalyzer output.
        logger [logging.Logger]: Logger instance.

    @output:
        dict[str, Any]: Raw result from DownloadAnalyzer.run_stat().

    @scenario:
        Fetch download statistics from HF/MS APIs via the DownloadAnalyzer submodule.
        This is the ONLY function that depends on the external submodule.

    IMPLEMENTATION NOTE:
        Uses subprocess to run the submodule script instead of direct import.
        This avoids importing matplotlib/seaborn at module level, which would
        require installing visualization dependencies in the main environment.
    """
    import subprocess

    submodule_path = _DOWNLOAD_ANALYZER_PATH / "run_stat.py"
    if not submodule_path.exists():
        logger.error("DownloadAnalyzer submodule not found at: %s", submodule_path)
        logger.error("Ensure submodule is initialized: git submodule update --init --recursive")
        raise FileNotFoundError(
            f"DownloadAnalyzer submodule not found at {submodule_path}. "
            "Run: git submodule update --init --recursive"
        )

    logger.info(
        "Calling DownloadAnalyzer via subprocess for org: HF=%s, MS=%s", hf_org_name, ms_org_name
    )

    # Use current Python interpreter. DownloadAnalyzer requires:
    # - matplotlib, seaborn (for visualization, but only used when generating plots)
    # - pandas, huggingface_hub, requests (for data fetching)
    #
    # If matplotlib is not installed, the script will fail. In that case,
    # either install it in the current environment or ensure system python
    # has it and modify this code to use /usr/bin/python3 instead.
    cmd = [
        sys.executable,
        str(submodule_path),
        "--org",
        hf_org_name,
        "--ms_org",
        ms_org_name,
        "--output",
        temp_output_dir,
        "--max_workers",
        "8",
    ]

    logger.debug("Subprocess command: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            logger.error("DownloadAnalyzer subprocess failed with exit code %d", result.returncode)
            if result.stderr:
                logger.error("stderr: %s", result.stderr)
            if result.stdout:
                logger.debug("stdout: %s", result.stdout)
            raise RuntimeError(
                f"DownloadAnalyzer subprocess failed with exit code {result.returncode}"
            )

        if result.stdout:
            logger.debug("DownloadAnalyzer stdout:\n%s", result.stdout)

    except subprocess.TimeoutExpired as e:
        logger.error("DownloadAnalyzer subprocess timed out after 300s")
        raise RuntimeError("DownloadAnalyzer subprocess timed out") from e
    except FileNotFoundError as e:
        logger.error("Failed to execute DownloadAnalyzer: %s", e)
        raise

    # Parse output directory to find generated files
    # run_stat.py prints output directory and totals
    output_lines = result.stdout.strip().split("\n") if result.stdout else []

    hf_total = 0
    ms_total = 0
    save_dir = temp_output_dir

    for line in output_lines:
        if "Output directory:" in line:
            save_dir = line.split(":", 1)[1].strip()
        elif "HuggingFace total:" in line:
            hf_total = int(line.split(":")[1].strip().replace(",", ""))
        elif "ModelScope total:" in line:
            ms_total = int(line.split(":")[1].strip().replace(",", ""))
        elif "Combined total:" in line:
            pass  # We'll compute this ourselves

    logger.info("DownloadAnalyzer completed. HF total: %d, MS total: %d", hf_total, ms_total)

    return {
        "save_dir": save_dir,
        "hf_csv": os.path.join(save_dir, "huggingface_dataset_downloads.csv"),
        "ms_csv": os.path.join(save_dir, "modelscope_dataset_downloads.csv"),
        "hf_total_downloads": hf_total,
        "ms_total_downloads": ms_total,
    }


def _read_csv_to_dict(csv_path: str, logger: logging.Logger) -> dict[str, dict[str, int]]:
    """
    Read download statistics CSV file into a dictionary.

    @input:
        csv_path [str]: Path to CSV file from DownloadAnalyzer.
        logger [logging.Logger]: Logger instance.

    @output:
        dict[str, dict[str, int]]: Mapping of dataset_name -> {downloads, likes}.

    @scenario:
        Parse CSV output from DownloadAnalyzer into structured dict for JSON serialization.
    """
    import pandas as pd

    if not os.path.exists(csv_path):
        logger.warning("CSV file not found: %s", csv_path)
        return {}

    try:
        df = pd.read_csv(csv_path)

        result = {}
        for _, row in df.iterrows():
            raw_name = str(row.get("dataset_name", ""))
            normalized_name = _normalize_dataset_name(raw_name)

            downloads = int(row.get("downloads", 0) or 0)
            likes = int(row.get("likes", 0) or 0)

            result[normalized_name] = {
                "downloads": downloads,
                "likes": likes,
            }

        logger.info("Parsed %d entries from CSV: %s", len(result), csv_path)
        return result

    except Exception as e:
        logger.error("Failed to read CSV %s: %s", csv_path, e, exc_info=True)
        return {}


def _filter_by_existing_datasets(
    stats_dict: dict[str, dict[str, int]],
    existing_names: set[str],
    logger: logging.Logger,
) -> dict[str, dict[str, int]]:
    """
    Filter download stats to only include datasets that exist locally.

    @input:
        stats_dict [dict]: Full download stats from HF/MS.
        existing_names [set]: Set of dataset names that exist in local assets.
        logger [logging.Logger]: Logger instance.

    @output:
        dict[str, dict[str, int]]: Filtered stats containing only local datasets.

    @scenario:
        HF/MS org may contain datasets not in local assets. Filter to only include
        datasets that exist in assets/dataset_info/.
    """
    filtered = {name: data for name, data in stats_dict.items() if name in existing_names}

    excluded_count = len(stats_dict) - len(filtered)
    if excluded_count > 0:
        logger.debug(
            "Excluded %d datasets not in local assets (total fetched: %d, kept: %d)",
            excluded_count,
            len(stats_dict),
            len(filtered),
        )

    return filtered


def generate_download_stats_json(
    dataset_info_dir: str,
    output_path: str,
    hf_org_name: str = "RoboCOIN",
    ms_org_name: str = "RoboCOIN",
    logger: logging.Logger | None = None,
) -> bool:
    """
    Generate download_stats.json by fetching statistics from HF/MS APIs.

    This is the main entry point for download statistics generation.

    DESIGN NOTE:
        This function depends on the git submodule: src/download_stat/DownloadAnalyzer
        It consumes info.yaml files from dataset_info_dir to determine which datasets
        exist locally for matching.

    @input:
        dataset_info_dir [str]: Path to assets/dataset_info/ containing YAML files.
        output_path [str]: Path to write download_stats.json.
        hf_org_name [str]: HuggingFace organization name (default: "RoboCOIN").
        ms_org_name [str]: ModelScope organization name (default: "RoboCOIN").
        logger [logging.Logger | None]: Logger instance. If None, creates default.

    @output:
        bool: True on success, False on failure (but never raises - logs errors instead).

    @scenario:
        Called by PageSync during the final aggregation step to generate download
        statistics for all datasets. Should not block the main PageSync workflow
        on failure - log warnings and continue.
    """
    _logger = logger or logging.getLogger(__name__)
    _logger.info("[DOWNLOAD_STAT] Starting download statistics collection...")
    _logger.info("  HF org: %s", hf_org_name)
    _logger.info("  MS org: %s", ms_org_name)
    _logger.info("  Dataset info dir: %s", dataset_info_dir)
    _logger.info("  Output path: %s", output_path)

    # Prepare output structure
    output_data: dict[str, Any] = {
        "last_updated": datetime.now().isoformat(),
        "huggingface": {
            "org_name": hf_org_name,
            "total_downloads": 0,
            "total_likes": 0,
            "dataset_count": 0,
            "datasets": {},
        },
        "modelscope": {
            "org_name": ms_org_name,
            "total_downloads": 0,
            "dataset_count": 0,
            "datasets": {},
        },
        "combined_total_downloads": 0,
    }

    try:
        # Step 1: Get existing dataset names from info.yaml files
        existing_dataset_names = _get_existing_dataset_names(dataset_info_dir, _logger)
        if not existing_dataset_names:
            _logger.warning("No existing dataset names found. Generating empty download stats.")

        # Step 2: Create temp directory for DownloadAnalyzer output
        temp_output_dir = Path(output_path).parent / ".temp_download_stat"
        temp_output_dir.mkdir(parents=True, exist_ok=True)

        # Step 3: Fetch download stats from submodule
        try:
            raw_result = _fetch_download_stats_from_submodule(
                hf_org_name=hf_org_name,
                ms_org_name=ms_org_name,
                temp_output_dir=str(temp_output_dir),
                logger=_logger,
            )
        except Exception as e:
            _logger.error("Failed to fetch download stats from submodule: %s", e, exc_info=True)
            _logger.warning("Generating empty download_stats.json due to fetch failure")
            # Write empty stats and return
            _write_download_stats_json(output_path, output_data, _logger)
            return False

        # Step 4: Parse HF CSV
        hf_csv_path = raw_result.get("hf_csv", "")
        if hf_csv_path and os.path.exists(hf_csv_path):
            hf_stats = _read_csv_to_dict(hf_csv_path, _logger)
            hf_filtered = _filter_by_existing_datasets(hf_stats, existing_dataset_names, _logger)

            total_hf_downloads = sum(d["downloads"] for d in hf_filtered.values())
            total_hf_likes = sum(d["likes"] for d in hf_filtered.values())

            output_data["huggingface"]["total_downloads"] = total_hf_downloads
            output_data["huggingface"]["total_likes"] = total_hf_likes
            output_data["huggingface"]["dataset_count"] = len(hf_filtered)
            output_data["huggingface"]["datasets"] = hf_filtered

            _logger.info(
                "HF stats: %d datasets, %d total downloads, %d total likes",
                len(hf_filtered),
                total_hf_downloads,
                total_hf_likes,
            )
        else:
            _logger.warning("HF CSV not found or empty: %s", hf_csv_path)

        # Step 5: Parse MS CSV
        ms_csv_path = raw_result.get("ms_csv", "")
        if ms_csv_path and os.path.exists(ms_csv_path):
            ms_stats = _read_csv_to_dict(ms_csv_path, _logger)
            ms_filtered = _filter_by_existing_datasets(ms_stats, existing_dataset_names, _logger)

            total_ms_downloads = sum(d["downloads"] for d in ms_filtered.values())

            output_data["modelscope"]["total_downloads"] = total_ms_downloads
            output_data["modelscope"]["dataset_count"] = len(ms_filtered)
            output_data["modelscope"]["datasets"] = ms_filtered

            _logger.info(
                "MS stats: %d datasets, %d total downloads", len(ms_filtered), total_ms_downloads
            )
        else:
            _logger.warning("MS CSV not found or empty: %s", ms_csv_path)

        # Step 6: Compute combined total
        output_data["combined_total_downloads"] = (
            output_data["huggingface"]["total_downloads"]
            + output_data["modelscope"]["total_downloads"]
        )

        _logger.info("Combined total downloads: %d", output_data["combined_total_downloads"])

        # Step 7: Write output JSON
        _write_download_stats_json(output_path, output_data, _logger)

        # Step 8: Cleanup temp directory
        try:
            import shutil

            shutil.rmtree(temp_output_dir)
            _logger.debug("Cleaned up temp directory: %s", temp_output_dir)
        except Exception as e:
            _logger.warning("Failed to cleanup temp directory %s: %s", temp_output_dir, e)

        _logger.info("[DOWNLOAD_STAT] Successfully generated download_stats.json")
        return True

    except Exception as e:
        _logger.error("Unexpected error generating download stats: %s", e, exc_info=True)
        _logger.warning("Writing empty download_stats.json due to error")
        try:
            _write_download_stats_json(output_path, output_data, _logger)
        except Exception:
            _logger.error("Failed to write fallback download_stats.json")
        return False


def _write_download_stats_json(
    output_path: str,
    data: dict[str, Any],
    logger: logging.Logger,
) -> None:
    """
    Write download stats dictionary to JSON file.

    @input:
        output_path [str]: Path to write JSON file.
        data [dict]: Download stats data structure.
        logger [logging.Logger]: Logger instance.

    @output:
        None. Writes file to output_path.
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info("Written download_stats.json to %s", output_file)
