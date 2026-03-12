#!/usr/bin/env python3
"""
Copy all RoboCOIN datasets (except excluded) into a single repo as first-level subfolders.

Purpose:
    Copy every dataset from https://huggingface.co/RoboCOIN into
    https://huggingface.co/RogersPyke/RoboCOIN_useable, each source repo as one
    top-level subfolder. Uses Hugging Face cache to avoid re-downloading when possible.

Dependencies:
    - huggingface_hub (HfApi, snapshot_download)
    - robocoin_dataset.utils.log_config (get_utc8_timestamp)
    - robocoin_dataset.utils.logger (setup_logger_utc8)

Usage:
    # Use HF_TOKEN env or will prompt for token when needed
    python scripts/hub_upload/tools/mv_usable.py

    # Optional: custom cache dir (default uses HF_HOME / ~/.cache/huggingface)
    python scripts/hub_upload/tools/mv_usable.py

Output:
    - Log file: logs/hub_upload/mv_usable/mv_usable_<YYYYMMDDHHMMSS>.log (UTC+8)
    - Target repo: RogersPyke/RoboCOIN_useable with one subfolder per source dataset
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import time
from pathlib import Path

# Add project root for imports (this file is in scripts/hub_upload/tools/)
_project_root = Path(__file__).resolve().parents[3]
if str(_project_root / "src") not in sys.path:
    sys.path.insert(0, str(_project_root / "src"))

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import HfHubHTTPError

from robocoin_dataset.utils.logger import setup_logger_utc8

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SOURCE_ORG = "RoboCOIN"
TARGET_NAMESPACE = "RogersPyke"
TARGET_REPO_NAME = "RoboCOIN_useable"
TARGET_REPO_ID = f"{TARGET_NAMESPACE}/{TARGET_REPO_NAME}"

# Repos to exclude from copy (exact match after normalizing to lowercase).
# Each entry is the dataset name only (no "RoboCOIN/" prefix).
EXCLUDE_REPO_NAMES = [
    "AIRBOT_MMK2_beauty_sponge_and_cake_to_place",
    "AIRBOT_MMK2_building_block_storage",
    "AgiBot-g1_battery_storage_c",
    "AgiBot-g1_box_storage_part_a",
    "AgiBot-g1_box_storage_tool",
    "AgiBot-g1_mobile_accessory_storage_box_d",
    "AgiBot-g1_picks_up_parts_b",
    "Cobot_Magic_classification_of_fruits_and_vegetables_a",
    "Cobot_Magic_desktop_organization",
    "Cobot_Magic_drawer_storage_mineral_water",
    "Cobot_Magic_food_packaging",
    "Cobot_Magic_make_hamburger",
    "Cobot_Magic_move_the_ball_and_the_cube_block",
    "Cobot_Magic_open_the_shoebox",
    "Cobot_Magic_pour_water_a",
    "Cobot_Magic_vase_storage_flower",
    "G1edu-u3_bowl_storage_grape_singletry",
    "G1edu-u3_food_storage",
    "G1edu-u3_plate_storage_doll",
    "G1edu-u3_pullBowl_storage_bread_b",
    "Galbot_g1_fold_clothe_b",
    "Galbot_g1_fold_clothe_c",
    "Galbot_g1_fold_clothe_e",
    "Galbot_g1_steamer_storage_baozi_a",
    "Galbot_g1_steamer_storage_baozi_b",
    "Galbot_g1_steamer_storage_baozi_c",
    "Galbot_g1_steamer_storage_baozi_d",
    "Galbot_g1_steamer_storage_baozi_e",
    "Galbot_g1_steamer_storage_baozi_f",
    "Galbot_g1_steamer_storage_baozi_g",
    "Galbot_g1_steamer_storage_baozi_h",
    "Galbot_g1_steamer_storage_baozi_i",
    "Galbot_g1_steamer_storage_baozi_j",
    "R1_Lite_cook_a_meal",
    "R1_Lite_drawer_storage_hair_dryer",
    "R1_Lite_move_the_position_of_the_coffee_capsule",
    "R1_Lite_move_the_position_of_the_soda",
    "R1_Lite_open_and_close_microwave_oven",
    "R1_Lite_pick_up_and_store_items",
    "R1_Lite_put_slippers_into_floor_standing_shoe_cabinet",
    "R1_Lite_take_clothes_out_of_the_washing_machine",
    "R1_Lite_throw_out_the_trash",
    "RMC-AIDA-L_box_up_down",
    "RMC-AIDA-L_fold_shirt",
    "Split_aloha_fold_the_pants",
    "Split_aloha_pour_tea",
    "Tianqin_A2_place_the_paper_box",
    "leju_robot_box_storage_parcel_f",
    "leju_robot_hotel_services_ah",
    "leju_robot_moving_parts_o",
    "leju_robot_moving_parts_s",
    "leju_robot_moving_parts_t",
]

# Max attempts for upload (1 initial + 4 retries = 5 total)
MAX_UPLOAD_RETRIES = 5
# Seconds to wait between upload retries
UPLOAD_RETRY_DELAY_SEC = 10

# ANSI colors for log messages (PROMPTS.md: WARNING/ERR red, SUCCESS green, URL/args blue)
ANSI_RED = "\033[91m"
ANSI_GREEN = "\033[92m"
ANSI_BLUE = "\033[94m"
ANSI_RESET = "\033[0m"


def _colorize(text: str, color: str) -> str:
    """Apply ANSI color if terminal supports it. Input: text and color code. Output: colored or plain string."""
    if os.getenv("TERM") in (None, "dumb") or os.getenv("NO_COLOR"):
        return text
    return f"{color}{text}{ANSI_RESET}"


def _log_success(logger: logging.Logger, message: str) -> None:
    logger.info(_colorize(message, ANSI_GREEN))


def _log_error(logger: logging.Logger, message: str) -> None:
    logger.error(_colorize(message, ANSI_RED))


def _log_warning(logger: logging.Logger, message: str) -> None:
    logger.warning(_colorize(message, ANSI_RED))


def _log_url(logger: logging.Logger, message: str, level: int = logging.INFO) -> None:
    logger.log(level, _colorize(message, ANSI_BLUE))


def _normalize_name_for_exclude(name: str) -> str:
    """Normalize repo name for exclude list comparison (lowercase). Input: repo name. Output: normalized string."""
    return name.strip().lower()


def _build_exclude_set() -> set[str]:
    """Build set of normalized names to exclude. Input: none. Output: set of lowercase names."""
    return {_normalize_name_for_exclude(n) for n in EXCLUDE_REPO_NAMES}


def list_source_datasets(api: HfApi, logger: logging.Logger) -> list[str]:
    """
    List all dataset repo names under SOURCE_ORG on Hugging Face.

    Input:
        api: HfApi instance (with token if needed).
        logger: Logger for messages.

    Output:
        List of dataset names (e.g. ["Airbot_MMK2_storage_cup", ...]), without "Org/" prefix.
        Empty list on error.

    Usage:
        Used to get the full list of repos to consider for copy.
    """
    try:
        repos = api.list_datasets(author=SOURCE_ORG)
        names = []
        for repo in repos:
            rid = getattr(repo, "id", None) or str(repo)
            if "/" in rid:
                names.append(rid.split("/", 1)[1])
            else:
                names.append(rid)
        logger.info(f"[LIST] Found {len(names)} datasets under {SOURCE_ORG}")
        return names
    except Exception as e:
        _log_error(logger, f"[LIST] ERR listing datasets: {e}")
        return []


def filter_included(names: list[str], exclude_set: set[str], logger: logging.Logger) -> list[str]:
    """
    Remove excluded names (case-insensitive). Idempotent: same input gives same output.

    Input:
        names: List of repo names from the hub.
        exclude_set: Set of normalized (lowercase) names to exclude.
        logger: Logger.

    Output:
        List of names not in exclude_set.

    Usage:
        Apply exclude list before copying.
    """
    included = [n for n in names if _normalize_name_for_exclude(n) not in exclude_set]
    skipped = len(names) - len(included)
    if skipped:
        logger.info(f"[FILTER] Excluded {skipped} repos; {len(included)} to copy")
    return included


def ensure_target_repo(api: HfApi, token: str | None, logger: logging.Logger) -> bool:
    """
    Create target dataset repo if it does not exist.

    Input:
        api: HfApi instance.
        token: HF token (can be None to use env).
        logger: Logger.

    Output:
        True if repo exists or was created, False on failure.

    Usage:
        Call once before uploading any subfolder.
    """
    try:
        if api.repo_exists(repo_id=TARGET_REPO_ID, repo_type="dataset", token=token):
            _log_url(logger, f"[REPO] Target already exists: {TARGET_REPO_ID}")
            return True
        api.create_repo(
            repo_id=TARGET_REPO_ID,
            repo_type="dataset",
            token=token,
            exist_ok=True,
        )
        _log_success(logger, f"[REPO] SUCCESS created {TARGET_REPO_ID}")
        return True
    except Exception as e:
        _log_error(logger, f"[REPO] ERR creating target repo: {e}")
        return False


def download_to_cache(
    repo_id: str,
    token: str | None,
    cache_dir: str | Path | None,
    logger: logging.Logger,
) -> str | None:
    """
    Download repo to Hugging Face cache (no local_dir to avoid duplicate storage).
    Uses cache so repeated runs reuse files.

    Input:
        repo_id: Full repo id (e.g. "RoboCOIN/Airbot_MMK2_storage_cup").
        token: HF token.
        cache_dir: Optional cache directory; default uses HF_HOME.
        logger: Logger.

    Output:
        Path to snapshot directory in cache, or None on failure.

    Usage:
        Get local path for upload_folder without copying to a temp dir.
    """
    try:
        path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            cache_dir=cache_dir,
            local_dir=None,
        )
        return path
    except Exception as e:
        _log_error(logger, f"[DOWNLOAD] ERR {repo_id}: {e}")
        return None


def upload_subfolder(
    api: HfApi,
    folder_path: str | Path,
    path_in_repo: str,
    token: str | None,
    logger: logging.Logger,
) -> bool:
    """
    Upload a local folder to target repo as first-level subfolder.
    Retries up to MAX_UPLOAD_RETRIES times on failure (e.g. network errors).

    Input:
        api: HfApi instance.
        folder_path: Local path to the downloaded snapshot.
        path_in_repo: Subfolder name in repo (e.g. "Airbot_MMK2_storage_cup").
        token: HF token.
        logger: Logger.

    Output:
        True if upload succeeded, False otherwise.

    Usage:
        One call per source dataset.
    """
    last_err = None
    for attempt in range(1, MAX_UPLOAD_RETRIES + 1):
        try:
            api.upload_folder(
                folder_path=str(folder_path),
                path_in_repo=path_in_repo,
                repo_id=TARGET_REPO_ID,
                repo_type="dataset",
                token=token,
                commit_message=f"Copy {path_in_repo} into RoboCOIN_useable",
            )
            _log_success(logger, f"[UPLOAD] SUCCESS {path_in_repo}")
            return True
        except (HfHubHTTPError, OSError, ConnectionError, TimeoutError) as e:
            last_err = e
            if attempt < MAX_UPLOAD_RETRIES:
                _log_warning(
                    logger,
                    f"[UPLOAD] attempt {attempt}/{MAX_UPLOAD_RETRIES} failed for {path_in_repo}, "
                    f"retry in {UPLOAD_RETRY_DELAY_SEC}s: {e}",
                )
                time.sleep(UPLOAD_RETRY_DELAY_SEC)
            else:
                _log_error(logger, f"[UPLOAD] ERR {path_in_repo} after {MAX_UPLOAD_RETRIES} attempts: {e}")
                return False
        except Exception as e:
            _log_error(logger, f"[UPLOAD] ERR {path_in_repo}: {e}")
            return False
    if last_err is not None:
        _log_error(logger, f"[UPLOAD] ERR {path_in_repo}: {last_err}")
    return False


def delete_repo_cache(snapshot_path: str | Path, logger: logging.Logger) -> None:
    """
    Remove this repo's cache dir so disk is not kept after upload.
    snapshot_download stores under .../hub/datasets--Org--Name/snapshots/<rev>;
    we remove the repo-level dir (parent of snapshot's parent).

    Input:
        snapshot_path: Path returned by snapshot_download (the snapshot dir).
        logger: Logger.

    Output:
        None. Logs and removes dir on success.

    Usage:
        Call after upload (success or fail) for each downloaded repo.
    """
    path = Path(snapshot_path).resolve()
    if not path.exists():
        return
    # Snapshot path is .../datasets--Org--Name/snapshots/<rev>; repo cache dir is parent.parent
    repo_cache_dir = path.parent.parent
    if "snapshots" not in path.parts or not repo_cache_dir.exists():
        return
    try:
        shutil.rmtree(repo_cache_dir)
        logger.info(f"[CACHE] Deleted cache dir: {repo_cache_dir}")
    except Exception as e:
        _log_warning(logger, f"[CACHE] Failed to delete {repo_cache_dir}: {e}")


def run_copy(token: str | None, cache_dir: str | Path | None, logger: logging.Logger) -> None:
    """
    Main workflow: list -> filter -> ensure target -> for each: download to cache -> upload subfolder.

    Input:
        token: Hugging Face token (or None for env / default).
        cache_dir: Optional cache dir; None uses default HF cache.
        logger: Logger.

    Output:
        None. Logs and side effects only.

    Usage:
        Single entry for the copy process.
    """
    api = HfApi(token=token)
    exclude_set = _build_exclude_set()

    names = list_source_datasets(api, logger)
    if not names:
        _log_error(logger, "[RUN] No source datasets found, aborting")
        return

    included = filter_included(names, exclude_set, logger)
    if not included:
        logger.info("[RUN] No datasets to copy after exclude list")
        return

    if not ensure_target_repo(api, token, logger):
        _log_error(logger, "[RUN] Cannot ensure target repo, aborting")
        return

    ok = 0
    fail = 0
    for i, name in enumerate(included, 1):
        repo_id = f"{SOURCE_ORG}/{name}"
        _log_url(logger, f"[RUN] ({i}/{len(included)}) {repo_id}")
        local_path = download_to_cache(repo_id, token, cache_dir, logger)
        if local_path is None:
            fail += 1
            continue
        if upload_subfolder(api, local_path, name, token, logger):
            ok += 1
        else:
            fail += 1
        delete_repo_cache(local_path, logger)

    _log_success(logger, f"[RUN] SUCCESS finished: {ok} copied, {fail} failed")
    if fail:
        _log_warning(logger, f"[RUN] WARNING {fail} dataset(s) failed")


def setup_log_dir_and_logger() -> tuple[logging.Logger, Path]:
    """
    Create log dir logs/hub_upload/mv_usable and logger with file mv_usable_<YYYYMMDDHHMMSS>.log (UTC+8).

    Input: none.
    Output: (logger, log_dir Path).
    """
    log_dir = _project_root / "logs" / "hub_upload" / "mv_usable"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger_utc8(
        name="mv_usable",
        log_dir=log_dir,
        level=logging.INFO,
        console_output=True,
        colored_console=True,
    )
    return logger, log_dir


def main() -> int:
    """
    Entry point: resolve token, setup logger, run copy.

    Input: from environment (HF_TOKEN) or None.
    Output: 0 on success, 1 on failure.
    """
    logger, _ = setup_log_dir_and_logger()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        _log_warning(logger, "[MAIN] No HF_TOKEN / HUGGING_FACE_HUB_TOKEN; some ops may fail or prompt")

    cache_dir = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if cache_dir:
        _log_url(logger, f"[MAIN] Using cache dir: {cache_dir}")

    try:
        run_copy(token=token, cache_dir=cache_dir, logger=logger)
        return 0
    except Exception as e:
        _log_error(logger, f"[MAIN] ERR: {e}")
        return 1


def mv_usable() -> None:
    """Convenience entry: run main and exit with return code. Use from CLI or as function."""
    sys.exit(main())


if __name__ == "__main__":
    mv_usable()
