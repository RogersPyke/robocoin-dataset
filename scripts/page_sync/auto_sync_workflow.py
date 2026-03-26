#!/usr/bin/env python3
"""
==================== WARNING (READ FIRST) ====================
This script only orchestrates the global workflow (run page sync + upload per cycle).
The database fields `dataset_info_sync_status/dataset_info_sync_err_msg` with `FAILED/err_msg`
are only written reliably inside each per-dataset loop `try/except`.

As a result, if a global post-processing step in `page_sync` fails
(for example, missing resources during consolidated metadata generation,
or non-fatal exceptions logged without entering dataset-level failure branches),
you may see:
- datasets still marked as `COMPLETED`
- no matching `FAILED` or full stack trace in DB
- HF upload still attempted afterward

Also, if HF upload fails, the current workflow only logs the exception and
does not automatically write failure state back to dataset-level DB status fields.
=============================================================

This script is an automated workflow for page sync and HF asset upload.

Recommended command examples (with `tmux`):
cd /home/rogerspyke/projects/robocoin-dataset

# One-time debug run
python scripts/page_sync/auto_sync_workflow.py \
  --db-cfg-path /mnt/db/postgresql_config.yaml \
  --target-dir /home/rogerspyke/projects \
  --log-level INFO \
  --update-videos \
  --crf 30 \
  --run-once

# Optional: pass HF token via --token/--hf-token (or set HF_TOKEN env var)
  --token <your_hf_token>

# Long-running loop in tmux (recommended on servers)
tmux new-session -d -s page-sync "\
cd /home/rogerspyke/projects/robocoin-dataset && \
python scripts/page_sync/auto_sync_workflow.py \
  --db-cfg-path db/postgresql_config.yaml \
  --target-dir ~/projects/robocoin_datamanager_assets \
  --log-level INFO \
  --update-videos \
  --crf 30"
tmux attach -t page-sync
# Detach with Ctrl+b, then d
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)


@dataclass
class SyncConfig:
    db_path: Path
    target_dir: Path
    crf: int
    update_videos: bool
    log_level: str
    interval_hours: float
    run_once: bool
    hf_token: str | None
    hf_repo_id: str
    hf_upload_enabled: bool
    hf_max_retries: int
    hf_retry_delay: float


def _run_subprocess(
    cmd: list[str] | str,
    *,
    cwd: Path | None = None,
    check: bool = True,
    shell: bool | None = None,
) -> subprocess.CompletedProcess:
    """
    Helper to run a subprocess with logging.

    Args:
        cmd: Command to run (list or string when shell=True)
        cwd: Working directory
        check: Whether to raise on non-zero exit
        shell: Whether to run through shell (defaults to False when cmd is list)
    """
    if isinstance(cmd, list):
        cmd_display = " ".join(cmd)
    else:
        cmd_display = cmd

    logger.debug("Running command: %s (cwd=%s)", cmd_display, cwd or ".")

    if shell is None:
        shell = isinstance(cmd, str)

    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        check=False,  # we'll handle check manually
        shell=shell,
        capture_output=True,
        text=True,
    )

    if result.stdout:
        logger.debug("Command stdout:\n%s", result.stdout)
    if result.stderr:
        logger.debug("Command stderr:\n%s", result.stderr)

    if check and result.returncode != 0:
        logger.error("Command failed with exit code %s: %s", result.returncode, cmd_display)
        raise subprocess.CalledProcessError(
            result.returncode,
            cmd_display,
            output=result.stdout,
            stderr=result.stderr,
        )

    return result


def _run_page_sync(config: SyncConfig) -> None:
    """Run page sync logic and generate/update required resources."""
    logger.info("Starting page sync...")
    logger.info("  Database: %s", config.db_path)
    logger.info("  Target dir: %s", config.target_dir)
    logger.info("  CRF: %s", config.crf)
    logger.info("  Update videos: %s", config.update_videos)

    # Reuse the same entrypoint as prepare_page_sync_files.py
    from robocoin_dataset.page_sync.page_sync import main as page_sync_main

    page_sync_main(
        db_path=str(config.db_path),
        target_dir=str(config.target_dir),
        crf=config.crf,
        update_videos=config.update_videos,
        log_level=config.log_level,
    )

    logger.info("Page sync completed successfully.")


def _resolve_page_assets_dir(target_dir: Path) -> Path:
    """
    Resolve generated assets folder.
    """
    return target_dir / "assets"


def _prefetch_hf_dataset_info(config: SyncConfig) -> None:
    """
    Pull existing remote dataset_info YAML files before page sync.

    This keeps local consolidation input complete (historical + newly generated).
    """
    assets_dir = _resolve_page_assets_dir(config.target_dir)
    assets_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Pulling existing dataset_info files from HuggingFace before sync...")
    logger.info("  Repo id: %s", config.hf_repo_id)
    logger.info("  Local assets dir: %s", assets_dir)

    snapshot_download(
        repo_id=config.hf_repo_id,
        repo_type="dataset",
        revision="main",
        local_dir=str(assets_dir),
        allow_patterns=["dataset_info/*.yaml", "dataset_info/*.yml"],
        token=config.hf_token or os.environ.get("HF_TOKEN"),
    )
    logger.info("Finished pulling dataset_info files from HuggingFace.")


def _run_hf_upload(config: SyncConfig) -> None:
    """
    Upload generated assets to HuggingFace.

    Token is expected from `--hf-token/--token` or `HF_TOKEN` env var (handled by the upload script).
    """
    if not config.hf_upload_enabled:
        logger.warning(
            "Skipping HuggingFace upload because token is not set. "
            "Set HF_TOKEN env var or pass --token/--hf-token."
        )
        return

    assets_dir = _resolve_page_assets_dir(config.target_dir)
    repo_root = Path(__file__).resolve().parents[2]

    cmd: list[str] = [
        sys.executable,
        "scripts/page_sync/upload_assets.py",
        "--assets-dir",
        str(assets_dir),
        "--repo-id",
        config.hf_repo_id,
        "--max-retries",
        str(config.hf_max_retries),
        "--retry-delay",
        str(config.hf_retry_delay),
        "--log-level",
        config.log_level,
    ]
    if config.hf_token:
        # If token exists only in env var, upload_assets.py will fall back automatically.
        cmd += ["--hf-token", config.hf_token]

    logger.info("Starting HuggingFace upload...")
    logger.info("  Assets dir: %s", assets_dir)
    logger.info("  Repo id: %s", config.hf_repo_id)
    _run_subprocess(cmd, cwd=repo_root, check=True)
    logger.info("HuggingFace upload finished.")


def _run_mount(config: SyncConfig) -> None:
    """Mount a path if needed."""
    if not config.mount_path:
        logger.debug("No mount path specified, skipping mount step.")
        return

    logger.info("Mounting path: %s", config.mount_path)

    try:
        _run_subprocess(["mount", str(config.mount_path)], check=True)
        logger.info("Mount completed: %s", config.mount_path)
    except subprocess.CalledProcessError as exc:
        logger.error(
            "Mount command failed for %s (exit=%s). Please check fstab/permissions.",
            config.mount_path,
            exc.returncode,
        )
        # Do not block later git steps on mount failure; only log it.


def _copy_assets_to_git_dir(config: SyncConfig) -> None:
    """Copy assets folder to docs/assets in git directory (force overwrite)."""
    source_assets = config.target_dir / "docs" / "assets"
    target_assets = config.git_dir / "docs" / "assets"

    logger.info("Source assets path: %s", source_assets)
    logger.info("Target assets path: %s", target_assets)
    logger.info("Source exists: %s", source_assets.exists())
    logger.info("Target exists: %s", target_assets.exists())

    if not source_assets.exists():
        logger.warning("Source assets directory does not exist: %s", source_assets)
        # Check whether assets already exist in target.
        if target_assets.exists():
            logger.info("Assets already exist in target directory, skipping copy")
            return
        raise FileNotFoundError(f"Source assets directory not found: {source_assets}")

    logger.info("Copying assets from %s to %s (force overwrite)", source_assets, target_assets)

    try:
        # Skip copy when source and target are identical.
        if source_assets.resolve() == target_assets.resolve():
            logger.info("Source and target are the same path, skipping copy")
            return

        # Ensure target parent directory exists.
        target_assets.parent.mkdir(parents=True, exist_ok=True)

        # Force overwrite: remove target first when it already exists.
        if target_assets.exists():
            import shutil
            logger.info("Removing existing target directory: %s", target_assets)
            shutil.rmtree(target_assets)

        # Prefer rsync; fall back to shutil when rsync is unavailable.
        try:
            logger.info("Using rsync to copy assets")
            _run_subprocess(["rsync", "-av", "--delete", str(source_assets) + "/", str(target_assets)], check=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning("rsync failed (%s), falling back to shutil", e)
            # rsync unavailable, use shutil.
            import shutil
            shutil.copytree(source_assets, target_assets)

        logger.info("Assets copy completed successfully to %s", target_assets)

        # Verify copy result.
        if not target_assets.exists():
            raise RuntimeError(f"Copy completed but target directory does not exist: {target_assets}")

    except Exception as exc:
        logger.error("Failed to copy assets to git directory: %s", exc)
        raise


def _count_datasets(db_path: Path) -> int:
    """Count datasets marked COMPLETED in database.

    This represents total datasets whose page sync is marked completed.
    """
    try:
        from robocoin_dataset.database.database import DatasetDatabase
        from robocoin_dataset.database.models import DatasetDB, TaskStatus

        db = DatasetDatabase(str(db_path))
        with db.with_session() as session:
            completed_count = session.query(DatasetDB).filter(
                DatasetDB.dataset_info_sync_status == TaskStatus.COMPLETED
            ).count()

        logger.info("Database shows %d datasets marked as COMPLETED", completed_count)
        return completed_count
    except Exception as e:
        logger.warning("Failed to query database for completed datasets: %s", e)
        return 0


def _setup_git_auth(config: SyncConfig) -> None:
    """Set git auth details to avoid interactive prompts."""
    if not config.git_username or not config.git_token:
        logger.debug("No git credentials provided, using default authentication (SSH or stored credentials)")
        return

    # Set git credential helper environment values.
    import os
    os.environ['GIT_USERNAME'] = config.git_username
    os.environ['GIT_TOKEN'] = config.git_token

    # Create a simple credential helper script.
    credential_script = """#!/bin/bash
echo "username=$GIT_USERNAME"
echo "password=$GIT_TOKEN"
"""
    credential_path = Path.home() / ".git_credential_helper.sh"
    credential_path.write_text(credential_script)
    credential_path.chmod(0o755)

    logger.info("Git credentials configured for user: %s", config.git_username)


def _run_git_sync(config: SyncConfig) -> None:
    """Run git add/commit/push in target dir to trigger GitHub Actions.

    Only add files under assets to avoid unrelated modifications.
    """
    target_dir = config.target_dir

    logger.info("Running git sync in %s", target_dir)

    # 0) Configure authentication when credentials are provided.
    _setup_git_auth(config)

    # 1) Add only docs/assets to avoid touching unrelated files.
    assets_path = target_dir / "docs" / "assets"
    if not assets_path.exists():
        logger.warning("Assets directory does not exist at %s. Skipping git add.", assets_path)
        return

    logger.info("Adding docs/assets directory to git")
    # Use relative path from target_dir.
    _run_subprocess(["git", "add", "docs/assets/"], cwd=target_dir, check=True)

    # 2) Skip commit/push when there are no staged changes.
    diff_result = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=str(target_dir),
    )

    if diff_result.returncode == 0:
        logger.info("No changes to commit. Skipping git commit and push.")
        return
    if diff_result.returncode not in (0, 1):
        logger.warning(
            "Unexpected return code from 'git diff --cached --quiet': %s. "
            "Will still attempt to commit/push.",
            diff_result.returncode,
        )

    # 3) Compute dataset count and update commit message.
    dataset_count = _count_datasets(config.db_path)
    commit_message = f"{config.git_commit_message} ({dataset_count} datasets)"

    # 4) git commit
    _run_subprocess(
        ["git", "commit", "-m", commit_message],
        cwd=target_dir,
        check=True,
    )
    logger.info("Git commit created with message: %s", commit_message)

    # 5) git push
    push_cmd = ["git", "push", config.git_remote, config.git_branch]

    # Use credential helper when credentials are provided.
    if config.git_username and config.git_token:
        env = os.environ.copy()
        env['GIT_ASKPASS'] = str(Path.home() / ".git_credential_helper.sh")
        result = subprocess.run(
            push_cmd,
            cwd=str(target_dir),
            env=env,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.error("Git push failed: %s", result.stderr)
            raise subprocess.CalledProcessError(result.returncode, push_cmd, result.stdout, result.stderr)
    else:
        _run_subprocess(push_cmd, cwd=target_dir, check=True)

    logger.info(
        "Git push completed to %s/%s. GitHub Actions (if configured) should be triggered.",
        config.git_remote,
        config.git_branch,
    )


def _run_git_sync_in_git_dir(config: SyncConfig) -> None:
    """Run git add/commit/push in git directory to trigger GitHub Actions.

    Only add files under docs/assets to avoid unrelated modifications.
    """
    git_dir = config.git_dir

    logger.info("Running git sync in git directory: %s", git_dir)

    # 0) Configure authentication when credentials are provided.
    _setup_git_auth(config)

    # 1) Add only docs/assets to avoid touching unrelated files.
    assets_path = git_dir / "docs" / "assets"
    if not assets_path.exists():
        logger.warning("Assets directory does not exist in git dir at %s. Skipping git add.", assets_path)
        return

    logger.info("Adding docs/assets directory to git")
    # Use relative path from git_dir.
    _run_subprocess(["git", "add", "docs/assets/"], cwd=git_dir, check=True)

    # 2) Skip commit/push when there are no staged changes.
    diff_result = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=str(git_dir),
    )

    if diff_result.returncode == 0:
        logger.info("No changes to commit in git directory. Skipping git commit and push.")
        return
    if diff_result.returncode not in (0, 1):
        logger.warning(
            "Unexpected return code from 'git diff --cached --quiet': %s. "
            "Will still attempt to commit/push.",
            diff_result.returncode,
        )

    # 3) Compute dataset count and update commit message.
    dataset_count = _count_datasets(config.db_path)
    commit_message = f"{config.git_commit_message} ({dataset_count} datasets)"

    # 4) git commit
    _run_subprocess(
        ["git", "commit", "-m", commit_message],
        cwd=git_dir,
        check=True,
    )
    logger.info("Git commit created with message: %s", commit_message)

    # 5) git push
    push_cmd = ["git", "push", config.git_remote, config.git_branch]

    # Use credential helper when credentials are provided.
    if config.git_username and config.git_token:
        env = os.environ.copy()
        env['GIT_ASKPASS'] = str(Path.home() / ".git_credential_helper.sh")
        result = subprocess.run(
            push_cmd,
            cwd=str(git_dir),
            env=env,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.error("Git push failed: %s", result.stderr)
            raise subprocess.CalledProcessError(result.returncode, push_cmd, result.stdout, result.stderr)
    else:
        _run_subprocess(push_cmd, cwd=git_dir, check=True)

    logger.info(
        "Git push completed to %s/%s. GitHub Actions (if configured) should be triggered.",
        config.git_remote,
        config.git_branch,
    )


def _run_single_cycle(config: SyncConfig) -> None:
    """Run one full sync cycle plus HuggingFace upload."""
    start_time = datetime.now()
    logger.info("===== Starting auto sync cycle at %s =====", start_time.isoformat(timespec="seconds"))

    try:
        _prefetch_hf_dataset_info(config)
    except Exception:  # noqa: BLE001
        logger.exception("HuggingFace prefetch step failed.")
        # Prefetch failure should not block local generation.

    try:
        _run_page_sync(config)
    except Exception:  # noqa: BLE001
        logger.exception("Page sync step failed.")
        # Keep going so later steps can still run if needed.

    try:
        _run_hf_upload(config)
    except Exception:  # noqa: BLE001
        logger.exception("HuggingFace upload step failed.")
        # Upload failure should not block later steps.

    end_time = datetime.now()
    logger.info(
        "===== Auto sync cycle finished at %s (duration: %s) =====",
        end_time.isoformat(timespec="seconds"),
        end_time - start_time,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto workflow for generating page assets and uploading them to HuggingFace.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run continuously every 2 hours (assets generation + incremental HF upload)
  python scripts/page_sync/auto_sync_workflow.py \\
    --db-cfg-path /mnt/db/postgresql_config.yaml \\
    --target-dir /home/rogerspyke/projects \\
    --log-level INFO \\
    --update-videos \\
    --crf 30
    --token <your_hf_token>

  # Run once for debugging
  python scripts/page_sync/auto_sync_workflow.py \\
    --db-cfg-path /mnt/db/postgresql_config.yaml \\
    --target-dir /home/rogerspyke/projects \\
    --run-once

  # Recommended: run loop mode inside tmux
  tmux new-session -d -s page-sync "cd /home/rogerspyke/projects/robocoin-dataset && \\
    python scripts/page_sync/auto_sync_workflow.py --db-cfg-path /mnt/db/postgresql_config.yaml --target-dir /home/rogerspyke/projects --update-videos --crf 30"
  tmux attach -t page-sync
        """,
    )

    parser.add_argument(
        "--db-cfg-path",
        type=str,
        required=True,
        help="Path to the PostgreSQL YAML config file (e.g., /mnt/db/postgresql_config.yaml)",
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        required=True,
        help="Root directory where the page project will generate `assets/`.",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=23,
        help="CRF value for video compression (default: 23, range: 0-51, lower = better quality)",
    )
    parser.add_argument(
        "--update-videos",
        action="store_true",
        help="Force regenerate videos and thumbnails even if they exist (default: False)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--interval-hours",
        type=float,
        default=2.0,
        help="Interval between sync cycles in hours (default: 2.0)",
    )

    parser.add_argument(
        "--hf-repo-id",
        type=str,
        default="RogersPyke/robocoin_datamanager_assets",
        help="Target HuggingFace repo id for uploading generated page assets.",
    )
    parser.add_argument(
        "--hf-token",
        "--token",
        dest="hf_token",
        type=str,
        default=None,
        help="HuggingFace token for uploading assets (optional; falls back to HF_TOKEN env var).",
    )
    parser.add_argument(
        "--hf-max-retries",
        type=int,
        default=3,
        help="Max HF upload attempts on failure (default: 3).",
    )
    parser.add_argument(
        "--hf-retry-delay",
        type=float,
        default=10.0,
        help="Seconds to wait between HF upload retries (default: 10.0).",
    )
    parser.add_argument(
        "--run-once",
        action="store_true",
        help="Run a single sync cycle and exit (useful for debugging).",
    )

    return parser.parse_args(argv)


def _build_config(args: argparse.Namespace) -> SyncConfig:
    db_cfg_path_str = args.db_cfg_path

    db_path = Path(db_cfg_path_str).expanduser().absolute()
    target_dir = Path(args.target_dir).expanduser().absolute()

    # Basic path validation
    if not db_path.exists():
        print(f"Error: DB config file not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    if not target_dir.exists():
        print(f"Error: Target directory not found: {target_dir}", file=sys.stderr)
        print("Please create the directory first or check the path.", file=sys.stderr)
        sys.exit(1)

    if not target_dir.is_dir():
        print(f"Error: Target path is not a directory: {target_dir}", file=sys.stderr)
        sys.exit(1)

    if args.interval_hours <= 0:
        print("Error: --interval-hours must be positive.", file=sys.stderr)
        sys.exit(1)

    resolved_hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    hf_upload_enabled = bool(resolved_hf_token)

    return SyncConfig(
        db_path=db_path,
        target_dir=target_dir,
        crf=args.crf,
        update_videos=args.update_videos,
        log_level=args.log_level,
        interval_hours=args.interval_hours,
        run_once=args.run_once,
        hf_token=args.hf_token,
        hf_repo_id=args.hf_repo_id,
        hf_upload_enabled=hf_upload_enabled,
        hf_max_retries=args.hf_max_retries,
        hf_retry_delay=args.hf_retry_delay,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # Setup logging
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = _build_config(args)

    logger.info("Auto sync workflow starting with configuration:")
    logger.info("  db_path: %s", config.db_path)
    logger.info("  target_dir: %s", config.target_dir)
    logger.info("  crf: %s", config.crf)
    logger.info("  update_videos: %s", config.update_videos)
    logger.info("  interval_hours: %s", config.interval_hours)
    logger.info("  hf_repo_id: %s", config.hf_repo_id)
    logger.info("  hf_upload_enabled: %s", config.hf_upload_enabled)

    if config.run_once:
        logger.info("Running in single-cycle mode (--run-once).")
        _run_single_cycle(config)
        return 0

    logger.info("Entering loop mode. Sync will run every %.2f hours.", config.interval_hours)

    interval_seconds = config.interval_hours * 3600

    try:
        while True:
            _run_single_cycle(config)
            logger.info("Sleeping for %.2f hours before next cycle.", config.interval_hours)
            time.sleep(interval_seconds)
    except KeyboardInterrupt:
        logger.info("Received KeyboardInterrupt, exiting auto sync workflow.")
        return 0
    except Exception:  # noqa: BLE001
        logger.exception("Unexpected fatal error in auto sync workflow.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
