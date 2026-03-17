"""
High-level helpers for publishing the generated `assets` folder to HuggingFace.

Design overview
---------------
1. UploadConfig collects every input required for a single upload run.
2. Validation helpers (_resolve_assets_dir / _resolve_token) guarantee all
   prerequisites are satisfied before any network calls happen.
3. upload_assets orchestrates the HuggingFace API call itself.
4. A tiny CLI (main) wires everything together for ad-hoc execution.

The default target repository is:
    RogersPyke/robocoin_datamanager_assets
which is expected to host the dataset assets for the RoboCOIN page project.
"""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from huggingface_hub import HfApi

HF_TOKEN_ENV_VAR = "HF_TOKEN"
DEFAULT_REPO_ID = "RogersPyke/robocoin_datamanager_assets"
DEFAULT_COMMIT_MESSAGE = "Update RoboCOIN assets"

logger = logging.getLogger(__name__)


class UploadAssetsError(RuntimeError):
    """Raised when the assets upload flow cannot proceed."""


@dataclass(frozen=True)
class UploadConfig:
    """Pure data container for one upload invocation."""

    repo_id: str = DEFAULT_REPO_ID
    repo_type: str = "dataset"
    revision: str = "main"
    commit_message: str = DEFAULT_COMMIT_MESSAGE
    assets_dir: Path = field(default_factory=lambda: Path.cwd() / "assets")
    token: str | None = None
    allow_patterns: tuple[str, ...] | None = None
    ignore_patterns: tuple[str, ...] | None = None


def _normalize_patterns(patterns: Sequence[str] | None) -> tuple[str, ...] | None:
    """Convert incoming pattern sequences to tuples the HF SDK expects."""

    if not patterns:
        return None
    return tuple(patterns)


def _resolve_assets_dir(raw_dir: Path) -> Path:
    """Ensure the assets directory exists and contains at least one artifact."""

    assets_dir = raw_dir.expanduser().resolve()
    if not assets_dir.exists():
        raise UploadAssetsError(f"Assets directory does not exist: {assets_dir}")
    if not assets_dir.is_dir():
        raise UploadAssetsError(f"Assets path is not a directory: {assets_dir}")
    if not any(assets_dir.iterdir()):
        raise UploadAssetsError(f"Assets directory is empty: {assets_dir}")
    return assets_dir


def _resolve_token(token: str | None) -> str:
    """
    Retrieve the HuggingFace token either from the provided value or the
    standard HF_TOKEN environment variable.
    """

    resolved = token or os.environ.get(HF_TOKEN_ENV_VAR)
    if not resolved:
        raise UploadAssetsError(
            "Missing HuggingFace token. Pass --hf-token or set HF_TOKEN env var."
        )
    return resolved


def upload_assets(config: UploadConfig) -> str:
    """
    Upload the prepared assets directory to HuggingFace Hub.

    Returns:
        str: The commit URL using the Hub's canonical repo_id (matches the repo name on the website).
    """

    assets_dir = _resolve_assets_dir(config.assets_dir)
    token = _resolve_token(config.token)

    api = HfApi(token=token)

    # Create the repo if it does not exist (no-op when exist_ok=True and repo exists).
    api.create_repo(
        repo_id=config.repo_id,
        repo_type=config.repo_type,
        exist_ok=True,
    )

    logger.info(
        "Uploading assets from %s to %s (repo_type=%s, revision=%s)",
        assets_dir,
        config.repo_id,
        config.repo_type,
        config.revision,
    )

    commit_sha = api.upload_folder(
        folder_path=str(assets_dir),
        repo_id=config.repo_id,
        repo_type=config.repo_type,
        revision=config.revision,
        commit_message=config.commit_message,
        allow_patterns=config.allow_patterns,
        ignore_patterns=config.ignore_patterns,
    )

    # Build commit URL using the Hub's canonical repo_id (matches the repo name shown on the website).
    raw_commit = commit_sha
    if "/commit/" in str(raw_commit):
        commit_sha = str(raw_commit).rstrip("/").split("/commit/")[-1]
    try:
        info = api.repo_info(repo_id=config.repo_id, repo_type=config.repo_type)
        canonical_id = info.id
    except Exception:
        canonical_id = config.repo_id
    path_prefix = {"dataset": "datasets", "model": "models", "space": "spaces"}.get(
        config.repo_type, "datasets"
    )
    commit_url = f"https://huggingface.co/{path_prefix}/{canonical_id}/commit/{commit_sha}"
    logger.info("Upload completed successfully: %s", commit_url)
    return commit_url


def sync_assets_to_hf(
    assets_dir: str | Path,
    *,
    repo_id: str = DEFAULT_REPO_ID,
    repo_type: str = "dataset",
    revision: str = "main",
    commit_message: str = DEFAULT_COMMIT_MESSAGE,
    token: str | None = None,
    allow_patterns: Sequence[str] | None = None,
    ignore_patterns: Sequence[str] | None = None,
) -> str:
    """
    Public helper with a minimal surface for orchestrators to call.

    Args mirror UploadConfig, but strings/Sequences are accepted for convenience.
    """

    config = UploadConfig(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        commit_message=commit_message,
        assets_dir=Path(assets_dir),
        token=token,
        allow_patterns=_normalize_patterns(allow_patterns),
        ignore_patterns=_normalize_patterns(ignore_patterns),
    )
    return upload_assets(config)


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser. Kept separate for easier testing."""

    parser = argparse.ArgumentParser(
        description=(
            "Upload the generated assets directory to "
            "RogersPyke/robocoin_datamanager_assets on HuggingFace."
        )
    )
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=None,
        help="Path to the local assets directory (defaults to ./assets).",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help="Target HuggingFace repo id.",
    )
    parser.add_argument(
        "--repo-type",
        default="dataset",
        choices=("dataset", "model", "space"),
        help="Type of repository on HuggingFace Hub.",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Branch or tag to push to.",
    )
    parser.add_argument(
        "--commit-message",
        default=DEFAULT_COMMIT_MESSAGE,
        help="Commit message recorded on the Hub.",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Explicit HuggingFace token. Falls back to HF_TOKEN env var.",
    )
    parser.add_argument(
        "--allow-pattern",
        action="append",
        dest="allow_patterns",
        default=None,
        help="Optional glob pattern to whitelist (can be repeated).",
    )
    parser.add_argument(
        "--ignore-pattern",
        action="append",
        dest="ignore_patterns",
        default=None,
        help="Optional glob pattern to ignore (can be repeated).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging verbosity (DEBUG, INFO, WARNING, ...).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> str:
    """
    CLI entrypoint for manual execution.

    Returns:
        str: Commit SHA returned by the upload call (mirrors upload_assets).
    """

    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    assets_dir = args.assets_dir or Path.cwd() / "assets"

    config = UploadConfig(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        revision=args.revision,
        commit_message=args.commit_message,
        assets_dir=assets_dir,
        token=args.hf_token,
        allow_patterns=_normalize_patterns(args.allow_patterns),
        ignore_patterns=_normalize_patterns(args.ignore_patterns),
    )

    try:
        return upload_assets(config)
    except UploadAssetsError as exc:
        logger.error("Assets upload failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
