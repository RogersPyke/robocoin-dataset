#!/usr/bin/env python3
"""
Command-line entrypoint to publish the generated assets folder to HuggingFace.

This script is intentionally thin: it only
1. Consumes CLI arguments (repo id, token, commit message, filters, etc.)
2. Normalizes them while setting up logging
3. Delegates the real upload work to `upload_assets_utils.sync_assets_to_hf`

Usage example:
    python scripts/page_sync/upload_assets.py \
        --assets-dir /home/rogerspyke/projects/assets \
        --repo-id RogersPyke/robocoin_datamanager_assets \
        --hf-token hf_xxx
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from robocoin_dataset.page_sync._upload import (
    DEFAULT_COMMIT_MESSAGE,
    DEFAULT_REPO_ID,
    HF_TOKEN_ENV_VAR,
    sync_assets_to_hf,
)


def _build_parser() -> argparse.ArgumentParser:
    """CLI definition kept small so it is easy to reuse in tests."""

    parser = argparse.ArgumentParser(
        description=(
            "Upload the local assets directory to the HuggingFace repo "
            "RogersPyke/robocoin_datamanager_assets (or a custom target)."
        )
    )
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=None,
        help="Path to the local assets folder (defaults to ./assets).",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help="Fully qualified HuggingFace repo id.",
    )
    parser.add_argument(
        "--repo-type",
        choices=("dataset", "model", "space"),
        default="dataset",
        help="Repository kind on HuggingFace Hub (default: dataset).",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Branch or tag to push to.",
    )
    parser.add_argument(
        "--commit-message",
        default=DEFAULT_COMMIT_MESSAGE,
        help="Commit message recorded on the HuggingFace repo.",
    )
    parser.add_argument(
        "--hf-token",
        dest="hf_token",
        default=None,
        help=f"HuggingFace token (falls back to ${HF_TOKEN_ENV_VAR}).",
    )
    parser.add_argument(
        "--allow-pattern",
        action="append",
        dest="allow_patterns",
        default=None,
        help="Optional glob pattern to whitelist. Can be repeated.",
    )
    parser.add_argument(
        "--ignore-pattern",
        action="append",
        dest="ignore_patterns",
        default=None,
        help="Optional glob pattern to exclude. Can be repeated.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging verbosity (DEBUG, INFO, WARNING, ...).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Number of upload attempts on failure (default: 3).",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=10.0,
        help="Seconds to wait between retries (default: 10).",
    )
    return parser


def _configure_logging(level: str) -> None:
    """Small helper for consistent logging output."""

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )


def run(argv: list[str] | None = None) -> str:
    """
    Parse arguments, configure logging, and trigger the upload routine.

    Returns:
        str: The commit URL returned by the upload (canonical repo name, matches the Hub page).
    """

    parser = _build_parser()
    args = parser.parse_args(argv)

    _configure_logging(args.log_level)

    # Allow running the script from any working directory.
    assets_dir = args.assets_dir or Path.cwd() / "assets"

    log = logging.getLogger(__name__)
    if args.max_retries < 1:
        raise ValueError("--max-retries must be >= 1")
    last_exc: BaseException | None = None
    for attempt in range(1, args.max_retries + 1):
        try:
            commit_sha = sync_assets_to_hf(
                assets_dir=assets_dir,
                repo_id=args.repo_id,
                repo_type=args.repo_type,
                revision=args.revision,
                commit_message=args.commit_message,
                token=args.hf_token,
                allow_patterns=args.allow_patterns,
                ignore_patterns=args.ignore_patterns,
            )
            log.info("Assets uploaded successfully: %s", commit_sha)
            return commit_sha
        except Exception as exc:
            last_exc = exc
            if attempt < args.max_retries:
                log.warning(
                    "Upload failed (attempt %d/%d), retrying in %.1fs: %s",
                    attempt,
                    args.max_retries,
                    args.retry_delay,
                    exc,
                )
                time.sleep(args.retry_delay)
            else:
                raise
    raise last_exc  # only when max_retries was 0 (invalid)


def main() -> None:
    """Entrypoint used by the `__main__` guard."""

    try:
        run()
    except SystemExit:
        # argparse already printed an error/help message; just bubble up.
        raise
    except Exception as exc:  # pragma: no cover - defensive log
        logging.getLogger(__name__).error("Upload failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
