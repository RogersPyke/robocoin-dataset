"""Dataset utilities for LeRobot datasets.
Here we assume all dataset Lerobot format, and do:
    1. Sample frames from dataset.
    2. Load test.(check structure and do video decoding-main time cost.)

This module provides utilities for:
- Creating LeRobot datasets.
- Episode sampling with downsampling support (single & multi-episode)
- Worker initialization for DataLoader
"""

import logging
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    pass

######################### Constants #########################

TASK_CATEGORY = "dataloader_detection"

######################### Logging setup #########################

# Suppress noisy INFO logs from httpx (used by HuggingFace datasets)
# These logs show 404s when checking for optional files - doesn't affect functionality
logging.getLogger("httpx").setLevel(logging.WARNING)

# Auto-add vendored lerobot to path and import
try:
    vendored_path = Path(__file__).resolve().parents[3] / "third_parties" / "robocoin-lerobot" / "src"
    if vendored_path.is_dir() and str(vendored_path) not in sys.path:
        sys.path.insert(0, str(vendored_path))
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore
except Exception:
    LeRobotDataset = None  # type: ignore

######################### business logic functions #########################

class MultiEpisodeSampler(torch.utils.data.Sampler):  # type: ignore

    def __init__(
        self,
        dataset: "LeRobotDataset",
        episode_indices: list[int],
        sample_ratio: float = 1.0,
    ) -> None:
        if not 0.0 <= sample_ratio <= 1.0:
            raise ValueError(f"sample_ratio must be between 0 and 1, got {sample_ratio}")

        self.frame_ids = []
        self.episode_boundaries = []

        for ep_idx in episode_indices:
            from_idx = dataset.episode_data_index["from"][ep_idx].item()
            to_idx = dataset.episode_data_index["to"][ep_idx].item()

            start_pos = len(self.frame_ids)

            if sample_ratio == 1.0:
                episode_frames = list(range(from_idx, to_idx))
            else:
                total_frames = to_idx - from_idx
                num_samples = max(1, int(total_frames * sample_ratio))

                if num_samples == 1:
                    episode_frames = [from_idx]
                else:
                    step = (total_frames - 1) / (num_samples - 1)
                    episode_frames = [from_idx + int(i * step) for i in range(num_samples)]

            self.frame_ids.extend(episode_frames)
            self.episode_boundaries.append((ep_idx, start_pos, len(self.frame_ids)))

    def __iter__(self) -> Iterator:
        return iter(self.frame_ids)

    def __len__(self) -> int:
        return len(self.frame_ids)


def create_lerobot_dataset(
    repo_id: str,
    root: str | Path | None = None,
    episodes: list[int] | None = None,
    image_transforms: "callable | None" = None,
    delta_timestamps: dict[list[float]] | None = None,
    tolerance_s: float = 0.001,
    revision: str | None = None,
    force_cache_sync: bool = False,
    download_videos: bool = True,
    video_backend: str = "pyav",
    batch_encoding_size: int = 1,
) -> "LeRobotDataset":

    if LeRobotDataset is None:
        raise RuntimeError(
            "LeRobotDataset is not available."
        )

    return LeRobotDataset(
        repo_id=repo_id,
        root=Path(root) if root is not None else None,
        episodes=episodes,
        image_transforms=image_transforms,
        delta_timestamps=delta_timestamps,
        tolerance_s=tolerance_s,
        revision=revision,
        force_cache_sync=force_cache_sync,
        download_videos=download_videos,
        video_backend=video_backend,
        batch_encoding_size=batch_encoding_size,
    )


def _run_detection(
    repo_path: str | Path,
    episode_indices: str | int | list[int] | None = "all",
    sample_ratio: float = 0.1,
    batch_size: int = 32,
    num_workers: int = 0,
    client_id: str | None = None,
    tqdm_position: int = 0,
    logger: logging.Logger | None = None,
) -> dict:
    """Fast dataloader detection using multi-episode sampling."""
    import contextlib
    import gc
    import time
    import traceback

    from tqdm import tqdm  # type: ignore

    _logger = logger or logging.getLogger(__name__)
    repo_path = Path(repo_path)
    start_time = time.perf_counter()
    result = {"success": False, "dataset_path": str(repo_path), "sample_ratio": sample_ratio,
              "backend": "pyav", "total_frames_sampled": 0}

    ds = None
    try:
        # Load dataset and parse episodes
        ds = create_lerobot_dataset(repo_id=repo_path.name, root=repo_path, download_videos=False)
        total_episodes = len(ds.episode_data_index["from"])
        episodes_to_test = _parse_episode_specification(episode_indices, total_episodes)

        result.update({
            "total_episodes_in_dataset": total_episodes,
            "episodes_tested": episodes_to_test,
        })

        if not episodes_to_test:
            result["success"] = True
            return result

        # Create sampler and dataloader
        sampler = MultiEpisodeSampler(ds, episodes_to_test, sample_ratio)
        dl = torch.utils.data.DataLoader(
            ds, num_workers=num_workers, batch_size=batch_size, sampler=sampler,
            worker_init_fn=_worker_init_suppress_output if num_workers > 0 else None,
        )

        # Run detection with progress bar (suppress stdout to avoid LeRobot prints)
        dataset_name = repo_path.name[:40]
        client_prefix = f"[{client_id}] " if client_id else ""

        with contextlib.redirect_stdout(open(os.devnull, "w")):
            for batch in tqdm(dl, total=len(dl), desc=f"{client_prefix}📦 {dataset_name}",
                            unit="batch", file=sys.stderr, position=tqdm_position, leave=False):
                result["total_frames_sampled"] += len(batch["index"]) if "index" in batch else batch_size

        # Properly shutdown DataLoader workers to avoid semaphore leaks
        if hasattr(dl, '_iterator') and dl._iterator is not None:
            dl._iterator._shutdown_workers()
        del dl
        gc.collect()

        # Calculate timing stats
        elapsed = time.perf_counter() - start_time
        result.update({
            "success": True,
            "total_time_s": elapsed,
            "time_per_episode_s": elapsed / len(episodes_to_test),
            "time_per_frame_s": elapsed / result["total_frames_sampled"] if result["total_frames_sampled"] else 0.0,
        })

    except Exception as e:
        result.update({
            "error_message": f"{e}\n{traceback.format_exc()}",
            "total_time_s": time.perf_counter() - start_time,
            "success": False,
        })
        # Log error with full traceback unconditionally
        _logger.exception(f"❌ Detection failed: {e}")

    finally:
        if ds:
            del ds
        gc.collect()

    return result


######################### helper functions #########################

def _worker_init_suppress_output(worker_id: int) -> None:
    """Suppress stdout/stderr in DataLoader worker processes.

    This prevents print statements in third-party libraries (like lerobot's video_utils.py)
    from flooding the console when using num_workers > 0.

    Args:
        worker_id: Worker process ID (provided by PyTorch DataLoader)
    """
    import atexit

    # Close existing file objects if they exist
    if hasattr(sys.stdout, "close") and sys.stdout not in (sys.__stdout__,):
        try:
            sys.stdout.close()
        except Exception:
            pass
    if hasattr(sys.stderr, "close") and sys.stderr not in (sys.__stderr__,):
        try:
            sys.stderr.close()
        except Exception:
            pass

    # Open new file handles
    devnull_out = open(os.devnull, "w")
    devnull_err = open(os.devnull, "w")

    sys.stdout = devnull_out
    sys.stderr = devnull_err

    # Register cleanup to close handles when worker exits
    def cleanup() -> None:
        try:
            devnull_out.close()
        except Exception:
            pass
        try:
            devnull_err.close()
        except Exception:
            pass

    atexit.register(cleanup)


def _parse_episode_specification(
    episode_spec: str | int | list[int] | None,
    total_episodes: int,
) -> list[int]:
    """Parse episode specification: None/"all", int, list[int], "0,1,2", "0-5", "0-5,10".

    Returns sorted unique episode indices, validates range [0, total_episodes-1].
    """
    # Handle None or "all"
    if episode_spec is None or (isinstance(episode_spec, str) and episode_spec.lower() == "all"):
        return list(range(total_episodes))

    # Convert int to list for unified handling
    if isinstance(episode_spec, int):
        episode_spec = [episode_spec]

    # Parse list[int] or string
    if isinstance(episode_spec, list):
        episodes = episode_spec
    elif isinstance(episode_spec, str):
        episodes = []
        for part in episode_spec.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part and not part.startswith("-"):
                start, end = map(int, part.split("-", 1))
                if start > end:
                    raise ValueError(f"Invalid range: {part} (start > end)")
                episodes.extend(range(start, end + 1))
            else:
                episodes.append(int(part))
    else:
        raise ValueError(f"Invalid episode specification type: {type(episode_spec)}")

    # Validate range
    for ep in episodes:
        if ep < 0 or ep >= total_episodes:
            raise ValueError(f"Episode {ep} out of range [0, {total_episodes - 1}]")

    return sorted(set(episodes))



#=== Public API Exports ===

__all__ = [
    "TASK_CATEGORY",
    "LeRobotDataset",
    "MultiEpisodeSampler",
    "create_lerobot_dataset",
    "create_episode_dataloader",
    "_run_detection",
    "_parse_episode_specification",
]
