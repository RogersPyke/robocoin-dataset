"""
Preview video selection for dataset metadata and page_sync.

Purpose:
    Central logic to select a single video that best shows surroundings and
    full task trajectory (high/head/front/top camera). Used by:
    - metadata collect stage: to set video_url in info.yaml (README/Hub).
    - page_sync: to sample one video for compression or display.

Dependencies:
    - pathlib.Path, logging.
"""

import logging
from pathlib import Path

# Priority keywords for camera folders (same as page_sync): prefer views that
# show the whole scene and task trajectory.
PREVIEW_PRIORITY_KEYWORDS = ["high", "top", "head", "front"]


def select_preview_video_path(
    dataset_path: Path,
    logger: logging.Logger,
) -> str | None:
    """
    Select one preview video path (relative to dataset root), deterministic.

    Chooses the first available video from camera folders that match
    PREVIEW_PRIORITY_KEYWORDS (high, top, head, front), so the README can
    show a representative task trajectory and surroundings.

    Input:
        dataset_path (Path): Dataset root (hardlink dir). Expects
            videos/chunk-*/observation.images.*/*.mp4.
        logger (logging.Logger): Logger for diagnostics.

    Output:
        str | None: Relative path from dataset_path (e.g.
            "videos/chunk-000/observation.images.cam_high_rgb/episode_000000.mp4"),
            or None if no videos found.

    Usage:
        Called by metadata apply_auto_fields to fill video_url in info.yaml.
    """
    root = Path(dataset_path).resolve()
    videos_dir = root / "videos"
    if not videos_dir.exists() or not videos_dir.is_dir():
        logger.debug("[VID_COLL] No videos directory at %s", videos_dir)
        return None

    all_videos = list(videos_dir.glob("chunk-*/observation.images.*/*.mp4"))
    if not all_videos:
        logger.debug("[VID_COLL] No observation.images.*/*.mp4 under %s", videos_dir)
        return None

    priority = [
        v for v in all_videos
        if any(kw in str(v).lower() for kw in PREVIEW_PRIORITY_KEYWORDS)
    ]
    candidates = priority if priority else all_videos

    # Deterministic: sort by chunk, then camera dir, then episode filename
    def _sort_key(p: Path) -> tuple:
        parts = p.relative_to(root).parts
        # parts e.g. ("videos", "chunk-000", "observation.images.cam_high_rgb", "episode_000000.mp4")
        chunk = parts[1] if len(parts) > 1 else ""
        cam = parts[2] if len(parts) > 2 else ""
        name = parts[3] if len(parts) > 3 else p.name
        return (chunk, cam, name)

    sorted_candidates = sorted(candidates, key=_sort_key)
    first = sorted_candidates[0]
    rel = first.relative_to(root).as_posix()
    logger.info("[VID_COLL] Selected preview video (relative): %s", rel)
    return rel


def get_one_video_path(
    dataset_path: Path | str,
    logger: logging.Logger,
    random: bool = False,
) -> str | None:
    """
    Get one video path (absolute) for page_sync or other consumers.

    Same camera priority as select_preview_video_path. When random=False,
    returns the same deterministic preview video (absolute path). When
    random=True, returns a random one from the priority set.

    Input:
        dataset_path (Path | str): Dataset root directory.
        logger (logging.Logger): Logger for diagnostics.
        random (bool): If True, pick randomly from priority videos; if False,
            use deterministic first video.

    Output:
        str | None: Absolute path to one .mp4 file, or None if none found.

    Usage:
        Called by page_sync when sampling one video for compression/display.
    """
    import random as _random_mod

    root = Path(dataset_path).expanduser().resolve()
    videos_dir = root / "videos"
    if not videos_dir.exists() or not videos_dir.is_dir():
        logger.debug("[VID_COLL] No videos directory at %s", videos_dir)
        return None

    all_videos = list(videos_dir.glob("chunk-*/observation.images.*/*.mp4"))
    if not all_videos:
        logger.debug("[VID_COLL] No observation.images.*/*.mp4 under %s", videos_dir)
        return None

    priority = [
        v for v in all_videos
        if any(kw in str(v).lower() for kw in PREVIEW_PRIORITY_KEYWORDS)
    ]
    candidates = priority if priority else all_videos

    if random:
        chosen = _random_mod.choice(candidates)
        logger.info("[VID_COLL] Sampled video (random): %s", chosen)
        return str(chosen.resolve())

    def _sort_key(p: Path) -> tuple:
        parts = p.relative_to(root).parts
        chunk = parts[1] if len(parts) > 1 else ""
        cam = parts[2] if len(parts) > 2 else ""
        name = parts[3] if len(parts) > 3 else p.name
        return (chunk, cam, name)

    sorted_candidates = sorted(candidates, key=_sort_key)
    first = sorted_candidates[0]
    logger.info("[VID_COLL] Selected video (deterministic): %s", first)
    return str(first.resolve())
