"""Utility functions for dataloader checker.

This module provides:
- EpisodeSampler: A sampler to pick a subset of frames from a dataset.
- load_repo: A function to load a LeRobot dataset and iterate through its dataloader.
"""

import random
import logging
from collections.abc import Iterator
from pathlib import Path

import torch
import torch.utils.data
import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore

class EpisodeSampler(torch.utils.data.Sampler):
    """
    Function: 
    A custom sampler that samples a percentage of frames from a LeRobotDataset.
    
    Expected input format:
    - dataset: LeRobotDataset instance.
    - sample_rate: float, range [0.0, 1.0], percentage of frames to sample.
    
    Expected output format:
    - Iterator of sampled frame indices.
    
    Expected usage/scenario:
    - Used to quickly check if a dataset can be loaded and iterated without reading every single frame.
    Speeds up dataloader checks by reducing the number of frames to process.
    """
    def __init__(self, dataset: LeRobotDataset, sample_rate: float = 0.1) -> None:
        self.frame_ids = random.sample(
            range(dataset.num_frames), k=int(dataset.num_frames * sample_rate)
        )
        # Apply downsampling if needed, though random.sample already reduces count.
        # Keeping the original logic from dataloader_checker.py
        self.frame_ids = self.frame_ids[:: int(1 / sample_rate)] if sample_rate < 1.0 and sample_rate > 0 else self.frame_ids

    def __iter__(self) -> Iterator:
        return iter(self.frame_ids)

    def __len__(self) -> int:
        return len(self.frame_ids)


def load_repo(
    repo_hardlink: str | Path,
    num_workers: int = 8,
    sample_rate: float = 0.1,
) -> None:
    """
    Function: 
    Loads a LeRobot dataset from a hardlink path and iterates through it using a DataLoader.
    
    Expected input format:
    - repo_hardlink: str or Path, path to the dataset root (hardlink).
    - num_workers: int, number of worker processes for DataLoader.
    - sample_rate: float, percentage of frames to sample (0.0-1.0).
    
    Expected output format:
    - None. Raises exceptions if loading or iteration fails.
    
    Expected usage/scenario:
    - Validates that the dataset structure is correct and videos can be decoded.
    Used by both local and distributed dataloader checking.
    """
    repo_hardlink = Path(repo_hardlink).expanduser().absolute()
    dataset = LeRobotDataset(
        repo_id="test/test_repo",
        root=repo_hardlink,
        video_backend="pyav",  # torchcodec is not supported.(2.0)
    )
    sampler = EpisodeSampler(dataset, sample_rate=sample_rate)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        sampler=sampler,
        num_workers=num_workers,
    )
    for _ in tqdm.tqdm(dataloader, total=len(dataloader), desc=f"Checking {Path(repo_hardlink).name}"):
        pass

__all__ = ["EpisodeSampler", "load_repo"]
