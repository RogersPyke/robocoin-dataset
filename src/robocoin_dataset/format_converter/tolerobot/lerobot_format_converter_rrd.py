import rerun as rr
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
import logging
from typing import Any, Optional

from robocoin_dataset.format_converter.tolerobot.lerobot_format_converter import LerobotFormatConverter
from robocoin_dataset.format_converter.tolerobot.constant import (
    ARGS_KEY,
    FEATURES_KEY,
    IMAGE_KEY,
    OBSERVATION_KEY,
    STATE_KEY,
    SUB_STATE_KEY,
    ACTION_KEY,
    SUB_ACTION_KEY,
    FPS,
    FRAME_IDX_KEY,
)
from robocoin_dataset.format_converter.tolerobot.exceptions import ConfigError

# rrd 预配置常量
DEFAULT_FILL_LIMIT = 6           # 最多向前填充帧数
RESAMPLE_MS_BASE = 1000          # 毫秒基数

class LerobotFormatConverterRrd(LerobotFormatConverter):
    """
    将 RRD（Rerun Data）格式转换为 LeRobot 格式。
    一个 .rrd 文件视为一个 task，含一个完整的 episode（重采样后）。

    配置要求（在 converter_config 中）：
    - fps: 目标帧率
    - features.observation.images[]: { cam_name, args: { rrd_path } }
    - features.observation.state.sub_state[]: { names, args: { rrd_path, range_from, range_to } }
    - features.action.sub_action[]: 同上（通常与 state 相同路径，不同 timeline_offset）
    """

    def __init__(
        self,
        dataset_path: str,
        output_path: str,
        converter_config: dict,
        repo_id: str,
        device_model: str | None = None,
        logger: logging.Logger | None = None,
        video_backend: str = "pyav",
        image_writer_processes: int = 4,
        image_writer_threads: int = 4,
        strict_episodes: int = 3,
        failure_threshold: float = 0.8,
    ) -> None:
        
        ds_path = Path(dataset_path)
        if ds_path.is_file() and ds_path.suffix == ".rrd":
            self._rrd_task_map = {ds_path: ds_path.stem}
        elif ds_path.is_dir():
            rrd_files = list(ds_path.rglob("*.rrd"))
            if not rrd_files:
                raise FileNotFoundError(f"No .rrd files found in {ds_path}")
            self._rrd_task_map = {f: f.stem for f in rrd_files}
        else:
            raise ValueError(f"dataset_path must be a .rrd file or a directory containing .rrd files: {ds_path}")
        # 用于缓存加载后的重采样 DataFrame
        self._rrd_data_cache: dict[Path, pd.DataFrame] = {}
        # 记录每个 task_path（.rrd 文件）对应的重采样帧数
        self._rrd_frame_counts: dict[Path, int] = {}
        super().__init__(
            dataset_path=dataset_path,
            output_path=output_path,
            converter_config=converter_config,
            repo_id=repo_id,
            device_model=device_model,
            logger=logger,
            video_backend=video_backend,
            image_writer_processes=image_writer_processes,
            image_writer_threads=image_writer_threads,
            strict_episodes=strict_episodes,
            failure_threshold=failure_threshold,
        )

    # ---------- 实现抽象方法 ----------
    def _prevalidate_files(self) -> None:
        """
        验证所有 .rrd 文件存在且可通过 rerun 正常打开。
        """
        for rrd_path in self.path_task_dict.keys():
            if not rrd_path.exists():
                raise FileNotFoundError(f"RRD file not found: {rrd_path}")
            try:
                server = rr.server.Server(datasets={"check": [rrd_path]})
                client = rr.catalog.CatalogClient(server.url())
                dataset = client.get_dataset("check")
                # 尝试读取少量数据，验证通道存在
                reader = dataset.filter_contents("/**").reader(index="timestamp")
                sample = reader.to_pandas()[:1]
                if sample.empty:
                    raise ValueError(f"No data readable from {rrd_path}")
            except Exception as e:
                raise ConfigError(f"Invalid RRD file {rrd_path}: {e}") from e

    def _get_tasks(self) -> list[str]:
        return list(self._rrd_task_map.values())

    def _get_dataset_task_paths(self) -> dict[Path, str]:
        return self._rrd_task_map

    def _get_task_episodes_num(self, task_path: Path) -> int:
        """当前版本每个 .rrd 仅含 1 个 episode"""
        return 1

    def _get_episode_frames_num(self, task_path: Path, ep_idx: int) -> int:
        self._load_and_cache_rrd(task_path)
        return self._rrd_frame_counts[task_path]

    def _load_and_cache_rrd(self, task_path):
        if task_path in self._rrd_data_cache:
            return self._rrd_data_cache[task_path]

        target_fps = self.converter_config.get(FPS, 30)
        fill_limit = self.converter_config.get("fill_limit", DEFAULT_FILL_LIMIT)

        # 1. 读取原始数据（与独立脚本完全一致）
        server = rr.server.Server(datasets={"data": [task_path]})
        client = rr.catalog.CatalogClient(server.url())
        dataset = client.get_dataset("data")
        # 先用 reader 获取 Reader 对象，再转为 DataFrame
        reader = dataset.filter_contents("/**").reader(index="timestamp")
        raw_df = reader.to_pandas()

        # 2. 时间戳处理：保留 timestamp 列并重新设置为索引（关键！）
        raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"])
        raw_df = raw_df.sort_values("timestamp").set_index("timestamp")

        # 3. 重采样至目标 fps
        resample_interval = f"{1000 / target_fps:.2f}ms"
        first = raw_df.resample(resample_interval).first()
        last = raw_df.resample(resample_interval).last()
        filled = last.ffill(limit=fill_limit)
        df_final = first.combine_first(filled).dropna(how='any')

        # 4. 缓存
        self._rrd_data_cache[task_path] = df_final
        self._rrd_frame_counts[task_path] = len(df_final)
        if self.logger:
            self.logger.info(
                f"Loaded & resampled {task_path.name}: {len(raw_df)} rows → {len(df_final)} frames @ {target_fps} fps"
            )
        return df_final

    def _prepare_episode_images_buffer(self, task_path: Path, ep_idx: int, is_test: bool = False) -> pd.DataFrame:
        return self._load_and_cache_rrd(task_path)

    def _prepare_episode_states_buffer(self, task_path: Path, ep_idx: int, is_test: bool = False) -> pd.DataFrame:
        return self._load_and_cache_rrd(task_path)

    def _prepare_episode_actions_buffer(self, task_path: Path, ep_idx: int, is_test: bool = False) -> pd.DataFrame:
        return self._load_and_cache_rrd(task_path)

    def _get_frame_image(self, task_path, ep_idx, frame_idx, args_dict, images_buffer=None):
        if images_buffer is None:
            images_buffer = self._prepare_episode_images_buffer(task_path, ep_idx)

        column = args_dict["rrd_path"]
        try:
            blob_row = images_buffer[column].iloc[frame_idx]
        except IndexError:
            raise IndexError(f"frame_idx {frame_idx} out of bounds for {task_path}")
        if blob_row is None:
            raise ValueError(f"Empty image blob at frame {frame_idx}, column {column}")

        # 独立脚本验证过的解码方式：blob_row 是 shape=(1,) 的 ndarray，取 [0] 得到 bytes
        if isinstance(blob_row, np.ndarray) and blob_row.ndim >= 1:
            raw_bytes = blob_row[0]
        elif isinstance(blob_row, bytes):
            raw_bytes = blob_row
        else:
            raise ValueError(f"Unexpected image blob type: {type(blob_row)}")

        img = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to decode image at frame {frame_idx}, column {column}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _get_frame_sub_states(
        self,
        task_path: Path,
        ep_idx: int,
        frame_idx: int,
        args_dict: dict,
        sub_states_buffer: Any = None,
    ) -> np.ndarray:
        if sub_states_buffer is None:
            sub_states_buffer = self._prepare_episode_states_buffer(task_path, ep_idx)

        column = args_dict["rrd_path"]
        from_idx = args_dict.get("range_from", None)
        to_idx = args_dict.get("range_to", None)

        try:
            cell = sub_states_buffer[column].iloc[frame_idx]
        except IndexError:
            raise IndexError(f"frame_idx {frame_idx} out of bounds for {task_path}")

        if cell is None or (isinstance(cell, float) and np.isnan(cell)):
            raise ValueError(f"Missing state data at frame {frame_idx}, column {column}")

        val = self._extract_array(cell)
        if from_idx is not None and to_idx is not None:
            if to_idx > len(val):
                raise ValueError(f"range_to {to_idx} exceeds data length {len(val)} for {column}")
            val = val[from_idx:to_idx]
        return val

    def _get_frame_sub_actions(
        self,
        task_path: Path,
        ep_idx: int,
        frame_idx: int,
        args_dict: dict,
        sub_actions_buffer: Any = None,
    ) -> np.ndarray:
        # 实现与 _get_frame_sub_states 完全一致
        if sub_actions_buffer is None:
            sub_actions_buffer = self._prepare_episode_actions_buffer(task_path, ep_idx)

        column = args_dict["rrd_path"]
        from_idx = args_dict.get("range_from", None)
        to_idx = args_dict.get("range_to", None)

        try:
            cell = sub_actions_buffer[column].iloc[frame_idx]
        except IndexError:
            raise IndexError(f"frame_idx {frame_idx} out of bounds for {task_path}")

        if cell is None or (isinstance(cell, float) and np.isnan(cell)):
            raise ValueError(f"Missing action data at frame {frame_idx}, column {column}")

        val = self._extract_array(cell)
        if from_idx is not None and to_idx is not None:
            if to_idx > len(val):
                raise ValueError(f"range_to {to_idx} exceeds data length {len(val)} for {column}")
            val = val[from_idx:to_idx]
        return val
    

    def _extract_array(self, cell):
        """
        递归展平任意嵌套的 list/tuple/np.ndarray，返回一维 float32 数组。
        """
        def flatten(obj):
            if isinstance(obj, (list, tuple, np.ndarray)):
                for item in obj:
                    yield from flatten(item)
            else:
                yield obj

        flat_values = list(flatten(cell))
        return np.array(flat_values, dtype=np.float32)

    # ---------- 可选：覆盖 __del__ 清理缓存 ----------
    def __del__(self):
        self._rrd_data_cache.clear()
        super().__del__()