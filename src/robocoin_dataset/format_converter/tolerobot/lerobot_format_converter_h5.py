import io
import logging
import tempfile
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import cv2
import h5py
import numpy as np
from natsort import natsorted
from PIL import Image
import json

from robocoin_dataset.format_converter.tolerobot.constant import (
    ARGS_KEY,
    DATASET_UUID_FILE,
    DESCRIBE_TXT_FILE,
    DESCRIPTION_TXT_FILE,
    DEVICE_MODEL_ANNOTATION_FILE,
    FEATURES_KEY,
    H5_SUFFIX,
    HDF5_SUFFIX,
    IMAGE_KEY,
    LOCAL_DATASET_INFO_FILE,
    LOCAL_TASK_INFO_FILE,
    OBSERVATION_KEY,
    STATE_KEY,
    SUB_STATE_KEY,
)
from robocoin_dataset.format_converter.tolerobot.lerobot_format_converter import (
    LerobotFormatConverter,
)
from robocoin_dataset.format_converter.utils.h5_file_cache import (
    H5FileCache,
)


@dataclass
class H5Buffer:
    h5_data: dict | None = None
    task_path: Path | None = None
    ep_idx: int | None = None


ALLOWED_RULES = {
    "exact_names": {
        LOCAL_TASK_INFO_FILE,
        LOCAL_DATASET_INFO_FILE,
        DATASET_UUID_FILE,
        DESCRIPTION_TXT_FILE,
        DESCRIBE_TXT_FILE,
        DEVICE_MODEL_ANNOTATION_FILE,
    },  # 允许的完整文件名
    "allowed_suffixes": {H5_SUFFIX, HDF5_SUFFIX},  # 允许的后缀
    "other_suffixes": {".yaml", ".txt", ".mp4", ".db", ".doc", ".docx", ".hdf5"},
}

# NAS_EADIR = "@eaDir"
NAS_SYSFILE_TAG = "@"


def is_allowed_file(file_path: Path) -> bool:
    filename = file_path.name
    suffix = file_path.suffix

    ret = (
        filename in ALLOWED_RULES["exact_names"]
        or suffix in ALLOWED_RULES["allowed_suffixes"]
        or NAS_SYSFILE_TAG in filename
        or suffix in ALLOWED_RULES["other_suffixes"]
    )
    if ret:
        if suffix in ALLOWED_RULES["allowed_suffixes"]:
            if file_path.stat().st_size < 1024:
                return False

    return ret


def find_unexpected_files(directory: Path, include_hidden: bool = False) -> list[str]:
    directory = Path(directory)

    if not directory.exists():
        parent_dir = directory.parent
        raise FileNotFoundError(
            f"❌ Directory does not exist.\n"
            f"   📂 Requested directory: {directory}\n"
            f"   📁 Parent directory: {parent_dir} {'(exists)' if parent_dir.exists() else '(NOT FOUND)'}\n"
            f"   💡 Check if path is correct and directory has been created"
        )
    if not directory.is_dir():
        raise NotADirectoryError(
            f"❌ Path is not a directory.\n"
            f"   📄 Path: {directory}\n"
            f"   💡 This path points to a file, not a directory"
        )

    unexpected_files: list[str] = []

    # 🔍 递归遍历所有文件
    for file_path in directory.rglob("*"):
        if file_path.is_file():
            # 跳过隐藏文件（可选）
            if not include_hidden and file_path.name.startswith("."):
                continue

            if not is_allowed_file(file_path):
                unexpected_files.append(str(file_path))

    return unexpected_files


def explore_hdf5_group(group, prefix="") -> None:  # noqa: ANN001
    for key in group.keys():
        item = group[key]
        if isinstance(item, h5py.Dataset):
            pass
        elif isinstance(item, h5py.Group):
            explore_hdf5_group(item, prefix + "  ")


def validate_h5file(h5_file_path: Path) -> list[Path]:
    if not h5_file_path.exists():
        parent_dir = h5_file_path.parent
        available_files = []
        if parent_dir.exists():
            available_files = [f.name for f in parent_dir.glob("*.h5") + parent_dir.glob("*.hdf5")]
        
        raise FileNotFoundError(
            f"❌ H5 file does not exist.\n"
            f"   🗂️  Expected file: {h5_file_path}\n"
            f"   📂 Parent directory: {parent_dir}\n"
            f"   📋 H5 files in directory: {available_files if available_files else 'None'}\n"
            f"   💡 Check if:\n"
            f"      1. File name is correct\n"
            f"      2. File has been created/recorded\n"
            f"      3. Path is correct"
        )

    if not h5_file_path.is_file():
        raise Exception(
            f"❌ Path is not a file.\n"
            f"   📂 Path: {h5_file_path}\n"
            f"   💡 This path points to a directory, not a file"
        )

    try:
        with h5py.File(h5_file_path, "r") as h5_file:
            explore_hdf5_group(h5_file)
    except Exception as e:
        raise Exception(
            f"❌ Error validating H5 file.\n"
            f"   🗂️  File: {h5_file_path}\n"
            f"   ❌ Error: {e!s}\n"
            f"   💡 H5 file may be corrupted or incompatible format"
        )


class LerobotFormatConverterHdf5(LerobotFormatConverter):
    def __init__(
        self,
        dataset_path: str,
        output_path: str,
        converter_config: dict,
        repo_id: str,
        device_model: str,
        logger: logging.Logger | None = None,
        video_backend: str = "pyav",
        image_writer_processes: int = 4,
        image_writer_threads: int = 4,
    ) -> None:
        self.h5_buffer: H5Buffer = H5Buffer()
        self._image_is_iobytes = True
        # 🚀 H5文件句柄缓存，大幅提升读取性能（需要在super().__init__之前初始化，因为_prevalidate_files会用到）
        self._h5_file_cache = H5FileCache(max_cache_size=100, logger=logger)
        # 存储损坏的H5文件列表，用于在转换时跳过
        self._invalid_h5_files: set = set()

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
        )

    def _prevalidate_files(self) -> None:
        unexpected_files: list[Path] = []
        for path in self.path_task_dict.keys():
            if not path.exists():
                parent_dir = path.parent
                raise FileNotFoundError(
                    f"❌ Task path does not exist.\n"
                    f"   📁 Task path: {path}\n"
                    f"   📂 Parent directory: {parent_dir} {'(exists)' if parent_dir.exists() else '(NOT FOUND)'}\n"
                    f"   💡 Check if task directory has been created"
                )
            if path.is_file():
                raise ValueError(
                    f"❌ Task path is a file, not a directory.\n"
                    f"   📄 Path: {path}\n"
                    f"   💡 Task path should be a directory containing episodes"
                )

            unexpected_files.extend(find_unexpected_files(path))

        if unexpected_files:
            # 改为警告而不是抛出异常，不应因文件命名问题导致整个验证失败
            warning_msg = (
                f"⚠️  Found unexpected files in dataset directory (non-blocking).\n"
                f"   📂 Task paths checked: {len(self.path_task_dict)} directories\n"
                f"   📋 Unexpected files ({len(unexpected_files)}):\n"
            )
            # 只显示前10个，避免输出过长
            for file_path in unexpected_files[:10]:
                warning_msg += f"      - {file_path}\n"
            if len(unexpected_files) > 10:
                warning_msg += f"      ... and {len(unexpected_files) - 10} more files\n"
            warning_msg += (
                "   💡 These files will be ignored during conversion.\n"
                "   💡 Check if files have incorrect naming (e.g., 'episode_139hdf5' should be 'episode_139.hdf5')"
            )
            if self.logger:
                self.logger.warning(warning_msg)

        invalid_h5_files = []
        for path in self.task_episode_h5file_paths:
            # 收集所有H5文件（包括.h5, .hdf5和命名错误的如episode_139hdf5）
            h5_files_to_validate = []
            h5_files_to_validate.extend(path.rglob("*.h5"))
            h5_files_to_validate.extend(path.rglob("*.hdf5"))
            # 额外查找命名错误的文件
            for file in path.rglob("*"):
                if file.is_file() and (file.name.endswith("hdf5") or file.name.endswith("h5")):
                    if file not in h5_files_to_validate:
                        h5_files_to_validate.append(file)
            
            # 验证所有找到的H5文件
            for file in h5_files_to_validate:
                try:
                    validate_h5file(file)
                except Exception:  # noqa: PERF203
                    invalid_h5_files.append(file)

        # 容错处理：跳过损坏的文件而不是抛出异常
        if invalid_h5_files:
            if self.logger:
                self.logger.warning(
                    f"⚠️  发现 {len(invalid_h5_files)} 个损坏的H5文件，将自动跳过这些文件\n"
                    f"   📋 损坏文件列表（已保存）:"
                )
                for h5_file in invalid_h5_files[:10]:  # 只显示前10个
                    self.logger.warning(f"      - {h5_file}")
                if len(invalid_h5_files) > 10:
                    self.logger.warning(f"      ... 以及 {len(invalid_h5_files) - 10} 个其他文件")
                
                # 保存完整的损坏文件列表到日志目录
                try:
                    output_path = Path(self.output_path)
                    corrupted_list_file = output_path / "corrupted_episodes.txt"
                    with open(corrupted_list_file, "w") as f:
                        f.write(f"损坏的H5文件列表 (总计: {len(invalid_h5_files)})\n")
                        f.write("=" * 80 + "\n\n")
                        for h5_file in invalid_h5_files:
                            f.write(f"{h5_file}\n")
                    self.logger.warning(f"   📄 完整列表已保存到: {corrupted_list_file}")
                except Exception as e:
                    self.logger.warning(f"   ⚠️  无法保存损坏文件列表: {e}")
            
            # 存储损坏文件列表，以便后续跳过
            self._invalid_h5_files = set(invalid_h5_files)

        # 验证HDF5文件内部结构
        self._validate_h5_structure()

    def _validate_h5_structure(self) -> None:
        """验证HDF5文件内部结构是否与配置的版本相符"""
        if self.logger:
            self.logger.info("Validating HDF5 internal structure...")

        # 收集所有需要验证的路径
        required_paths = set()
        compressed_video_image_paths = set()  # 跟踪使用compressed video的image paths

        # 从图像配置中收集路径
        for image_config in self.converter_config[FEATURES_KEY][OBSERVATION_KEY][IMAGE_KEY]:
            if ARGS_KEY in image_config and "h5_path" in image_config[ARGS_KEY]:
                h5_path = image_config[ARGS_KEY]["h5_path"]
                use_compressed_video = image_config[ARGS_KEY].get("use_compressed_video", False)
                
                if use_compressed_video:
                    # 对于compressed video，images是空的，不验证
                    # 而是验证video和video_index
                    compressed_video_image_paths.add(h5_path)
                    video_path = h5_path.replace("/images", "/video")
                    video_index_path = h5_path.replace("/images", "/video_index")
                    required_paths.add(video_path)
                    required_paths.add(video_index_path)
                else:
                    required_paths.add(h5_path)

        # 从状态配置中收集路径
        for state_config in self.converter_config[FEATURES_KEY][OBSERVATION_KEY][STATE_KEY][
            SUB_STATE_KEY
        ]:
            if ARGS_KEY in state_config and "h5_path" in state_config[ARGS_KEY]:
                required_paths.add(state_config[ARGS_KEY]["h5_path"])

        # 从动作配置中收集路径
        if "sub_action" in self.converter_config[FEATURES_KEY]["action"]:
            for action_config in self.converter_config[FEATURES_KEY]["action"]["sub_action"]:
                if ARGS_KEY in action_config and "h5_path" in action_config[ARGS_KEY]:
                    required_paths.add(action_config[ARGS_KEY]["h5_path"])

        if not required_paths:
            if self.logger:
                self.logger.warning(
                    "No h5_path found in configuration, skipping H5 structure validation"
                )
            return

        # 验证每个任务路径下的H5文件
        validation_errors = []
        for task_path in self.path_task_dict.keys():
            h5_files = self.task_episode_h5file_paths.get(task_path, [])

            if not h5_files:
                validation_errors.append(f"No H5 files found in task path: {task_path}")
                continue

            # 验证第一个H5文件作为样本（假设同一任务下的H5文件结构一致）
            sample_h5_file = h5_files[0]
            try:
                # 🚀 使用H5FileCache提升性能
                with self._h5_file_cache.open(sample_h5_file) as h5_file:
                    missing_paths = []
                    invalid_paths = []

                    for required_path in required_paths:
                        if required_path not in h5_file:
                            missing_paths.append(required_path)
                        else:
                            # 检查是否为有效的数据集
                            try:
                                dataset = h5_file[required_path]
                                if not isinstance(dataset, h5py.Dataset):
                                    invalid_paths.append(f"{required_path} (not a dataset)")
                                else:
                                    # 对于 video 数据集，shape=() 是正常的（单个压缩视频blob）
                                    # 对于其他数据集，shape=(0,) 或包含0维度的shape是空数据集
                                    is_video_dataset = '/video' in required_path and not required_path.endswith('/video_index')
                                    
                                    if is_video_dataset:
                                        # Video datasets: scalar shape=() is valid (compressed video blob)
                                        # Only flag if it has dimensions with 0 size like shape=(0,) or (10, 0, 3)
                                        if len(dataset.shape) > 0 and any(dim == 0 for dim in dataset.shape):
                                            invalid_paths.append(f"{required_path} (empty video dataset with shape {dataset.shape})")
                                    else:
                                        # Non-video datasets: both shape=() and shape=(0,) are suspicious
                                        if len(dataset.shape) == 0:
                                            invalid_paths.append(f"{required_path} (scalar dataset, expected array)")
                                        elif any(dim == 0 for dim in dataset.shape):
                                            invalid_paths.append(f"{required_path} (empty dataset with shape {dataset.shape})")
                            except Exception as e:
                                invalid_paths.append(f"{required_path} (error: {e})")

                    if missing_paths:
                        validation_errors.append(
                            f"   🗂️  File: {sample_h5_file.name}\n"
                            f"      ❌ Missing paths: {missing_paths}"
                        )

                    if invalid_paths:
                        validation_errors.append(
                            f"   🗂️  File: {sample_h5_file.name}\n"
                            f"      ⚠️  Invalid datasets: {invalid_paths}"
                        )

                    if self.logger and not missing_paths and not invalid_paths:
                        self.logger.info(f"✓ H5 structure validation passed for task: {task_path.name}")

            except Exception as e:
                validation_errors.append(
                    f"   🗂️  File: {sample_h5_file.name}\n"
                    f"      ❌ Read error: {e!s}"
                )

        if validation_errors:
            error_msg = (
                f"❌ H5 structure validation failed.\n"
                f"   📊 Issues found in {len(validation_errors)} file(s)\n"
                f"   📋 Validation errors:\n"
            )
            error_msg += "\n".join(validation_errors)
            error_msg += (
                f"\n   💡 Check if:\n"
                f"      1. H5 files match expected structure\n"
                f"      2. Config h5_path values are correct\n"
                f"      3. All required datasets exist in H5 files"
            )
            raise ValueError(error_msg)

        if self.logger:
            self.logger.info("H5 structure validation completed successfully")

    def _get_frame_image(
        self,
        task_path: Path,
        ep_idx: int,
        frame_idx: int,
        args_dict: dict,
        images_buffer: any = None,
    ) -> np.ndarray:
        if not images_buffer:
            images_buffer = self._prepare_episode_images_buffer(task_path, ep_idx)
          
        try:
            h5_path = args_dict["h5_path"]
        except KeyError as e:
            available_keys = list(args_dict.keys())
            raise KeyError(
                f"❌ Missing required 'h5_path' in args_dict.\n"
                f"   📁 Location: task={task_path.name}, ep_idx={ep_idx}, frame_idx={frame_idx}\n"
                f"   📋 Available keys: {available_keys}\n"
                f"   💡 args_dict must contain 'h5_path' key specifying the H5 dataset path"
            ) from e

        # 根据配置检查是否使用压缩视频格式
        use_compressed_video = args_dict.get("use_compressed_video", False)
        
        if use_compressed_video:
            # 使用压缩视频格式
            video_path = h5_path.replace('/images', '/video')
            video_index_path = h5_path.replace('/images', '/video_index')
            
            if video_path not in images_buffer or video_index_path not in images_buffer:
                available_paths = list(images_buffer.keys())[:10]
                raise KeyError(
                    f"❌ Compressed video format configured but video data not found.\n"
                    f"   🔍 Expected video path: {video_path}\n"
                    f"   🔍 Expected index path: {video_index_path}\n"
                    f"   📁 Location: task={task_path.name}, ep_idx={ep_idx}, frame_idx={frame_idx}\n"
                    f"   📋 Available paths (showing first 10): {available_paths}\n"
                    f"   💡 Set use_compressed_video: false in config if using normal image format"
                )
            
            return self._get_frame_from_compressed_video(
                task_path, ep_idx, frame_idx, 
                images_buffer[video_path],
                images_buffer[video_index_path],
                h5_path
            )

        try:
            image_data = images_buffer[h5_path][frame_idx]
        except KeyError as e:
            available_paths = list(images_buffer.keys())[:10]
            total_paths = len(images_buffer.keys())
            raise KeyError(
                f"❌ H5 path not found in images buffer.\n"
                f"   🔍 Requested path: {h5_path}\n"
                f"   📁 Location: task={task_path.name}, ep_idx={ep_idx}, frame_idx={frame_idx}\n"
                f"   📋 Available paths (showing first 10 of {total_paths}): {available_paths}\n"
                f"   💡 Check if h5_path is correct in your config"
            ) from e
        except IndexError as e:
            try:
                max_frames = len(images_buffer[h5_path])
            except Exception:
                max_frames = "unknown"
            raise IndexError(
                f"❌ Frame index out of range.\n"
                f"   🎯 Requested frame: {frame_idx}\n"
                f"   📁 Location: task={task_path.name}, ep_idx={ep_idx}\n"
                f"   🔍 H5 path: {h5_path}\n"
                f"   📐 Available frames: 0 to {max_frames-1 if isinstance(max_frames, int) else max_frames}\n"
                f"   💡 Check if frame index is within valid range"
            ) from e

        if self._image_is_iobytes:
            try:
                img = Image.open(io.BytesIO(image_data))
                return np.array(img)
            except Exception as e:
                self._image_is_iobytes = False
                # Log warning but continue with raw data
                if self.logger:
                    self.logger.warning(
                        f"⚠️ Failed to decode image as bytes, falling back to raw data.\n"
                        f"   📁 Location: task={task_path.name}, ep_idx={ep_idx}, frame_idx={frame_idx}\n"
                        f"   🔍 H5 path: {h5_path}\n"
                        f"   Error: {e!s}"
                    )
        return image_data

    def _get_frame_from_compressed_video(
        self,
        task_path: Path,
        ep_idx: int,
        frame_idx: int,
        video_data: np.ndarray,
        video_index: np.ndarray,
        h5_path: str,
    ) -> np.ndarray:
        """
        从压缩视频数据中提取指定帧
        
        Args:
            task_path: 任务路径
            ep_idx: episode 索引
            frame_idx: 帧索引
            video_data: 压缩的视频数据（void 类型的 numpy 数组）
            video_index: 视频索引数组，表示帧到字节的映射
            h5_path: H5 路径（用于日志）
        
        Returns:
            解码后的图像数组 (H, W, C)
        """
        try:
            # 将 void 类型转换为字节
            if isinstance(video_data, np.void) or (isinstance(video_data, np.ndarray) and video_data.dtype.kind == 'V'):
                video_bytes = np.array(video_data).tobytes()
            else:
                video_bytes = bytes(video_data)
            
            # 创建临时文件保存视频
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
                tmp_path = tmp_file.name
                tmp_file.write(video_bytes)
            
            try:
                # 使用 OpenCV 读取视频
                cap = cv2.VideoCapture(tmp_path)
                if not cap.isOpened():
                    raise RuntimeError(f"Failed to open video file: {tmp_path}")
                
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                # 获取总的机械臂数据帧数（从第一个非空的 state 数据推断）
                # 假设所有 state 数据帧数相同
                # TODO: 可以从配置或 H5 文件的其他字段获取更准确的值
                total_arm_frames = 327  # 硬编码，后续可以改进
                
                # 计算视频帧索引（使用最近邻插值）
                # 将机械臂帧映射到视频帧
                if total_frames > 0:
                    video_frame_idx = min(int(frame_idx * total_frames / total_arm_frames), total_frames - 1)
                else:
                    video_frame_idx = 0
                
                # 跳转到指定帧
                cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame_idx)
                ret, frame = cap.read()
                cap.release()
                
                if not ret:
                    raise RuntimeError(f"Failed to read frame {video_frame_idx} from video (total: {total_frames})")
                
                # OpenCV 读取的是 BGR 格式，转换为 RGB
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
            finally:
                # 清理临时文件
                import os
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                    
        except Exception as e:
            if self.logger:
                self.logger.error(
                    f"❌ Failed to decode compressed video frame.\n"
                    f"   📁 Location: task={task_path.name}, ep_idx={ep_idx}, frame_idx={frame_idx}\n"
                    f"   🔍 H5 path: {h5_path}\n"
                    f"   Error: {e!s}"
                )
            raise

    # @override
    def _get_frame_sub_states(
        self,
        task_path: Path,
        ep_idx: int,
        frame_idx: int,
        args_dict: dict,
        sub_states_buffer: any = None,
    ) -> np.ndarray:
        # Validate required keys
        required_keys = ["h5_path", "range_from", "range_to"]
        missing_keys = [key for key in required_keys if key not in args_dict]
        
        if missing_keys:
            available_keys = list(args_dict.keys())
            raise KeyError(
                f"❌ Missing required keys in args_dict for sub_states.\n"
                f"   ❌ Missing: {missing_keys}\n"
                f"   📋 Available: {available_keys}\n"
                f"   📁 Location: task={task_path.name}, ep_idx={ep_idx}, frame_idx={frame_idx}\n"
                f"   💡 args_dict must contain: h5_path, range_from, range_to\n"
                f"      Example: {{h5_path: '/observations/qpos', range_from: 0, range_to: 7}}"
            )

        h5_path = args_dict["h5_path"]
        from_idx = args_dict["range_from"]
        to_idx = args_dict["range_to"]

        try:
            frame_data = sub_states_buffer[h5_path][frame_idx]
        except KeyError as e:
            available_paths = list(sub_states_buffer.keys())[:10]
            total_paths = len(sub_states_buffer.keys())
            raise KeyError(
                f"❌ H5 path not found in sub_states buffer.\n"
                f"   🔍 Requested path: {h5_path}\n"
                f"   📁 Location: task={task_path.name}, ep_idx={ep_idx}, frame_idx={frame_idx}\n"
                f"   📋 Available paths (showing first 10 of {total_paths}): {available_paths}\n"
                f"   💡 Check if h5_path is correct in your config"
            ) from e
        except IndexError as e:
            try:
                max_frames = len(sub_states_buffer[h5_path])
            except Exception:
                max_frames = "unknown"
            raise IndexError(
                f"❌ Frame index out of range.\n"
                f"   🎯 Requested frame: {frame_idx}\n"
                f"   📁 Location: task={task_path.name}, ep_idx={ep_idx}\n"
                f"   🔍 H5 path: {h5_path}\n"
                f"   📐 Available frames: 0 to {max_frames-1 if isinstance(max_frames, int) else max_frames}\n"
                f"   💡 Check if frame index is within valid range"
            ) from e

        # Check if frame_data is a scalar (0-dimensional)
        if not isinstance(frame_data, np.ndarray):
            frame_data = np.array(frame_data)
        
        if frame_data.ndim == 0:
            # Scalar value - cannot slice
            if from_idx == 0 and to_idx == 1:
                # Special case: extracting a single scalar value
                return np.array([frame_data.item()])
            
            raise ValueError(
                f"❌ Cannot slice scalar data.\n"
                f"   🔢 Requested range: [{from_idx}:{to_idx}]\n"
                f"   📐 Data shape: {frame_data.shape} (scalar)\n"
                f"   📊 Data value: {frame_data}\n"
                f"   📁 Location: task={task_path.name}, ep_idx={ep_idx}, frame_idx={frame_idx}\n"
                f"   🔍 H5 path: {h5_path}\n"
                "   💡 Possible causes:\n"
                "      1. H5 data is stored as scalar instead of array\n"
                "      2. Wrong H5 path in config (pointing to wrong dataset)\n"
                "      3. Config expects array but data is single value\n"
                "   🔧 Solutions:\n"
                "      1. If data is single value, use range_from: 0, range_to: 1\n"
                "      2. Check H5 file structure to verify data dimensions\n"
                "      3. Update config to match actual H5 data structure"
            )
        
        # Validate slicing range for array data
        try:
            data_len = len(frame_data)
        except Exception:
            data_len = None

        if data_len is not None:
            if from_idx < 0 or to_idx > data_len or from_idx >= to_idx:
                raise ValueError(
                    f"❌ Invalid slicing range for sub_states.\n"
                    f"   🔢 Requested range: [{from_idx}:{to_idx}]\n"
                    f"   📐 Data length: {data_len}\n"
                    f"   📐 Data shape: {frame_data.shape}\n"
                    f"   📁 Location: task={task_path.name}, ep_idx={ep_idx}, frame_idx={frame_idx}\n"
                    f"   🔍 H5 path: {h5_path}\n"
                    f"   💡 Valid range should be: 0 <= range_from < range_to <= {data_len}"
                )

        result = frame_data[from_idx:to_idx]
        
        # 🔧 Ensure result is 1D array for concatenation
        # Special handling for unexpected 2D data (e.g., zhipingfang effector data)
        if result.ndim > 1:
            # If result is (1, N) where we expected (N,), squeeze the first dimension
            # Example: zhipingfang effector has shape (frames, 1000) but config expects (frames, 1)
            # After slicing [0:1], we get (1,) when accessing 1D frame_data, or (1, 1000) if frame_data is 2D
            if result.shape[0] == 1:
                # Squeeze only the first dimension: (1, N) → (N,)
                result = result.squeeze(axis=0)
                if self.logger:
                    self.logger.debug(
                        f"🔧 Squeezed first dimension: {h5_path} "
                        f"from shape {frame_data[from_idx:to_idx].shape} to {result.shape}"
                    )
            else:
                # Fallback: flatten entirely
                original_shape = result.shape
                result = result.flatten()
                if self.logger:
                    self.logger.warning(
                        f"⚠️  Flattening unexpected 2D result:\n"
                        f"   Path: {h5_path}\n"
                        f"   Original shape: {original_shape} → Flattened: {result.shape}"
                    )
        
        return result

    # @override
    def _get_frame_sub_actions(
        self,
        task_path: Path,
        ep_idx: int,
        frame_idx: int,
        args_dict: dict,
        sub_actions_buffer: any = None,
    ) -> np.ndarray:
        # Validate required keys
        required_keys = ["h5_path", "range_from", "range_to"]
        missing_keys = [key for key in required_keys if key not in args_dict]
        
        if missing_keys:
            available_keys = list(args_dict.keys())
            raise KeyError(
                f"❌ Missing required keys in args_dict for sub_actions.\n"
                f"   ❌ Missing: {missing_keys}\n"
                f"   📋 Available: {available_keys}\n"
                f"   📁 Location: task={task_path.name}, ep_idx={ep_idx}, frame_idx={frame_idx}\n"
                f"   💡 args_dict must contain: h5_path, range_from, range_to\n"
                f"      Example: {{h5_path: '/action', range_from: 0, range_to: 7}}"
            )

        h5_path = args_dict["h5_path"]
        from_idx = args_dict["range_from"]
        to_idx = args_dict["range_to"]

        try:
            frame_data = sub_actions_buffer[h5_path][frame_idx]
        except KeyError as e:
            available_paths = list(sub_actions_buffer.keys())[:10]
            total_paths = len(sub_actions_buffer.keys())
            raise KeyError(
                f"❌ H5 path not found in sub_actions buffer.\n"
                f"   🔍 Requested path: {h5_path}\n"
                f"   📁 Location: task={task_path.name}, ep_idx={ep_idx}, frame_idx={frame_idx}\n"
                f"   📋 Available paths (showing first 10 of {total_paths}): {available_paths}\n"
                f"   💡 Check if h5_path is correct in your config"
            ) from e
        except IndexError as e:
            try:
                max_frames = len(sub_actions_buffer[h5_path])
            except Exception:
                max_frames = "unknown"
            raise IndexError(
                f"❌ Frame index out of range.\n"
                f"   🎯 Requested frame: {frame_idx}\n"
                f"   📁 Location: task={task_path.name}, ep_idx={ep_idx}\n"
                f"   🔍 H5 path: {h5_path}\n"
                f"   📐 Available frames: 0 to {max_frames-1 if isinstance(max_frames, int) else max_frames}\n"
                f"   💡 Check if frame index is within valid range"
            ) from e

        # Validate slicing range
        try:
            data_len = len(frame_data)
        except Exception:
            data_len = None

        if data_len is not None:
            if from_idx < 0 or to_idx > data_len or from_idx >= to_idx:
                raise ValueError(
                    f"❌ Invalid slicing range for sub_actions.\n"
                    f"   🔢 Requested range: [{from_idx}:{to_idx}]\n"
                    f"   📐 Data length: {data_len}\n"
                    f"   📁 Location: task={task_path.name}, ep_idx={ep_idx}, frame_idx={frame_idx}\n"
                    f"   🔍 H5 path: {h5_path}\n"
                    f"   💡 Valid range should be: 0 <= range_from < range_to <= {data_len}"
                )

        result = frame_data[from_idx:to_idx]
        
        # 🔧 Ensure result is 1D array for concatenation
        # Special handling for unexpected 2D data
        if result.ndim > 1:
            if result.shape[0] == 1:
                # Squeeze only the first dimension: (1, N) → (N,)
                result = result.squeeze(axis=0)
                if self.logger:
                    self.logger.debug(
                        f"🔧 Squeezed first dimension for actions: {h5_path} "
                        f"from shape {frame_data[from_idx:to_idx].shape} to {result.shape}"
                    )
            else:
                # Fallback: flatten entirely
                original_shape = result.shape
                result = result.flatten()
                if self.logger:
                    self.logger.warning(
                        f"⚠️  Flattening unexpected 2D actions result:\n"
                        f"   Path: {h5_path}\n"
                        f"   Original shape: {original_shape} → Flattened: {result.shape}"
                    )
        
        return result

    # @override
    def _get_episode_frames_num(self, task_path: Path, ep_idx: int) -> int:
        args = self.converter_config[FEATURES_KEY][OBSERVATION_KEY][STATE_KEY][SUB_STATE_KEY][0][
            ARGS_KEY
        ]
        if "h5_path" not in args:
            available_keys = list(args.keys()) if args else []
            raise ValueError(
                f"❌ h5_path not specified in config.\n"
                f"   📁 Location: observation.state.sub_state[0].args\n"
                f"   📋 Available keys in args: {available_keys}\n"
                f"   💡 Config must specify h5_path to determine frame count\n"
                f"      Example: args: {{h5_path: '/observations/qpos', ...}}"
            )
        h5_path = args["h5_path"]

        h5_file_path = self.task_episode_h5file_paths[task_path][ep_idx]
        try:
            # 🚀 使用H5FileCache提升性能
            with self._h5_file_cache.open(h5_file_path) as h5_file:
                # 获取参考帧数
                reference_frame_count = h5_file[h5_path].shape[0]
                
                # ⭐ 新增：检查所有 sub_state 的帧数是否一致
                frame_count_issues = []
                all_sub_states = self.converter_config[FEATURES_KEY][OBSERVATION_KEY][STATE_KEY][SUB_STATE_KEY]
                
                for i, sub_state in enumerate(all_sub_states):
                    sub_state_args = sub_state.get(ARGS_KEY, {})
                    if "h5_path" not in sub_state_args:
                        continue
                    
                    check_h5_path = sub_state_args["h5_path"]
                    if check_h5_path not in h5_file:
                        continue  # 路径不存在，其他地方会报错
                    
                    dataset = h5_file[check_h5_path]
                    shape = dataset.shape
                    
                    # 跳过视频压缩数据（shape=()）
                    if 'video' in check_h5_path.lower() and shape == ():
                        continue
                    
                    # 跳过空数据集（某些arm未使用时shape[0]=0）
                    if len(shape) > 0 and shape[0] == 0:
                        continue
                    
                    # 检查帧数
                    if len(shape) > 0:
                        current_frame_count = shape[0]
                        if current_frame_count != reference_frame_count:
                            frame_count_issues.append(
                                f"    sub_state[{i}] {check_h5_path}: {current_frame_count} 帧"
                            )
                
                # 如果发现帧数不一致，自动移动文件到 error/ 并返回 -1（跳过标记）
                if frame_count_issues:
                    error_msg = (
                        f"❌ H5数据集帧数不一致（数据质量问题）\n"
                        f"   🗂️  文件: {h5_file_path.name}\n"
                        f"   📁 完整路径: {h5_file_path}\n"
                        f"   📍 任务: {task_path.name}\n"
                        f"   📍 Episode索引: {ep_idx}\n"
                        f"   \n"
                        f"   📊 参考帧数（来自 sub_state[0]）:\n"
                        f"    {h5_path}: {reference_frame_count} 帧\n"
                        f"   \n"
                        f"   ❌ 以下数据集帧数不一致:\n"
                        + "\n".join(frame_count_issues)
                    )
                    
                    # 尝试自动移动到 error/ 目录
                    try:
                        import shutil
                        error_dir = h5_file_path.parent / "error"
                        error_dir.mkdir(exist_ok=True)
                        
                        dest_path = error_dir / h5_file_path.name
                        if not dest_path.exists():
                            shutil.move(str(h5_file_path), str(dest_path))
                            
                            if self.logger:
                                self.logger.warning(
                                    f"📦 自动移动问题文件到 error/:\n"
                                    f"   {h5_file_path.name} -> {error_dir}/\n"
                                    f"   原因: 帧数不一致\n"
                                    + error_msg
                                )
                            
                            # 返回 -1 表示此 episode 应该被跳过
                            return -1
                        else:
                            if self.logger:
                                self.logger.warning(
                                    f"⚠️  问题文件已存在于 error/ 目录，跳过:\n"
                                    f"   {dest_path}\n"
                                    + error_msg
                                )
                            return -1
                    
                    except Exception as move_error:
                        # 移动失败，抛出原始错误
                        if self.logger:
                            self.logger.error(
                                f"❌ 自动移动文件失败: {move_error}\n"
                                + error_msg
                                + f"\n   \n"
                                f"   💡 请手动移动文件:\n"
                                f"      mkdir -p '{h5_file_path.parent}/error'\n"
                                f"      mv '{h5_file_path}' '{h5_file_path.parent}/error/'\n"
                            )
                        
                        raise ValueError(
                            error_msg
                            + f"\n   \n"
                            f"   💡 自动移动失败: {move_error}\n"
                            f"   🔧 请手动移动文件:\n"
                            f"      mkdir -p '{h5_file_path.parent}/error'\n"
                            f"      mv '{h5_file_path}' '{h5_file_path.parent}/error/'\n"
                        )
                
                return reference_frame_count
        except OSError as e:
            error_str = str(e)
            if "bad global heap collection signature" in error_str:
                raise ValueError(
                    f"❌ H5 file corruption detected.\n"
                    f"   🗂️  File: {h5_file_path.name}\n"
                    f"   📁 Location: task={task_path.name}, ep_idx={ep_idx}\n"
                    f"   ❌ Error: Corrupted global heap collection signature\n"
                    f"   💡 This H5 file is corrupted and must be regenerated.\n"
                    f"      Original error: {error_str}"
                ) from e
            raise ValueError(
                f"❌ Cannot read H5 file (OSError).\n"
                f"   🗂️  File: {h5_file_path.name}\n"
                f"   📁 Location: task={task_path.name}, ep_idx={ep_idx}\n"
                f"   🔍 H5 path: {h5_path}\n"
                f"   ❌ Error: {error_str}\n"
                f"   💡 Check if:\n"
                f"      1. File is not corrupted\n"
                f"      2. File is not being written to\n"
                f"      3. File permissions are correct"
            ) from e
        except Exception as e:
            raise ValueError(
                f"❌ Error reading frame count from H5 file.\n"
                f"   🗂️  File: {h5_file_path.name}\n"
                f"   📁 Location: task={task_path.name}, ep_idx={ep_idx}\n"
                f"   🔍 H5 path: {h5_path}\n"
                f"   ❌ Error: {e!s}\n"
                f"   💡 Check if:\n"
                f"      1. h5_path exists in file\n"
                f"      2. Dataset has valid shape\n"
                f"      3. File format is correct"
            ) from e

    # @override
    def _get_task_episodes_num(self, task_path: Path) -> int:
        return len(self.task_episode_h5file_paths[task_path])

    # @override
    def _prepare_episode_images_buffer(self, task_path: Path, ep_idx: int) -> any:
        return self._get_episode_h5_data(task_path, ep_idx)

    # @override
    def _prepare_episode_states_buffer(self, task_path: Path, ep_idx: int) -> any:
        return self._get_episode_h5_data(task_path, ep_idx)

    # @override
    def _prepare_episode_actions_buffer(self, task_path: Path, ep_idx: int) -> any:
        return self._get_episode_h5_data(task_path, ep_idx)

    @cached_property
    def task_episode_h5file_paths(self) -> dict[Path, list[Path]]:
        """获取每个task的H5 episode文件路径
        
        策略：
        1. 使用rglob递归搜索（支持任意深度嵌套）
        2. 排除特定子目录（record/, calibration/等）
        3. 宁可多找也不能漏
        """
        task_episode_paths = {}
        
        # 需要排除的目录（这些是原始数据/配置/日志目录，不是episode）
        skip_dirs = {
            'record', 'calibration', 'config', 'parameters', 
            'logs', 'camera', 'meta_info', 'others', 'error',
            '@eaDir', '__pycache__', '.git', '.idea', '.vscode'
        }
        
        for path in self.path_task_dict.keys():
            if not path.exists():
                continue
            
            # 递归查找所有.h5、.hdf5和以h5/hdf5结尾的文件（容错处理）
            h5_files = []
            h5_files.extend(path.rglob("*.h5"))
            h5_files.extend(path.rglob("*.hdf5"))
            # 额外查找命名错误的文件（如 episode_139hdf5）
            for file in path.rglob("*"):
                if file.is_file() and (file.name.endswith("hdf5") or file.name.endswith("h5")):
                    # 避免重复添加
                    if file not in h5_files:
                        h5_files.append(file)
            
            # 过滤：排除特定目录下的文件
            filtered_files = []
            for h5_file in h5_files:
                # 检查文件路径中是否包含需要排除的目录
                relative_path = h5_file.relative_to(path)
                path_parts = set(relative_path.parts[:-1])  # 不包括文件名
                
                # 如果路径中包含任何需要排除的目录，则跳过
                if path_parts & skip_dirs:
                    continue
                
                # 排除隐藏文件和@开头的目录
                if any(part.startswith('.') or part.startswith('@') for part in relative_path.parts):
                    continue
                
                # 跳过损坏的H5文件（在预验证阶段标记的）
                if h5_file in self._invalid_h5_files:
                    continue
                
                filtered_files.append(h5_file)
            
            task_episode_paths[path] = natsorted(filtered_files)
        
        return task_episode_paths

    def _get_episode_h5_data(self, task_path: Path, ep_idx: int) -> any:
        should_load = self.h5_buffer.task_path != task_path or self.h5_buffer.ep_idx != ep_idx
        if not should_load:
            return self.h5_buffer.h5_data
        self.h5_buffer.h5_data = {}

        h5_file_path = self.task_episode_h5file_paths[task_path][ep_idx]
        
        # 收集所有配置的相机路径及其 use_compressed_video 设置
        camera_configs = {}
        for image_config in self.converter_config.get(FEATURES_KEY, {}).get(OBSERVATION_KEY, {}).get(IMAGE_KEY, []):
            if ARGS_KEY in image_config and "h5_path" in image_config[ARGS_KEY]:
                h5_path = image_config[ARGS_KEY]["h5_path"]
                use_compressed = image_config[ARGS_KEY].get("use_compressed_video", False)
                camera_configs[h5_path] = use_compressed

        def _get_dataset(name: str, obj: any) -> None:
            if isinstance(obj, h5py.Dataset):
                try:
                    # 检查这个路径是否是配置中的图像路径
                    is_image_path = name in camera_configs
                    
                    if is_image_path:
                        use_compressed = camera_configs[name]
                        if use_compressed:
                            # 如果配置使用压缩视频，跳过加载 images，改为加载 video 和 video_index
                            # 跳过 images 路径，不加载
                            if self.logger:
                                self.logger.debug(f"Skipping images path {name}, will load video data instead")
                            return
                        # 否则正常加载 images
                    
                    self.h5_buffer.h5_data[name] = obj[()]
                except Exception as e:
                    # 提供详细的H5文件错误诊断信息
                    error_msg = (
                        f"H5 Dataset Read Error: Failed to read dataset '{name}' from H5 file '{h5_file_path}'. "
                        f"Task: {task_path}, Episode: {ep_idx}. "
                        f"Dataset shape: {getattr(obj, 'shape', 'Unknown')}, "
                        f"Dataset dtype: {getattr(obj, 'dtype', 'Unknown')}, "
                        f"Dataset size: {getattr(obj, 'size', 'Unknown')} bytes. "
                        f"Original error: {type(e).__name__}: {e}. "
                        f"This might indicate file corruption or incompatible H5 format."
                    )
                    
                    if self.logger:
                        self.logger.error(f"H5 File Error: {error_msg}")
                        self.logger.error("WARNING: If you see a simplified OSError, check the full error above!")
                    
                    raise ValueError(error_msg) from e

        try:
            # 🚀 使用H5FileCache提升性能
            with self._h5_file_cache.open(h5_file_path) as h5_file:
                h5_file.visititems(_get_dataset)
                
                # 对于配置了 use_compressed_video 的相机，额外加载 video 和 video_index
                for h5_path, use_compressed in camera_configs.items():
                    if use_compressed:
                        video_path = h5_path.replace('/images', '/video')
                        video_index_path = h5_path.replace('/images', '/video_index')
                        
                        if video_path in h5_file:
                            self.h5_buffer.h5_data[video_path] = h5_file[video_path][()]
                            if self.logger:
                                self.logger.debug(f"Loaded compressed video: {video_path}")
                        else:
                            if self.logger:
                                self.logger.warning(f"Video path not found: {video_path}")
                        
                        if video_index_path in h5_file:
                            self.h5_buffer.h5_data[video_index_path] = h5_file[video_index_path][()]
                            if self.logger:
                                self.logger.debug(f"Loaded video index: {video_index_path}")
                        else:
                            if self.logger:
                                self.logger.warning(f"Video index path not found: {video_index_path}")
                
                self.h5_buffer.task_path = task_path
                self.h5_buffer.ep_idx = ep_idx
        except OSError as e:
            # 特定处理 H5 文件损坏错误
            error_str = str(e)
            if "bad global heap collection signature" in error_str:
                error_msg = (
                    f"H5 File Corruption Error: H5 file has corrupted global heap collection signature. "
                    f"File path: {h5_file_path}, "
                    f"Task: {task_path.name}, "
                    f"Episode: {ep_idx}, "
                    f"File size: {h5_file_path.stat().st_size if h5_file_path.exists() else 'N/A'} bytes. "
                    f"This indicates severe file corruption. Original error: {error_str}. "
                    f"Please regenerate or re-download this H5 file."
                )
            elif "unable to open file" in error_str.lower():
                error_msg = (
                    f"H5 File Access Error: Cannot open H5 file. "
                    f"File path: {h5_file_path}, "
                    f"Task: {task_path.name}, "
                    f"Episode: {ep_idx}, "
                    f"File exists: {h5_file_path.exists()}, "
                    f"File size: {h5_file_path.stat().st_size if h5_file_path.exists() else 'N/A'} bytes. "
                    f"Original error: {error_str}"
                )
            else:
                error_msg = (
                    f"H5 File OSError: H5 file operation failed. "
                    f"File path: {h5_file_path}, "
                    f"Task: {task_path.name}, "
                    f"Episode: {ep_idx}, "
                    f"Original error: {error_str}"
                )
            
            if self.logger:
                self.logger.error(f"H5 File Corruption Detected: {error_msg}")
            
            raise ValueError(error_msg) from e
        except Exception as e:
            # 捕获文件级别的其他错误
            if not isinstance(e, ValueError):  # 避免重复包装我们自己的ValueError
                error_msg = (
                    f"H5 File Access Error: Failed to access H5 file '{h5_file_path}'. "
                    f"Task: {task_path}, Episode: {ep_idx}. "
                    f"Original error: {type(e).__name__}: {e}. "
                    f"Please check if the file exists and is not corrupted."
                )
                
                if self.logger:
                    self.logger.error(f"H5 File Access Error: {error_msg}")
                
                raise ValueError(error_msg) from e
            
            # 重新抛出我们自己的ValueError
            raise

        return self.h5_buffer.h5_data
    
    def _get_episode_source_files(self, task_path: Path, ep_idx: int) -> dict:
        """获取 H5 episode 的源文件信息
        
        Args:
            task_path: 任务路径
            ep_idx: episode 索引
        
        Returns:
            dict: 包含源文件信息的字典，包括 h5_file 和 absolute_path
        """
        h5_files = self.task_episode_h5file_paths.get(task_path, [])
        if ep_idx < len(h5_files):
            h5_file = h5_files[ep_idx]
            return {
                "format": "H5",
                "h5_file": str(h5_file.relative_to(self.dataset_path)),
                "absolute_path": str(h5_file.absolute()),
            }
        return {}
    
    def load_camera_params(self, h5_scalar_data: bytes | np.ndarray) -> dict:
        """
        从HDF5标量JSON字节数据加载相机参数
        兼容：标量bytes / 标量np.void / 数组包裹的bytes
        Args:
            h5_scalar_data: H5中存储的JSON字节数据（如infos/camera_params/chest.json）
        Returns:
            包含内参、外参、分辨率的参数字典
        """
        # ===================== 正确提取标量 bytes =====================
        if isinstance(h5_scalar_data, np.ndarray):
            if h5_scalar_data.shape == ():
                data_bytes = h5_scalar_data.item()
            else:
                data_bytes = h5_scalar_data.tobytes()
        else:
            data_bytes = h5_scalar_data

        if not isinstance(data_bytes, bytes):
            raise ValueError(f"相机参数数据类型错误，期望bytes，得到 {type(data_bytes)}")

        # 解析 JSON
        try:
            params = json.loads(data_bytes.decode('utf-8'))
        except json.JSONDecodeError as e:
            raise ValueError(f"相机参数JSON解析失败: {str(e)[:100]}") from e

        # ===================== 🔥 核心修复：字段名匹配（intrinsics → intrinsic） =====================
        def safe_array(val, default=[]):
            return np.array(val, dtype=np.float32) if val is not None else np.array(default, dtype=np.float32)

        # 1. 内参：从 fx, fy, cx, cy 构造 3x3 矩阵
        intr = params.get('intrinsics', {})
        fx, fy = intr.get('fx', 0.0), intr.get('fy', 0.0)
        cx, cy = intr.get('cx', 0.0), intr.get('cy', 0.0)
        intrinsic_mat = np.array([
            [fx,  0,  cx],
            [ 0, fy,  cy],
            [ 0,  0,  1.0]
        ], dtype=np.float32)

        # 2. 畸变系数
        dist_coeffs = intr.get('distortion_coeffs', [])

        # 3. 外参：旋转矩阵(9) + 平移(3) → 4x4  extrinsic
        extr = params.get('extrinsics', {})
        rot_mat = np.array(extr.get('rotation_matrix', np.eye(3).flatten()), dtype=np.float32).reshape(3,3)
        trans_vec = np.array(extr.get('translation_vector', [0,0,0]), dtype=np.float32).reshape(3,1)
        extrinsic_mat = np.eye(4, dtype=np.float32)
        extrinsic_mat[:3,:3] = rot_mat
        extrinsic_mat[:3, 3] = trans_vec.flatten()

        return {
            "intrinsic": intrinsic_mat,
            "distortion": safe_array(dist_coeffs),
            "extrinsic": extrinsic_mat,
            "resolution": [
                params.get('resolution', {}).get('width', 640),
                params.get('resolution', {}).get('height', 480)
            ]
        }



    def save_camera_params_to_json(self, output_dir: Path | None = None) -> None:
        """
        【正式修复版】
        直接读取配置中的 parameters_path，不猜、不漏、不崩溃
        完全匹配你的数据集结构 + 你的 YAML 配置
        """
        if output_dir is None:
            output_dir = Path(self.output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        json_save_path = output_dir / "camera_params.json"

        camera_params_collection = {}
        processed_cameras = set()

        for task_path in self.path_task_dict.keys():
            h5_files = self.task_episode_h5file_paths.get(task_path, [])
            if not h5_files:
                continue

            sample_h5_file = h5_files[0]
            try:
                with self._h5_file_cache.open(sample_h5_file) as h5_file:
                    image_configs = (
                        self.converter_config
                        .get(FEATURES_KEY, {})
                        .get(OBSERVATION_KEY, {})
                        .get(IMAGE_KEY, [])
                    )

                    for img_cfg in image_configs:
                        cam_name = img_cfg.get("cam_name") or img_cfg.get("name")
                        if not cam_name or cam_name in processed_cameras:
                            continue

                        args = img_cfg.get("args", {})
                        # ===================== 真正读取你配置里的 parameters_path =====================
                        param_path = args.get("parameters_path")
                        if not param_path:
                            if self.logger:
                                self.logger.warning(f"⚠️ 相机 {cam_name} 未配置 parameters_path，跳过")
                            processed_cameras.add(cam_name)
                            continue

                        if param_path not in h5_file:
                            if self.logger:
                                self.logger.warning(f"⚠️ 相机 {cam_name} 参数不存在：{param_path}")
                            processed_cameras.add(cam_name)
                            continue

                        # 读取并解析
                        data = h5_file[param_path][()]
                        cam_params = self.load_camera_params(data)

                        serializable = {
                            "camera_name": cam_name,
                            "intrinsic": cam_params["intrinsic"].tolist(),
                            "distortion": cam_params["distortion"].tolist(),
                            "extrinsic": cam_params["extrinsic"].tolist(),
                            "resolution": self._get_camera_resolution(cam_name, img_cfg),
                            "source_path": param_path,
                            "file": str(sample_h5_file.name)
                        }
                        camera_params_collection[cam_name] = serializable
                        processed_cameras.add(cam_name)

            except Exception as e:
                if self.logger:
                    self.logger.error(f"❌ 读取相机参数失败：{str(e)[:100]}")
                continue

        if camera_params_collection:
            try:
                final_data = {
                    "dataset": self.repo_id,
                    "device_model": self.device_model,
                    "total_cameras": len(camera_params_collection),
                    "camera_parameters": camera_params_collection
                }
                with open(json_save_path, "w", encoding="utf-8") as f:
                    json.dump(final_data, f, indent=2, ensure_ascii=False)
                if self.logger:
                    self.logger.info(f"✅ 相机参数已保存：{json_save_path}")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"❌ 保存相机参数失败：{e}")
        else:
            if self.logger:
                self.logger.info("ℹ️ 未提取到相机参数")



    def _get_camera_resolution(self, cam_name: str, image_config: dict) -> list:
        """从图像配置中获取相机分辨率 [宽度, 高度]"""
        try:
            h, w, _ = image_config["shape"]
            return [w, h]
        except:
            return [None, None]

