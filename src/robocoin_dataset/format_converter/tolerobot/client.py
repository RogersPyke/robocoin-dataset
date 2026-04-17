import logging
from pathlib import Path

from tqdm import tqdm

from robocoin_dataset.distribution_computation.constant import (
    DATASET_PATH,
    DEVICE_MODEL,
)
from robocoin_dataset.distribution_computation.task_client import TaskClient
from robocoin_dataset.format_converter.tolerobot.constant import (
    AUTO_REENCODE,
    CONVERTER_CLASS_NAME,
    CONVERTER_CONFIG,
    CONVERTER_LOG_DIR,
    CONVERTER_LOG_NAME,
    CONVERTER_MODULE_PATH,
    IMAGE_WRITER_PROCESSES,
    IMAGE_WRITER_THREADS,
    IS_TEST,
    LEFORMAT_PATH,
    REPO_ID,
    VIDEO_BACKEND,
)
from robocoin_dataset.format_converter.tolerobot.lerobot_format_converter import (
    LerobotFormatConverter,
    LerobotFormatConverterFactory,
)
from robocoin_dataset.utils.logger import setup_logger


class LeFormatConverterTaskClient(TaskClient):
    def __init__(
        self,
        server_uri: str = "ws://localhost:8765",
        heartbeat_interval: float = 10.0,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(
            server_uri=server_uri,
            heartbeat_interval=heartbeat_interval,
            logger=logger,
        )

    def get_task_category(self) -> str:
        return "LeFormatConvert"

    def generate_task_request_desc(self) -> dict:
        """客户端可自定义任务请求参数"""
        return {}

    def _sync_process_task(self, task_content: dict) -> dict:
        # 🔧 修复：提前声明converter，避免finally块中的NameError
        converter = None
        
        try:
            # 去除路径字符串中的前导和尾随空格
            dataset_path_str = task_content.get(DATASET_PATH)
            if isinstance(dataset_path_str, str):
                dataset_path_str = dataset_path_str.strip()
            dataset_path = Path(dataset_path_str)
            
            device_model = task_content.get(DEVICE_MODEL)
            
            output_path_str = task_content.get(LEFORMAT_PATH)
            if isinstance(output_path_str, str):
                output_path_str = output_path_str.strip()
            output_path = Path(output_path_str)
            converter_config = task_content.get(CONVERTER_CONFIG)
            repo_id = task_content.get(REPO_ID)
            device_model = task_content.get(DEVICE_MODEL, None)
            module_path = task_content.get(CONVERTER_MODULE_PATH)
            class_name = task_content.get(CONVERTER_CLASS_NAME)
            video_backend = task_content.get(VIDEO_BACKEND)
            image_writer_processes = task_content.get(IMAGE_WRITER_PROCESSES, 4)
            image_writer_threads = task_content.get(IMAGE_WRITER_THREADS, 4)
            converter_log_dir = task_content.get(CONVERTER_LOG_DIR)
            converter_log_name = task_content.get(CONVERTER_LOG_NAME)
            is_test = task_content.get(IS_TEST, False)
            auto_reencode = task_content.get(AUTO_REENCODE, False)

            logger = setup_logger(
                converter_log_name,
                converter_log_dir,
                logging.INFO,
            )

            # 🔧 修复：将converter创建放到try-except中，确保初始化失败时也能返回有意义的结果
            try:
                converter: LerobotFormatConverter = LerobotFormatConverterFactory.create_converter(
                    dataset_path=dataset_path,
                    device_model=device_model,
                    output_path=output_path,
                    converter_config=converter_config,
                    converter_module_path=module_path,
                    converter_class_name=class_name,
                    repo_id=repo_id,
                    video_backend=video_backend,
                    image_writer_processes=image_writer_processes,
                    image_writer_threads=image_writer_threads,
                    logger=logger,
                    auto_reencode=auto_reencode,
                )
            except Exception as e:
                # 🆕 Converter创建失败（初始化阶段错误）
                logger.error(f"❌ Failed to create converter for {dataset_path}")
                logger.error(f"   Error: {type(e).__name__}: {e}")
                logger.error(f"   This usually means:")
                logger.error(f"   1. Dataset path is incorrect or inaccessible")
                logger.error(f"   2. Dataset structure doesn't match expected format")
                logger.error(f"   3. Required files (metadata, H5, etc.) are missing")
                
                # 抛出异常让外层捕获
                raise RuntimeError(
                    f"❌ Converter initialization failed for {dataset_path}.\n"
                    f"   Error: {type(e).__name__}: {e}\n"
                    f"   💡 Check:\n"
                    f"      1. Dataset path exists and is accessible\n"
                    f"      2. Dataset structure matches expected format\n"
                    f"      3. All required files are present"
                ) from e

            try:
                self.logger.info(f"converter_log_dir: {converter_log_dir}")
                total_episodes = converter.get_episodes_num()
                
                # 🆕 Test模式下，预估只处理少量episodes（避免tqdm进度条显示混乱）
                if is_test:
                    # Test模式：最多2个tasks × 1个episode = 2 episodes
                    estimated_test_episodes = min(2, total_episodes)
                    tqdm_total = estimated_test_episodes
                    mode_str = "TEST"
                else:
                    tqdm_total = total_episodes
                    mode_str = "FORMAL"
                
                # 🆕 明确显示转换模式
                self.logger.info(f"{'='*60}")
                self.logger.info(f"🚀 Starting conversion in {mode_str} MODE")
                self.logger.info(f"Dataset: {dataset_path}")
                self.logger.info(f"Total episodes in dataset: {total_episodes}")
                if is_test:
                    self.logger.info(f"Estimated episodes to process (test): {estimated_test_episodes}")
                self.logger.info(f"{'='*60}")
                
                converted_count = 0
                for task_content, task_ep_idx, ep_idx in tqdm(
                    converter.convert(is_test),
                    total=tqdm_total,
                    desc=f"Converting Dataset ({mode_str})",
                    unit="episode",
                ):
                    self.logger.info(
                        f"Converted episode {task_ep_idx} of task {task_content}, total ep_idx is:{ep_idx}"
                    )
                    converted_count += 1
                
                # Log conversion statistics
                skipped_count = total_episodes - converted_count
                if skipped_count > 0:
                    self.logger.warning(
                        f"📊 Conversion completed with some episodes skipped:\n"
                        f"   Total episodes found: {total_episodes}\n"
                        f"   Successfully converted: {converted_count}\n"
                        f"   Skipped (data quality issues): {skipped_count}\n"
                        f"   ✅ Check error/ directories for skipped files"
                    )
                else:
                    self.logger.info(
                        f"✅ Conversion completed successfully: {converted_count}/{total_episodes} episodes"
                    )
                
                # Save episode source mapping after conversion completes
                if not is_test:
                    converter.save_episode_source_mapping()
                    converter.save_original_data_paths()  # 🆕 保存绝对路径mapping
                
                # 🆕 返回转换统计信息给Server
                return {
                    "total_episodes": total_episodes,
                    "converted_episodes": converted_count,
                    "skipped_episodes": skipped_count,
                    "is_test": is_test,
                }
            finally:
                # 🔥 确保清理资源，防止semaphore泄漏
                # 🔧 修复：检查converter是否已创建
                if converter is not None:
                    try:
                        if hasattr(converter, 'lerobot_dataset') and converter.lerobot_dataset is not None:
                            converter.lerobot_dataset.stop_image_writer()
                            self.logger.debug("✅ 已清理image writer资源")
                    except Exception as e:
                        self.logger.warning(f"⚠️  清理converter资源时出错: {e}")

        except Exception as e:
            raise RuntimeError(f"convert dataset {dataset_path} failed") from e
