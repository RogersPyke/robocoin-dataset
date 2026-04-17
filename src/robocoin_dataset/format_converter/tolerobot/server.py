import logging
from datetime import datetime
from pathlib import Path
import re

import yaml
from sqlalchemy import and_, or_

# 导入外部的夹爪归一化类
from robocoin_dataset.format_converter.tolerobot.gripper_normalization import GripperOpenNormalizer
from robocoin_dataset.format_converter.tolerobot.data_range_check import LerobotDatasetValidator
from robocoin_dataset.constant import ROBOCOIN_PLATFORM
from robocoin_dataset.database.database import DatasetDatabase
from robocoin_dataset.database.models import (
    DatasetDB,
    TaskStatus,
)
from robocoin_dataset.distribution_computation.constant import (
    DATASET_NAME,
    DATASET_PATH,
    DATASET_UUID,
    DEVICE_MODEL,
    ERR_MSG,
    TASK_RESULT_CONTENT,
    TASK_RESULT_STATUS,
    TASK_SUCCESS,
)
from robocoin_dataset.distribution_computation.task_server import TaskServer
from robocoin_dataset.format_converter.tolerobot.constant import (
    AUTO_REENCODE,
    CONVERTER_CLASS_NAME,
    CONVERTER_CONFIG,
    CONVERTER_LOG_DIR,
    CONVERTER_LOG_NAME,
    CONVERTER_MODULE_PATH,
    DEFAULT_DEVICE_MODEL_VERSION,
    DEVICE_MODEL_VERSION_KEY,
    IMAGE_WRITER_PROCESSES,
    IMAGE_WRITER_THREADS,
    IS_TEST,
    LEFORMAT_PATH,
    REPO_ID,
    VIDEO_BACKEND,
)


class LeFormatConverterTaskServer(TaskServer):
    def __init__(
        self,
        db_file: Path,
        convert_root_path: Path,
        converter_factory_config_path: Path,
        host: str = "0.0.0.0",
        port: int = 8765,
        heartbeat_interval: float = 30.0,
        timeout: float = 15.0,
        specific_device_model: str | None = None,
        logger: logging.Logger | None = None,
        video_backend: str = "pyav",
        image_writer_processes: int = 4,
        image_writer_threads: int = 4,
        is_test: bool = False,
        auto_reencode: bool = False,
        enable_gripper_normalization: bool = True,
        enable_dataset_validation: bool = True
    ) -> None:
        super().__init__(
            logger=logger,
            host=host,
            port=port,
            heartbeat_interval=heartbeat_interval,
            timeout=timeout,
        )
        # 异常捕获：初始化数据库路径
        try:
            db_file_path = Path(db_file).expanduser().absolute()
            self.db = DatasetDatabase(db_file=db_file_path)
        except Exception as e:
            self.logger.critical(f"初始化数据库失败，服务无法启动: {e}", exc_info=True)
            raise

        if not convert_root_path:
            raise ValueError("convert_root must be provided")
        self.convert_root_path = convert_root_path
        self.specific_device_model = specific_device_model
        self.factory_config_dir = Path(converter_factory_config_path).parent
        self.video_backend = video_backend
        self.image_writer_processes = image_writer_processes
        self.image_writer_threads = image_writer_threads

        self.is_test = is_test
        self.auto_reencode = auto_reencode
        self.enable_gripper_normalization = enable_gripper_normalization
        self.enable_dataset_validation = enable_dataset_validation

        # 异常捕获：加载工厂配置
        try:
            with open(converter_factory_config_path) as f:
                self.converter_factory_config = yaml.safe_load(f)
        except Exception as e:
            self.logger.critical(f"加载转换器工厂配置失败: {e}", exc_info=True)
            raise

    # 时间工具函数
    def _get_formatted_time(self) -> tuple[str, float]:
        try:
            now = datetime.now()
            formatted_str = now.strftime("%Y-%m-%d %H:%M")
            timestamp = datetime.timestamp(now)
            return formatted_str, timestamp
        except Exception as e:
            self.logger.error(f"生成时间格式失败: {e}", exc_info=True)
            return "", 0.0

    # 时间解析工具函数
    def _parse_time_str_to_datetime(self, time_str: str) -> datetime:
        try:
            return datetime.strptime(time_str, "%Y-%m-%d %H:%M")
        except ValueError:
            try:
                pattern = r"(\d+)年(\d+)月(\d+)日(\d+)点(\d+)分"
                match = re.match(pattern, time_str)
                if not match:
                    raise ValueError(f"无法解析时间字符串：{time_str}")
                year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
                hour, minute = int(match.group(4)), int(match.group(5))
                return datetime(year, month, day, hour, minute)
            except Exception as e:
                self.logger.error(f"解析时间字符串失败: {e}", exc_info=True)
                return datetime.now()

    # 配置获取工具函数（内部调用，异常向上抛出，由顶级逻辑捕获）
    def _get_converter_class_name(self, device_model: str) -> str:
        if device_model not in self.converter_factory_config:
            raise ValueError(f"Device model {device_model} not found in factory config.")
        return self.converter_factory_config[device_model]["class"]

    def _get_converter_module_path(self, device_model: str) -> str:
        if device_model not in self.converter_factory_config:
            raise ValueError(f"Device model {device_model} not found in factory config.")
        return self.converter_factory_config[device_model]["module"]

    def _get_converter_config(self, device_model: str) -> dict:
        if device_model not in self.converter_factory_config:
            raise ValueError(f"Device model {device_model} not found in factory config.")
        convertor_config_path = (
            self.factory_config_dir
            / self.converter_factory_config[device_model]["converter_config_path"]
        )
        with open(convertor_config_path) as f:
            return yaml.safe_load(f)

    def _get_converter_module_class_config(
        self, device_model: str, device_model_version: str = ""
    ) -> tuple[str, str, dict]:
        if device_model not in self.converter_factory_config:
            raise ValueError(f"Device model {device_model} not found in factory config.")

        device_model_configs_list = self.converter_factory_config[device_model]
        if device_model_configs_list is None or not isinstance(device_model_configs_list, list):
            raise ValueError(f"Device model {device_model} config invalid.")

        converter_module_path = None
        for device_model_config in device_model_configs_list:
            if not device_model_version:
                if device_model_config[DEVICE_MODEL_VERSION_KEY] == DEFAULT_DEVICE_MODEL_VERSION:
                    converter_module_path = device_model_config["module"]
                    converter_class_name = device_model_config["class"]
                    converter_config_file_path = self.factory_config_dir / device_model_config["converter_config_path"]
                break
            if device_model_config[DEVICE_MODEL_VERSION_KEY] == device_model_version:
                converter_module_path = device_model_config["module"]
                converter_class_name = device_model_config["class"]
                converter_config_file_path = self.factory_config_dir / device_model_config["converter_config_path"]
                break

        if converter_module_path is None:
            raise ValueError(f"Device model {device_model} version {device_model_version} not found.")
        
        with open(converter_config_file_path) as f:
            converter_config = yaml.safe_load(f)
        return converter_module_path, converter_class_name, converter_config

    def get_task_category(self) -> str:
        return "lerobot_format_convert"

    def generate_task_content(self) -> dict | None:
        """
        【顶级异常捕获】任何错误都只打日志，返回None，不中断服务
        数据库异常自动回滚，保证事务安全
        ✅ 修复：增加数据库行锁，原子性获取任务，彻底解决多客户端重复分发问题
        ✅ 修复：遇到任何错误，强制将当前数据集状态改为FAILED，避免卡死
        """
        current_item = None
        try:
            convert_start_str, convert_start_ts = self._get_formatted_time()
            
            with self.db.with_session() as session:
                try:
                    if self.is_test:
                        query = session.query(DatasetDB).filter(
                            DatasetDB.convert_test_status == TaskStatus.PENDING,
                        )
                        if self.specific_device_model:
                            query = query.filter(DatasetDB.device_model == self.specific_device_model)

                        # 🔒 核心修复：with_for_update() 行锁 + 原子查询
                        item = query.with_for_update(skip_locked=True).first()
                        if item is None:
                            session.commit()
                            return None

                        current_item = item
                        base_version = item.convert_version if isinstance(item.convert_version, int) else 0
                        item.convert_test_version = base_version + 1
                        item.convert_test_status = TaskStatus.PROCESSING

                    else:
                        query = session.query(DatasetDB).filter(
                            and_(
                                DatasetDB.convert_test_status == TaskStatus.COMPLETED,
                                DatasetDB.convert_status == TaskStatus.PENDING,
                            )
                        )
                        if self.specific_device_model:
                            query = query.filter(DatasetDB.device_model == self.specific_device_model)

                        # 🔒 核心修复：with_for_update() 行锁 + 原子查询
                        item = query.with_for_update(skip_locked=True).first()
                        if item is None:
                            session.commit()
                            return None
                        
                        current_item = item
                        item.convert_status = TaskStatus.PROCESSING
                        test_version = item.convert_test_version if isinstance(item.convert_test_version, int) else 0
                        item.convert_version_ps = test_version
                        current_version = item.convert_version if isinstance(item.convert_version, int) else 0
                        item.convert_version = current_version + 1

                    # 路径处理
                    item.convert_path = str(Path(self.convert_root_path) / f"{item.dataset_name}_{item.dataset_name_id}")
                    item.data_path = str(Path(item.yaml_file_path).parent)
                    item.convert_start_timestamp = convert_start_str
                    session.commit()

                except Exception as e:
                    session.rollback()
                    self.logger.error(f"数据库查询/更新任务失败: {e}", exc_info=True)
                    if current_item:
                        self._mark_task_failed(session, current_item, str(e))
                    return None

                # 获取转换器配置
                try:
                    converter_module_path, converter_class_name, converter_config = (
                        self._get_converter_module_class_config(
                            device_model=item.device_model,
                            device_model_version=item.device_model_version,
                        )
                    )
                except Exception as e:
                    session.rollback()
                    self.logger.error(f"获取转换器配置失败: {e}", exc_info=True)
                    self._mark_task_failed(session, item, str(e))
                    return None
                
                leformat_name = f"{item.dataset_name.lower()}_{item.dataset_name_id}"
                client_log_path = Path(self.convert_root_path) / "client_logs" / leformat_name
                repo_id = f"{ROBOCOIN_PLATFORM}/{item.dataset_name}_{item.dataset_name_id}"

                task_content = {
                    DATASET_UUID: item.dataset_uuid,
                    DATASET_NAME: f"{item.dataset_name}_{item.dataset_name_id}",
                    LEFORMAT_PATH: item.convert_path,
                    DATASET_PATH: item.data_path,
                    DEVICE_MODEL: item.device_model,
                    CONVERTER_CONFIG: converter_config,
                    CONVERTER_MODULE_PATH: converter_module_path,
                    CONVERTER_CLASS_NAME: converter_class_name,
                    VIDEO_BACKEND: self.video_backend,
                    IMAGE_WRITER_PROCESSES: self.image_writer_processes,
                    IMAGE_WRITER_THREADS: self.image_writer_threads,
                    CONVERTER_LOG_DIR: str(client_log_path),
                    REPO_ID: repo_id,
                    CONVERTER_LOG_NAME: leformat_name,
                    IS_TEST: self.is_test,
                    AUTO_REENCODE: self.auto_reencode,
                    "convert_start_str": convert_start_str,
                    "convert_start_ts": convert_start_ts,
                    "dataset_uuid": item.dataset_uuid
                }

                self.logger.info(f"开始转换数据集 {task_content[DATASET_NAME]} (UUID: {item.dataset_uuid}), 开始时间: {convert_start_str}")
                return task_content

        except Exception as e:
            self.logger.error(f"生成转换任务失败，跳过当前任务: {e}", exc_info=True)
            try:
                if current_item:
                    with self.db.with_session() as session:
                        self._mark_task_failed(session, current_item, str(e))
            except:
                pass
            return None


    def _mark_task_failed(self, session, item: DatasetDB, err_msg: str):
        """
        统一标记任务失败：更新数据库状态 + 错误信息
        """
        try:
            item.convert_err_msg = err_msg
            if self.is_test:
                item.convert_test_status = TaskStatus.FAILED
            else:
                item.convert_status = TaskStatus.FAILED
            session.commit()
            self.logger.error(f"✅ 已自动标记数据集 {item.dataset_name} 为失败状态: {err_msg}")
        except Exception as e:
            session.rollback()
            self.logger.error(f"标记任务失败时发生错误: {e}", exc_info=True)


    def _normalize_gripper_open(self, dataset_path: str):
        """
        【全量异常捕获】夹爪归一化，任何错误都只打日志，不抛出
        """
        try:
            if not self.enable_gripper_normalization:
                self.logger.info("夹爪归一化功能已禁用，跳过该步骤")
                return
            
            self.logger.info(f"开始对数据集 {dataset_path} 执行夹爪开度归一化...")
            normalizer = GripperOpenNormalizer(dataset_path)
            success = normalizer.run()
            
            if success:
                self.logger.info(f"数据集 {dataset_path} 夹爪归一化完成")
            else:
                self.logger.warning(f"数据集 {dataset_path} 夹爪归一化未完全执行")
                
        except Exception as e:
            self.logger.error(f"执行夹爪归一化完全失败: {e}", exc_info=True)

    def handle_task_result(self, task_content: dict, task_result_content: dict) -> None:
        """
        【顶级异常捕获】处理任务结果，任何错误都标记任务失败，不中断服务
        保证数据库状态一定会更新，日志一定会记录
        """
        try:
            # 基础数据解析
            ds_uuid = task_content.get(DATASET_UUID)
            dataset_name = task_content.get(DATASET_NAME, "未知数据集")
            dataset_path = task_content.get(LEFORMAT_PATH)
            convert_start_str = task_content.get("convert_start_str")
            convert_start_ts = task_content.get("convert_start_ts")

            # 生成结束时间
            convert_end_str, convert_end_ts = self._get_formatted_time()

            # 计算耗时
            convert_duration = 0.0
            try:
                if convert_start_ts:
                    convert_duration = convert_end_ts - convert_start_ts
                else:
                    convert_start_dt = self._parse_time_str_to_datetime(convert_start_str)
                    convert_end_dt = self._parse_time_str_to_datetime(convert_end_str)
                    convert_duration = (convert_end_dt - convert_start_dt).total_seconds()
            except Exception as e:
                self.logger.warning(f"计算任务耗时失败: {e}", exc_info=True)

            # 解析任务结果
            task_status = task_result_content.get(TASK_RESULT_STATUS)
            task_status_msg = task_result_content.get(ERR_MSG, "未知错误")
            actual_result = task_result_content.get(TASK_RESULT_CONTENT, {})
            total_episodes = actual_result.get("total_episodes", 0)
            converted_episodes = actual_result.get("converted_episodes", 0)
            skipped_episodes = actual_result.get("skipped_episodes", 0)

            # 默认任务失败
            convert_status = TaskStatus.COMPLETED if task_status == TASK_SUCCESS else TaskStatus.FAILED

            # 更新数据库状态
            with self.db.with_session() as session:
                try:
                    item = session.query(DatasetDB).filter(DatasetDB.dataset_uuid == ds_uuid).first()
                    if not item:
                        self.logger.error(f"Dataset {ds_uuid} 不存在，无法更新状态")
                        return

                    # 更新任务结果
                    item.convert_err_msg = task_status_msg
                    item.converted_episodes = converted_episodes
                    item.total_episodes = total_episodes
                    item.skipped_episodes = skipped_episodes
                    item.convert_end_timestamp = convert_end_str
                    item.convert_duration_seconds = convert_duration

                    # 更新任务状态
                    if self.is_test:
                        item.convert_test_status = convert_status
                    else:
                        item.convert_status = convert_status
                    
                    session.commit()
                except Exception as e:
                    session.rollback()
                    self.logger.error(f"更新数据库任务结果失败: {e}", exc_info=True)
                    return

            # 日志输出
            duration_str = f"{convert_duration:.2f} 秒"
            if convert_duration > 60:
                duration_str += f" ({convert_duration/60:.2f} 分钟)"
            self.logger.info(
                f"数据集 {dataset_name} (UUID: {ds_uuid}) 处理完成！状态: {convert_status}, 耗时: {duration_str}"
            )

            # 仅任务成功时执行后置处理
            if task_status == TASK_SUCCESS and dataset_path:
                # 夹爪归一化（内部已捕获所有异常）
                self._normalize_gripper_open(dataset_path)

                # 数据集校验（全量捕获异常）
                if self.enable_dataset_validation:
                    try:
                        self.logger.info(f"开始对数据集 {dataset_path} 执行后置物理极限校验...")
                        validator = LerobotDatasetValidator(dataset_path)
                        is_valid, error_details = validator.run()
                        if is_valid:
                            self.logger.info(f"✅ 数据集 {dataset_path} 后置校验通过")
                        else:
                            self.logger.error(f"❌ 数据集 {dataset_path} 校验未通过: {error_details}")
                    except Exception as e:
                        self.logger.error(f"数据集校验执行失败: {e}", exc_info=True)

        # 【全局捕获】处理结果的所有异常，保证服务不崩
        except Exception as e:
            self.logger.error(f"处理任务结果完全失败: {e}", exc_info=True)
            # 尝试强制更新数据库为失败状态
            try:
                ds_uuid = task_content.get(DATASET_UUID)
                with self.db.with_session() as session:
                    item = session.query(DatasetDB).filter(DatasetDB.dataset_uuid == ds_uuid).first()
                    if item:
                        if self.is_test:
                            item.convert_test_status = TaskStatus.FAILED
                        else:
                            item.convert_status = TaskStatus.FAILED
                        item.convert_err_msg = f"服务内部处理失败: {str(e)}"
                        session.commit()
            except:
                pass