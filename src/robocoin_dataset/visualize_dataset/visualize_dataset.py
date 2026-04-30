import gc
import logging
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import rerun as rr
import torch
import torch.utils.data
import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore
from sqlalchemy.sql import and_, or_

from robocoin_dataset.database.database import DatasetDatabase
from robocoin_dataset.database.models import DatasetDB, DatasetHardLinkDB, TaskStatus
from robocoin_dataset.distribution_computation.constant import (
    DATASET_UUID,
    ERR_MSG,
    TASK_RESULT_STATUS,
    TASK_SUCCESS,
)
from robocoin_dataset.distribution_computation.task_client import TaskClient
from robocoin_dataset.distribution_computation.task_server import TaskServer
from robocoin_dataset.format_converter.tolerobot.constant import (
    HARD_LINK_PATH,
)


class EpisodeSampler(torch.utils.data.Sampler):
    def __init__(self, dataset: LeRobotDataset, episode_index: int) -> None:
        from_idx = dataset.episode_data_index["from"][episode_index].item()
        to_idx = dataset.episode_data_index["to"][episode_index].item()
        self.frame_ids = range(from_idx, to_idx)

    def __iter__(self) -> Iterator:
        return iter(self.frame_ids)

    def __len__(self) -> int:
        return len(self.frame_ids)


def to_hwc_uint8_numpy(chw_float32_torch: torch.Tensor) -> np.ndarray:
    assert chw_float32_torch.dtype == torch.float32
    assert chw_float32_torch.ndim == 3
    c, h, w = chw_float32_torch.shape
    assert c < h and c < w, f"expect channel first images, but instead {chw_float32_torch.shape}"
    return (chw_float32_torch * 255).type(torch.uint8).permute(1, 2, 0).numpy()


def get_annotation_text(batch: dict, idx: int) -> str:
    annotation_text = ""
    # tasks
    if "task" in batch:
        annotation_text += f"Task:\n{batch['task'][idx]};\n"
    annotation_text += "\n"
    # subtasks
    if "subtasks" in batch:
        annotation_text += f"Subtasks:\n{batch['subtasks'][idx]}\n"
    annotation_text += "\n"

    if "scene" in batch:
        annotation_text += f"Scene Annotation:\n{batch['scene'][idx]}\n"
    # left arm state annotations
    # EEF acceleration magnitude
    annotation_text += "\n"
    annotation_text += "Motion Annotations:\n"
    if "left_eef_acc_mag_state" in batch:
        annotation_text += f"Left EEF acc mag state: {batch['left_eef_acc_mag_state'][idx]}\n"
    # EEF direction
    if "left_eef_direction_state" in batch:
        annotation_text += f"Left EEF direction state: {batch['left_eef_direction_state'][idx]}\n"
    # EEF velocity
    if "left_eef_velocity_state" in batch:
        annotation_text += f"Left EEF velocity state: {batch['left_eef_velocity_state'][idx]}\n"
    # Gripper mode
    if "left_gripper_mode_state" in batch:
        annotation_text += f"Left gripper mode state: {batch['left_gripper_mode_state'][idx]}\n"
    # Gripper activity
    if "left_gripper_activity_state" in batch:
        annotation_text += (
            f"Left gripper activity state: {batch['left_gripper_activity_state'][idx]}\n"
        )

    annotation_text += "\n"

    # left arm action annotations
    if "left_eef_acc_mag_action" in batch:
        annotation_text += f"Left EEF acc mag action: {batch['left_eef_acc_mag_action'][idx]}\n"
    if "left_eef_direction_action" in batch:
        annotation_text += f"Left EEF direction action: {batch['left_eef_direction_action'][idx]}\n"
    if "left_eef_velocity_action" in batch:
        annotation_text += f"Left EEF velocity action: {batch['left_eef_velocity_action'][idx]}\n"
    if "left_gripper_mode_action" in batch:
        annotation_text += f"Left gripper mode action: {batch['left_gripper_mode_action'][idx]}\n"
    if "left_gripper_activity_action" in batch:
        annotation_text += (
            f"Left gripper activity action: {batch['left_gripper_activity_action'][idx]}\n"
        )

    annotation_text += "\n"

    # right arm state annotations
    if "right_eef_acc_mag_state" in batch:
        annotation_text += f"Right EEF acc mag state: {batch['right_eef_acc_mag_state'][idx]}\n"
    if "right_eef_direction_state" in batch:
        annotation_text += f"Right EEF direction state: {batch['right_eef_direction_state'][idx]}\n"
    if "right_eef_velocity_state" in batch:
        annotation_text += f"Right EEF velocity state: {batch['right_eef_velocity_state'][idx]}\n"
    if "right_gripper_mode_state" in batch:
        annotation_text += f"Right gripper mode state: {batch['right_gripper_mode_state'][idx]}\n"
    if "right_gripper_activity_state" in batch:
        annotation_text += (
            f"Right gripper activity state: {batch['right_gripper_activity_state'][idx]}\n"
        )

    annotation_text += "\n"

    # right arm action annotations
    if "right_eef_acc_mag_action" in batch:
        annotation_text += f"Right EEF acc mag action: {batch['right_eef_acc_mag_action'][idx]}\n"
    if "right_eef_direction_action" in batch:
        annotation_text += (
            f"Right EEF direction action: {batch['right_eef_direction_action'][idx]}\n"
        )
    if "right_eef_velocity_action" in batch:
        annotation_text += f"Right EEF velocity action: {batch['right_eef_velocity_action'][idx]}\n"
    if "right_gripper_mode_action" in batch:
        annotation_text += f"Right gripper mode action: {batch['right_gripper_mode_action'][idx]}\n"
    if "right_gripper_activity_action" in batch:
        annotation_text += (
            f"Right gripper activity action: {batch['right_gripper_activity_action'][idx]}\n"
        )

    return annotation_text


def visualize_dataset(
    repo_path: str | Path,
    episode_index: int = 0,
    batch_size: int = 32,
    num_workers: int = 4,
    save: bool = False,
    output_dir: Path | None = None,
) -> None:
    if save:
        assert output_dir is not None, (
            "Set an output directory where to write .rrd files with `--output-dir path/to/directory`."
        )

    repo_path = Path(repo_path).expanduser().absolute()
    dataset = LeRobotDataset("test/visualize_dataset", repo_path)
    episode_sampler = EpisodeSampler(dataset, episode_index)
    _visualize_episode(
        dataset, episode_sampler, repo_path, num_workers=num_workers, batch_size=batch_size
    )
    user_input = input("输入e后回车，提交错误信息；输入c后回车，确认数据无误并提交结果")
    while True:
        if user_input == "e":
            err_msg = input("请输入错误信息后按回车")
            raise RuntimeError(f"人工检查发现错误: {err_msg}")
        if user_input == "c":
            break
        user_input = input("输入错误，请重新输入")


def _visualize_episode(
    dataset: LeRobotDataset,
    sampler: EpisodeSampler,
    repo_path: str | Path,
    num_workers: int = 4,
    batch_size: int = 32,
) -> None:
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        sampler=sampler,
    )

    logging.info("Starting Rerun")

    rr.init(f"{str(repo_path)}", spawn=True)

    gc.collect()

    logging.info("Logging to Rerun")

    for batch in tqdm.tqdm(dataloader, total=len(dataloader)):
        # iterate over the batch
        for i in range(len(batch["index"])):
            rr.set_time("frame_index", sequence = batch["frame_index"][i].item())
            rr.set_time("timestamp", seconds = batch["timestamp"][i].item())

            # display each camera image
            for key in dataset.meta.camera_keys:
                # TODO(rcadene): add `.compress()`? is it lossless?
                rr.log(key, rr.Image(to_hwc_uint8_numpy(batch[key][i])))

            annotation_text = get_annotation_text(batch, i)
            if annotation_text:
                rr.log("annotations", rr.TextLog(str(repo_path) + annotation_text))

    return


class DatasetVisualizerServer(TaskServer):
    def __init__(
        self,
        db_file_path: str | Path,
        host: str = "0.0.0.0",
        port: int = 2110,
        heartbeat_interval: float = 30.0,  # 服务端每30秒发一次 ping
        timeout: float = 15.0,  # 等待 pong 超过15秒则断开
        device_model: str | None = None,
        target_dataset_uuid: str | None = None,  # 新增参数
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(
            logger=logger,
            host=host,
            port=port,
            heartbeat_interval=heartbeat_interval,
            timeout=timeout,
        )
        db_file_path = Path(db_file_path).expanduser().absolute()

        self.db_file_path: Path = Path(db_file_path).expanduser().absolute()
        self.db = DatasetDatabase(self.db_file_path)
        self.device_model = device_model
        self.target_dataset_uuid = target_dataset_uuid  # 新增属性
        self.logger = logger or logging.getLogger(__name__)

    def get_task_category(self) -> str:
        return "visualize_dataset"

    def generate_task_content(self) -> dict | None:
        with self.db.with_session() as session:
            query = session.query(DatasetDB).filter(
                and_(
                    # 关键修改1：前置条件改为 data_merge_status 完成
                    DatasetDB.data_merge_status == TaskStatus.COMPLETED,
                    # 两个触发分支
                    or_(
                        # 分支1: 正在排队
                        DatasetDB.visualize_check_status == TaskStatus.PENDING,
                        # 分支2: 已完成但版本过期
                        and_(
                            DatasetDB.visualize_check_status == TaskStatus.COMPLETED,
                            # 关键修改2：版本基准改为 data_merge_version
                            DatasetDB.visualize_check_version_ps != DatasetDB.data_merge_version,
                        ),
                    ),
                )
            )
            
            # ========== 新增：如果指定了 UUID，仅筛选该 UUID ==========
            if self.target_dataset_uuid:
                query = query.filter(DatasetDB.dataset_uuid == self.target_dataset_uuid)
            
            print(f"device_model: {self.device_model}")
            if self.device_model:
                query = query.filter(
                    DatasetDB.device_model == self.device_model,
                )

            item = query.first()

            if not item:
                return None

            print(item.dataset_uuid)
            query = session.query(DatasetHardLinkDB).filter(
                DatasetHardLinkDB.dataset_uuid == item.dataset_uuid
            )
            hard_link_item = query.first()
            if not hard_link_item:
                return None

            if not hard_link_item.hard_link_path:
                return None

            item.visualize_check_status = TaskStatus.PROCESSING
            item.visualize_check_version = item.visualize_check_version + 1
            # 关键修改3：版本关联改为 data_merge_version
            item.visualize_check_version_ps = item.data_merge_version

            session.commit()

            return {
                DATASET_UUID: item.dataset_uuid,
                HARD_LINK_PATH: hard_link_item.hard_link_path,  # 源路径保持不变
            }

    def handle_task_result(self, task_content: dict, task_result_content: dict) -> None:
        ds_uuid = task_content.get(DATASET_UUID)

        task_status = task_result_content.get(TASK_RESULT_STATUS)
        task_status_msg = task_result_content.get(ERR_MSG)

        convert_status = TaskStatus.COMPLETED if task_status == TASK_SUCCESS else TaskStatus.FAILED

        with self.db.with_session() as session:
            item = session.query(DatasetDB).filter(DatasetDB.dataset_uuid == ds_uuid).first()
            if item is None:
                self.logger.error(f"Dataset {ds_uuid} not found in dataset DB.")
                return  # 增加防护，避免后续报错

            print(item.visualize_check_status)
            item.visualize_check_status = convert_status
            item.visualize_check_err_msg = task_status_msg
            self.logger.info(
                f"Upsert {item.convert_path} visualize check status to {convert_status}, "
                f"update_message: {task_status_msg}"
            )
            session.commit()


class DatasetVisualizerClient(TaskClient):
    def __init__(
        self,
        server_uri: str = "ws://localhost:2110",
        heartbeat_interval: float = 10.0,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(
            server_uri=server_uri,
            heartbeat_interval=heartbeat_interval,
            logger=logger,
        )

    def get_task_category(self) -> str:
        return "visualize_dataset"

    def generate_task_request_desc(self) -> dict:
        """客户端可自定义任务请求参数"""
        return {}

    async def process_task(self, task_data: dict) -> dict:
        import asyncio
        import traceback

        from robocoin_dataset.distribution_computation.constant import (
            ERR_MSG,
            TASK_FAILED,
            TASK_ID,
            TASK_RESULT_CONTENT,
            TASK_RESULT_STATUS,
            TASK_SUCCESS,
        )

        loop = asyncio.get_event_loop()
        task_id = task_data.get(TASK_ID)
        try:
            task_result_content = await loop.run_in_executor(
                None, self._sync_process_task, task_data
            )
            return {TASK_RESULT_STATUS: TASK_SUCCESS, TASK_RESULT_CONTENT: task_result_content}
        except Exception as e:
            if self.logger:
                self.logger.error(f"Task {task_id} failed. {traceback.format_exc()}")
            # 检查是否为人工检查错误
            if isinstance(e, RuntimeError) and str(e).startswith("人工检查发现错误: "):
                err_msg = str(e)
            else:
                err_msg = f"Task {task_id} failed. {traceback.format_exc()}"
            return {
                TASK_RESULT_STATUS: TASK_FAILED,
                ERR_MSG: err_msg,
                TASK_RESULT_CONTENT: {},
            }

    def _sync_process_task(self, task_content: dict) -> dict:
        try:
            hard_link_path = task_content.get(HARD_LINK_PATH)

            visualize_dataset(
                repo_path=hard_link_path,
            )

            return {}
        except Exception as e:
            raise e
