from pathlib import Path
import os
import json
from multiprocessing import Pool, cpu_count
from functools import partial
import atexit

from robocoin_dataset.quality_check.checkers import (
    close_hand_detector,
    check_video_3_scores
)



def disable_gpu():
    """强制禁用 GPU 解码（通过环境变量）"""
    os.environ["ENABLE_GPU_ACCELERATION"] = "False"

def process_one_video(video_path: str, output_dir: Path) -> None:
    """单个视频的处理函数，结果写入 output_dir 下的独立 JSON 文件"""
    result_json = check_video_3_scores(video_path) 
    print(result_json)
    # video_name = Path(video_path).stem
    # out_file = output_dir / f"{video_name}_result.json"
    # with open(out_file, "w", encoding="utf-8") as f:
    #     f.write(result_json)

def batch_process_videos(video_paths: list[str], output_dir: str | Path, num_workers: int = None):
    """
    多进程批量处理视频
    :param video_paths: 视频路径列表
    :param output_dir: 结果保存目录
    :param num_workers: 进程数，默认为 CPU 核心数的一半（至少为1）
    """
    # 主进程禁用 GPU（环境变量会继承到子进程）
    disable_gpu()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if num_workers is None:
        num_workers = max(1, cpu_count() // 2)

    # 固定 output_dir 参数
    worker_func = partial(process_one_video, output_dir=output_dir)

    print(f"启动 {num_workers} 个进程，处理 {len(video_paths)} 个视频...")
    with Pool(processes=num_workers) as pool:
        # 使用 imap_unordered 可尽快获取结果，每完成一个视频打印进度
        for i, _ in enumerate(pool.imap_unordered(worker_func, video_paths), 1):
            if i % 100 == 0 or i == len(video_paths):
                print(f"已处理 {i}/{len(video_paths)} 个视频")

    print(f"全部处理完成！结果保存在 {output_dir}")



import atexit
atexit.register(close_hand_detector)

# ====================== 【测试示例】 ======================
if __name__ == "__main__":
    # # 测试视频路径
    # TEST_VIDEO = ["/home/liuyou/Downloads/demo_data/episode_000023.mp4", "/home/liuyou/Downloads/demo_data/head_right_camera_undistorted.mp4"]
    # # 调用方法，输出3个算子的分数JSON
    # for video in TEST_VIDEO:
    #     result_json = check_video_3_scores(video)
    #     print("===== 三个算子合并分数 =====")
    #     print(result_json)

     # 示例：生成视频路径列表（您需要根据实际数据来源替换）
    all_videos = [
        "/home/liuyou/Downloads/demo_data/episode_000023.mp4",
        "/home/liuyou/Downloads/demo_data/head_right_camera_undistorted.mp4",
        "/home/liuyou/Downloads/demo_data/episode_000023.mp4",
        "/home/liuyou/Downloads/demo_data/head_right_camera_undistorted.mp4",
        "/home/liuyou/Downloads/demo_data/episode_000023.mp4",
        "/home/liuyou/Downloads/demo_data/head_right_camera_undistorted.mp4",
        # 更多视频路径...
    ]

    RESULT_DIR = "./video_quality_results"

    # 若您已通过手动开关禁用 GPU（ENABLE_GPU_ACCELERATION = False），可直接调用
    batch_process_videos(all_videos, RESULT_DIR, num_workers=4)  # 可调整进程数