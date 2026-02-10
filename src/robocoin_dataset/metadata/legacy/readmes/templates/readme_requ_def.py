# dataset_info_example_with_sources.py
"""
Dataset Info 字段示例及数据来源说明
用于生成 dataset_info.yml，进而生成 README.md

数据来源说明：
1. 模板固定值：从 dataset_info.yml 模板直接使用的固定值
2. meta/info.json：从数据集的 meta/info.json 文件自动提取
3. meta/tasks.jsonl：从数据集的 meta/tasks.jsonl 文件自动提取
4. annotations/subtask_annotations.jsonl：从标注文件提取
5. 数据库：从 DatasetDB 数据库表提取
6. 计算生成：根据其他字段计算得出
"""

dataset_info_dict = {
    # ========================================================================
    # 1. 基础元数据 - 来源：模板固定值（dataset_info.yml template）
    # ========================================================================
    "license": "apache-2.0",  # 来源：模板固定值

    "language": ["en", "zh"],  # 来源：模板固定值，支持的语言列表

    "task_categories": ["robotics"],  # 来源：模板固定值，任务分类

    "tags": [  # 来源：模板固定值 + 可选的自定义标签
        "RoboCoin",  # 模板固定值
        "LeRobot",   # 模板固定值
        # 如果配置了 task_tags_yamls_dir，会从 {dataset_name}.yml 额外添加标签
    ],

    "frame_range": "100K-1M",  # 来源：计算生成（根据 total_frames 自动计算）
    # 规则：<1K, 1K-10K, 10K-100K, 100K-1M, 1M-10M, ...

    "configs": [  # 来源：模板固定值
        {
            "config_name": "default",
            "data_files": "data/*/*.parquet"
        }
    ],

    # ========================================================================
    # 2. 作者信息 - 来源：模板固定值
    # ========================================================================
    "authors": {
        "contributed_by": [  # 来源：模板固定值，可手动修改
            {
                "name": "RoboCoin",
                "url": "https://RoboCoin.github.io",  # 可选
                "affiliation": "RoboCoin Team"  # 可选
            }
        ],
        "annotated_by": [  # 可选，来源：模板或手动添加
            {
                "name": "RoboCoin",
                "url": "https://RoboCoin.github.io",
                "affiliation": "RoboCoin Team"
            }
        ]
    },

    # ========================================================================
    # 3. 数据集描述 - 来源：模板固定值
    # ========================================================================
    "dataset_description": (
        "This dataset uses an extended format based on [LeRobot]"
        "(https://github.com/huggingface/lerobot) and is fully compatible with LeRobot."
    ),  # 来源：模板固定值

    # ========================================================================
    # 4. 外部链接 - 来源：模板固定值
    # ========================================================================
    "homepage": "https://RoboCoin.github.io/",  # 来源：模板固定值
    "paper": "in comming",  # 来源：模板固定值（可选）
    "repository": "https://github.com/RoboCoin/robocoin-dataset",  # 来源：模板固定值
    "project_page": "https://RoboCoin.github.io/",  # 来源：模板固定值（可选）

    # ========================================================================
    # 5. 机器人信息 - 来源：meta/info.json
    # ========================================================================
    "robot_type": "unitree_g1",  # 来源：meta/info.json["robot_type"]
    "codebase_version": "v2.1",  # 来源：meta/info.json["codebase_version"]

    # ========================================================================
    # 6. 任务描述 - 来源：meta/tasks.jsonl 和 annotations/
    # ========================================================================
    "tasks": "pick up the apple from the table and place it into the basket.",
    # 来源：meta/tasks.jsonl，读取所有 task 字段，用换行符连接

    "sub_tasks": [  # 来源：annotations/subtask_annotations.jsonl
        "End",
        "Grasp the apple with the left gripper",
        "Grasp the basket with the right gripper",
        "Move the basket to the front of the table with the right gripper",
        "null",
        "Place the apple into the basket with the left gripper",
        "Static"
    ],
    # 提取方式：
    # 1. 读取 annotations/subtask_annotations.jsonl
    # 2. 提取所有 "subtask" 字段
    # 3. 去重（不区分大小写）
    # 4. 排序并返回列表

    # ========================================================================
    # 7. 统计信息 - 来源：meta/info.json
    # ========================================================================
    "statistics": {
        "total_episodes": 90,     # 来源：meta/info.json["total_episodes"]
        "total_frames": 46769,    # 来源：meta/info.json["total_frames"]
        "total_tasks": 1,         # 来源：meta/info.json["total_tasks"]
        "total_videos": 270,      # 来源：meta/info.json["total_videos"]
        "total_chunks": 1,        # 来源：meta/info.json["total_chunks"]
        "chunks_size": 1000,      # 来源：meta/info.json["chunks_size"]
        "fps": 30,                # 来源：meta/info.json["fps"]
        # 以下字段可选，如果 meta/info.json 中存在则提取
        "total_duration": "可选",     # 来源：meta/info.json["total_duration"]（如果存在）
        "video_resolution": "可选",   # 来源：meta/info.json["video_resolution"]（如果存在）
        "state_dim": "可选",         # 来源：meta/info.json["state_dim"]（如果存在）
        "action_dim": "可选",        # 来源：meta/info.json["action_dim"]（如果存在）
        "camera_views": "可选"       # 来源：meta/info.json["camera_views"]（如果存在）
    },

    # ========================================================================
    # 8. 数据分割 - 来源：meta/info.json
    # ========================================================================
    "splits": {
        "train": "0:89",  # 来源：meta/info.json["splits"]["train"]
        "val": "可选",     # 来源：meta/info.json["splits"]["val"]（如果存在）
        "test": "可选"     # 来源：meta/info.json["splits"]["test"]（如果存在）
    },

    # ========================================================================
    # 9. 路径模式 - 来源：meta/info.json
    # ========================================================================
    "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    # 来源：meta/info.json["data_path"]

    "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    # 来源：meta/info.json["video_path"]

    # ========================================================================
    # 10. Features Schema - 来源：meta/info.json
    # ========================================================================
    "features": {
        # 完整的 features 定义从 meta/info.json["features"] 提取
        # 包括：observation.images.*, observation.state, action, timestamp,
        #       frame_index, episode_index, index, task_index,
        #       各种标注字段（subtask_annotation, scene_annotation, 等）
        #       以及额外的运动特征（eef_*, gripper_*）

        "observation.images.cam_high_rgb": {
            "dtype": "video",
            "shape": [480, 640, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": 480,
                "video.width": 640,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": 30,
                "video.channels": 3,
                "has_audio": False
            }
        },
        # ... 其他相机视角 ...
        "observation.state": {
            "dtype": "float32",
            "shape": [28],
            "names": [
                "left_arm_joint_1_rad",
                # ... 其他关节 ...
            ]
        },
        "action": {
            "dtype": "float32",
            "shape": [28],
            "names": ["..."]  # 与 observation.state 类似
        },
        # ... 时间戳和索引字段 ...
        # ... 标注字段 ...
        # ... 运动特征字段 ...
    },
    # 来源：完整复制 meta/info.json["features"]

    # ========================================================================
    # 11. 相机信息 - 来源：模板或自动生成
    # ========================================================================
    "cameras": "auto_generated",
    # 可以手动指定或保持 "auto_generated"，从 features 中自动计算相机数量

    # ========================================================================
    # 12. 观测和动作空间 - 来源：模板（可自动生成）
    # ========================================================================
    "observation_space": {
        "images": "auto_generated",  # 从 features 中提取
        "state": "auto_generated"    # 从 features 中提取
    },
    "action_space": "auto_generated",  # 从 features 中提取

    # ========================================================================
    # 13. 标注信息 - 来源：模板（表示存在哪些标注）
    # ========================================================================
    "annotations": {
        "subtask_annotation": "auto_generated",   # 表示有子任务标注
        "scene_annotation": "auto_generated",     # 表示有场景标注
        "eef_direction": "auto_generated",        # 表示有末端执行器方向标注
        "eef_velocity": "auto_generated",         # 表示有末端执行器速度标注
        "eef_acc_mag": "auto_generated",          # 表示有末端执行器加速度标注
        "gripper_mode": "auto_generated",         # 表示有夹爪模式标注
        "gripper_activity": "auto_generated"      # 表示有夹爪活动状态标注
    },

    # ========================================================================
    # 14. 额外特征 - 来源：模板（表示存在哪些额外特征）
    # ========================================================================
    "eef_sim_pose": "auto_generated",        # 表示有末端执行器仿真位姿
    "gripper_open_scale": "auto_generated",  # 表示有夹爪开合比例

    # ========================================================================
    # 15. 深度相机支持 - 来源：meta/info.json（计算生成）
    # ========================================================================
    "depth_enabled": False,
    # 来源：检查 features 中的所有 observation.images.* 字段
    # 如果任一 info["video.is_depth_map"] 为 True，则此字段为 True

    # ========================================================================
    # 16. 数据模式描述 - 来源：模板或手动添加
    # ========================================================================
    "data_schema": "auto_generated",  # 可选，数据模式的文本描述

    # ========================================================================
    # 17. 目录结构 - 来源：计算生成
    # ========================================================================
    "structure": "auto_generated",
    # 来源：自动遍历数据集目录，生成树状结构
    # 显示叶子目录及其前5个文件
    # 生成方式：generate_folder_structure(dataset_path, max_files_per_dir=5)

    # ========================================================================
    # 18. 联系信息 - 来源：模板固定值
    # ========================================================================
    "contact_email": "contact@robocoin.ai",  # 来源：模板固定值（可选）

    "contact_info": (
        "For questions, issues, or feedback regarding this dataset, "
        "please contact us.\n"
    ),  # 来源：模板固定值（可选）

    "support_info": (
        "For technical support, please open an issue on our GitHub repository.\n"
    ),  # 来源：模板固定值（可选）

    # ========================================================================
    # 19. 许可证详情 - 来源：模板固定值
    # ========================================================================
    "license_details": (
        "Please refer to the LICENSE file for full license terms and conditions.\n"
    ),  # 来源：模板固定值（可选）

    # ========================================================================
    # 20. 引用信息 - 来源：模板固定值
    # ========================================================================
    "citation_bibtex": None,  # 来源：模板固定值，可手动添加 BibTeX

    "additional_citations": (
        "If you use this dataset, please also consider citing:\n"
        "- LeRobot Framework: https://github.com/huggingface/lerobot\n"
    ),  # 来源：模板固定值（可选）

    # ========================================================================
    # 21. 版本信息 - 来源：模板固定值
    # ========================================================================
    "version_info": (
        "## Version History\n"
        "- v1.0.0 (2025-11): Initial release\n"
    )  # 来源：模板固定值（可选）
}


# ============================================================================
# 数据来源详细说明
# ============================================================================

"""
数据流程：

1. 数据集创建阶段（格式转换）：
   - 原始数据 → 格式转换 → 生成 meta/info.json 和 meta/tasks.jsonl
   - 这些文件包含数据集的核心统计信息和特征定义

2. 标注阶段：
   - 生成 annotations/subtask_annotations.jsonl（子任务标注）
   - 生成 annotations/scene_annotations.jsonl（场景标注）

3. 数据库记录阶段：
   - 数据集信息录入 DatasetDB 表
   - 多对多关系：scene_types, task_descriptions, atomic_actions, objects

4. README生成阶段：
   - Step 1: 运行 dataset_info_util.py 生成 dataset_info.yml
     * 从模板加载固定值
     * 从 meta/info.json 提取统计和特征信息
     * 从 meta/tasks.jsonl 提取任务描述
     * 从 annotations/ 提取标注信息
     * 合并所有信息生成 dataset_info.yml

   - Step 2: 运行 dataset_readme_util.py 生成 README.md
     * 读取 dataset_info.yml
     * 使用 readme.j2 模板渲染
     * 生成最终的 README.md

文件位置说明：
- 模板文件：src/robocoin_dataset/readmes/templates/
  * dataset_info.yml - 数据集信息模板
  * readme.j2 - README Jinja2 模板

- 数据集文件结构：
  {dataset_path}/
    ├── meta/
    │   ├── info.json              # 核心统计和特征信息
    │   └── tasks.jsonl            # 任务描述
    ├── annotations/
    │   ├── subtask_annotations.jsonl  # 子任务标注
    │   └── scene_annotations.jsonl    # 场景标注（可选）
    ├── data/                      # Parquet 数据文件
    ├── videos/                    # 视频文件
    └── README.md                  # 生成的 README

- 输出文件：
  {output_path}/{dataset_name}/dataset_info.yml  # 生成的数据集信息文件

数据库相关字段（如需从数据库获取）：
- DatasetDB 表字段（一对一）：
  * device_model, device_model_version
  * end_effector_type
  * operation_platform_height
  * total_episodes, converted_episodes
  * 各种状态字段

- 多对多关系表（一对多）：
  * scene_types (SceneTypeDB): scene_name
  * task_descriptions (TaskDescriptionDB): task_description
  * atomic_actions (AtomicActionDB): action_name
  * objects (ObjectDB): object_name, level1-5_category

注意：
1. 所有标记为 "auto_generated" 的字段会在生成过程中自动填充
2. 模板固定值可以根据需要手动修改模板文件
3. 从 meta/info.json 提取的字段依赖于格式转换过程是否正确写入
4. 数据库信息目前未直接用于 README 生成，但可以用于数据集管理
"""
