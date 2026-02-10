"""
统一元数据（Unified Metadata）。

这个模块只做一件事：定义 `UnifiedMetadata` 这一份“对外展示的字段契约”。
它是网页侧（导出的 `*.yml`）和 Hub README 侧（Jinja2 模板渲染）的共同输入，
从而保证“页面展示”和“README 描述”不会因为两套逻辑而漂移。

它在哪里被如何使用（最重要的调用链）：
- **聚合构建**：`prepare_metadata/metadata_collect.py:create_unified_metadata()`
  会把 DB / YAML / meta/ / annotations/ / 文件系统派生字段合并后填充到本结构。
- **服务封装**：`prepare_metadata/metadata_service.py:MetadataSyncService`
  把上面的构建流程封装为“收集 + 落盘”的服务接口，供页面同步与上传流程复用。
- **README 渲染**：`hub_upload/gen_readme/*` 会把 `UnifiedMetadata.to_dict()` 的结果作为模板 context。

设计取向（非常关键，避免误解）：
- `UnifiedMetadata` 是“数据容器”，不是“校验器/纠错器”。字段内容对不对主要由上游数据源决定。
- `update()` 会忽略不存在的字段：这是为了兼容历史字段名、以及不同流程的可选字段集合。
"""
from dataclasses import asdict, dataclass, field
from typing import Any

import yaml


@dataclass
class UnifiedMetadata:
    """统一的数据集元数据结构，包含所有来源的字段"""

    # ========== 来源：YAML文件直接字段 ==========
    # 这些字段直接从数据库中给出路径的原始YAML文件读取，可能包含错误

    dataset_name: str = ""
    # 数据来源: YAML文件的 dataset_name 字段
    # 用途: 数据集的基本名称标识
    # 示例: "battery_storage_b"
    ### 可能需要进行修改以包含完整的机器人名称+任务名称，目前不修改

    dataset_uuid: str | None = None
    # 数据来源: YAML文件的 dataset_uuid 字段
    # 用途: 数据集的唯一标识符（当前多数为null）
    # 示例: None (暂未分配UUID)

    # task_descriptions: List[str] = field(default_factory=list)
    # 数据来源: YAML文件的 task_descriptions 字段（数组）
    # 用途: 描述机器人需要执行的具体任务
    # 示例: ["place the batteries in the box on the table."]
    # 处理: 在data-manager.js中直接映射到 description 字段
    ### 废弃：现在直接使用tasks.jsonl中的task字段，因为原本的可能是错误的，
    # 需要使用任务标注的精确结果，请直接读取tasks字段

    scene_type: list[str] = field(default_factory=list)
    # 数据来源: YAML文件的 scene_type 字段（数组）
    # 用途: 标识任务执行的环境场景类型
    # 示例: ["home", "restaurant", "office"]
    # 处理: 在data-manager.js中映射到 scenes 字段
    # 使用: filter-manager.js 的 'scene' 过滤器组

    atomic_actions: list[str] = field(default_factory=list)
    # 数据来源: YAML文件的 atomic_actions 字段（数组）
    # 用途: 任务分解的基本动作单元
    # 示例: ["grasp", "place", "pick"]
    # 处理: 在data-manager.js中映射到 actions 字段
    # 使用: filter-manager.js 的 'action' 过滤器组

    # device_model: str = ""
    # 数据来源: YAML文件的 device_model 字段
    # 用途: 标识执行任务的机器人型号
    # 示例: "AgiBot-g1", "AIRBOT_MMK2", "R1_Lite", "Cobot_Magic"
    # 处理: data-manager.js 映射到 robot 字段
    # 使用: filter-manager.js 的 'robot' 过滤器组
    # 注意: 可能是字符串或字符串数组（支持多机器人）
    ### 现在直接使用robot_type字段

    end_effector_type: list[str] = field(default_factory=list)
    # 数据来源: 数据库或 YAML 的 end_effector_type 字段（以 "/" 分割多个值）
    # 用途: 标识机器人末端执行器的类型，支持多种组合
    # 示例: ["two_finger_gripper"], ["three_finger_hand", "suction_cup"]
    # 处理: data-manager.js 映射到 endEffector 字段
    # 使用: filter-manager.js 的 'end' 过滤器组
    ### 注意!!! yaml里这个信息是错误的，需要从数据库里读取正确值！

    operation_platform_height: float | None = None
    # 数据来源: YAML文件的 operation_platform_height 字段
    # 用途: 操作平台的高度（单位：厘米）
    # 示例: 77.2, None
    # 处理: data-manager.js 映射到 platformHeight 字段
    # 注意: 此参数不用于过滤，仅作为描述性元数据

    objects: list[dict[str, Any]] = field(default_factory=list)
    # 数据来源: YAML文件的 objects 数组
    # 用途: 操作对象列表及其层次分类信息
    # 格式: [{"object_name": str, "level1": str|None, "level2": str|None,
    #         "level3": str|None, "level4": str|None, "level5": str|None}, ...]
    # 示例: [{"object_name": "table", "level1": "furniture", "level2": "table", ...}]
    # 处理: data-manager.js 将 level1-5 合并为 hierarchy 数组
    # 使用: filter-manager.js 的 'object' 层次过滤器
    # 字段说明:
    #   - object_name: 操作对象的具体名称 (如: table, box, battery)
    #   - level1: 对象分类的第一级/最粗粒度 (如: furniture, container)
    #   - level2: 对象分类的第二级/中等粒度 (如: table, box)
    #   - level3-5: 对象分类的更细粒度 (通常为null)

    # ========== 来源：自动生成/计算字段 ==========

    path: str = ""
    # 数据来源: 由文件名或数据集键名生成
    # 计算位置: data-manager.js createDatasetObject() 方法
    # 格式: {robot}_{task_name}
    # 示例: "AgiBot-g1_battery_storage_b"
    # 用途: 作为数据集的唯一标识符和文件路径基础

    video_url: str = ""
    # 数据来源: 根据 path 自动生成
    # 计算位置: data-manager.js line 99
    # 代码逻辑: video_url = `${this.config.paths.videos}/${path}.mp4`
    # 格式: ./assets/videos/{path}.mp4
    # 示例: "./assets/videos/AgiBot-g1_battery_storage_b.mp4"
    # 用途: 视频文件的URL路径

    thumbnail_url: str = ""
    # 数据来源: 根据 path 自动生成
    # 计算位置: data-manager.js line 102
    # 代码逻辑: thumbnail_url = `${this.config.paths.assetsRoot}/thumbnails/${path}.jpg`
    # 格式: ./assets/thumbnails/{path}.jpg
    # 示例: "./assets/thumbnails/AgiBot-g1_battery_storage_b.jpg"
    # 用途: 缩略图的URL路径
    # 注意: 缩略图必须预先存在，不会自动生成

    # ========== 来源：模板固定值（README生成用） ==========
    # 这些字段来自 dataset_info.yml 模板或配置文件

    license: str = "apache-2.0"
    # 数据来源: 模板固定值
    # 用途: 数据集的开源许可证类型
    # 示例: "apache-2.0"

    language: list[str] = field(default_factory=lambda: ["en", "zh"])
    # 数据来源: 模板固定值
    # 用途: 数据集支持的语言列表
    # 示例: ["en", "zh"]

    task_categories: list[str] = field(default_factory=lambda: ["robotics"])
    # 数据来源: 模板固定值
    # 用途: 任务分类标签
    # 示例: ["robotics"]

    tags: list[str] = field(default_factory=lambda: ["RoboCOIN", "LeRobot"])
    # 数据来源: 模板固定值 + 可选的自定义标签
    # 用途: 数据集的标签列表，用于分类和检索
    # 示例: ["RoboCoin", "LeRobot"]
    # 注意: 如果配置了 task_tags_yamls_dir，会从 {dataset_name}.yml 额外添加标签

    frame_range: str = "100K-1M"
    # 数据来源: 根据 total_frames 自动计算
    # 用途: 数据集帧数范围标签
    # 规则: <1K, 1K-10K, 10K-100K, 100K-1M, 1M-10M, ...
    # 示例: "100K-1M"

    dataset_size: str = ""
    # 数据来源: 根据数据集目录实际计算的文件大小
    # 用途: 数据集的实际文件大小
    # 格式: 人类可读的格式，如 "2.7GB", "234MB"
    # 示例: "1.2GB"

    configs: list[dict[str, str]] = field(default_factory=lambda: [
        {"config_name": "default", "data_files": "data/*/*.parquet"}
    ])
    # 数据来源: 模板固定值
    # 用途: 数据文件配置，定义如何加载数据
    # 格式: [{"config_name": str, "data_files": str}, ...]
    # 示例: [{"config_name": "default", "data_files": "data/*/*.parquet"}]

    authors: dict[str, list[dict[str, str]]] = field(default_factory=lambda: {
        "contributed_by": [{"name": "RoboCOIN", "url": "https://flagopen.github.io/RoboCOIN/", "affiliation": "RoboCOIN Team"}],
        "annotated_by": [{"name": "RoboCOIN", "url": "https://flagopen.github.io/RoboCOIN/", "affiliation": "RoboCOIN Team"}]
    })
    # 数据来源: 模板固定值，可手动修改
    # 用途: 作者信息，包括贡献者和标注者
    # 格式: {"contributed_by": [...], "annotated_by": [...]}
    # 字段说明:
    #   - name: 作者名称 (必需)
    #   - url: 作者链接 (可选)
    #   - affiliation: 作者所属机构 (可选)

    dataset_description: str = "This dataset uses an extended format based on LeRobot and is fully compatible with LeRobot."
    # 数据来源: 模板固定值
    # 用途: 数据集的简短描述

    homepage: str = "https://flagopen.github.io/RoboCOIN/"
    # 数据来源: 模板固定值
    # 用途: 项目主页链接

    paper: str = "https://arxiv.org/abs/2511.17441"
    # 数据来源: 模板固定值（可选）
    # 用途: 相关论文链接

    repository: str = "https://github.com/FlagOpen/RoboCOIN"
    # 数据来源: 模板固定值
    # 用途: 代码仓库链接

    project_page: str = "https://flagopen.github.io/RoboCOIN/"
    # 数据来源: 模板固定值（可选）
    # 用途: 项目页面链接


    issues_url: str = "https://github.com/FlagOpen/RoboCOIN/issues"
    # 数据来源: 模板固定值（可选）
    # 用途: GitHub Issues 链接，用于问题追踪


    # ========== 来源：meta/info.json ==========
    # 这些字段从数据集的 meta/info.json 文件中自动提取

    robot_type: str = ""
    # 数据来源: 通过字符串匹配 prepare_metadata/names.yml 与数据集文件夹名
    #          如果匹配失败，则回退到 meta/info.json["robot_type"]
    # 用途: 机器人类型标识
    # 示例: "G1edu-u3", "RMC-AIDA-L", "AIRBOT_MMK2"
    # 注意: 使用 names.yml 进行一对多映射，直接在文件夹名中查找匹配的设备名称

    codebase_version: str = ""
    # 数据来源: meta/info.json["codebase_version"]
    # 用途: 代码库版本号
    # 示例: "v2.1"

    statistics: dict[str, Any] = field(default_factory=lambda: {
        "total_episodes": 0,
        "total_frames": 0,
        "total_tasks": 0,
        "total_videos": 0,
        "total_chunks": 0,
        "chunks_size": 0,
        "fps": 30
    })
    # 数据来源: meta/info.json 的多个字段
    # 用途: 数据集的统计信息
    # 字段说明:
    #   - total_episodes: 总episode数 (meta/info.json["total_episodes"])
    #   - total_frames: 总帧数 (meta/info.json["total_frames"])
    #   - total_tasks: 总任务数 (meta/info.json["total_tasks"])
    #   - total_videos: 总视频数 (meta/info.json["total_videos"])
    #   - total_chunks: 总chunk数 (meta/info.json["total_chunks"])
    #   - chunks_size: 每个chunk的大小 (meta/info.json["chunks_size"])
    #   - fps: 帧率 (meta/info.json["fps"])
    # 可选字段 (如果meta/info.json中存在):
    #   - total_duration: 总时长
    #   - video_resolution: 视频分辨率
    #   - state_dim: 状态维度
    #   - action_dim: 动作维度
    #   - camera_views: 相机视角列表

    splits: dict[str, str] = field(default_factory=lambda: {"train": "0:0"})
    # 数据来源: meta/info.json["splits"]
    # 用途: 数据集的训练/验证/测试分割
    # 格式: {"train": "start:end", "val": "start:end", "test": "start:end"}
    # 示例: {"train": "0:89", "val": "89:99", "test": "99:109"}
    # 注意: val 和 test 是可选的

    data_path: str = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    # 数据来源: meta/info.json["data_path"]
    # 用途: 数据文件的路径模式（支持变量占位符）
    # 格式: 包含 {episode_chunk}, {episode_index} 等占位符
    # 示例: "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"

    video_path: str = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    # 数据来源: meta/info.json["video_path"]
    # 用途: 视频文件的路径模式（支持变量占位符）
    # 格式: 包含 {episode_chunk}, {video_key}, {episode_index} 等占位符
    # 示例: "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"

    features: dict[str, Any] = field(default_factory=dict)
    # 数据来源: meta/info.json["features"] (完整复制)
    # 用途: 完整的数据特征定义，包括所有观测和动作字段
    # 包括内容:
    #   - observation.images.*: 各个相机的图像特征
    #     格式: {"dtype": "video", "shape": [h,w,c], "names": [...], "info": {...}}
    #   - observation.state: 状态向量特征
    #     格式: {"dtype": "float32", "shape": [dim], "names": [...]}
    #   - action: 动作向量特征
    #     格式: {"dtype": "float32", "shape": [dim], "names": [...]}
    #   - timestamp, frame_index, episode_index, index, task_index: 时间和索引字段
    #   - 各种标注字段: subtask_annotation, scene_annotation 等
    #   - 运动特征字段: eef_*, gripper_* 等

    # ========== 来源：meta/tasks.jsonl ==========

    tasks: str = ""
    # 数据来源: meta/tasks.jsonl 文件
    # 用途: 任务描述文本
    # 提取方式: 读取所有行的 task 字段，用换行符连接
    # 示例: "pick up the apple from the table and place it into the basket."

    # ========== 来源：annotations/ ==========

    sub_tasks: list[str] = field(default_factory=list)
    # 数据来源: annotations/subtask_annotations.jsonl
    # 用途: 子任务列表（去重后）
    # 提取方式:
    #   1. 读取 annotations/subtask_annotations.jsonl
    #   2. 提取所有 "subtask" 字段
    #   3. 去重（不区分大小写）
    #   4. 排序并返回列表
    # 示例: ["End", "Grasp the apple with the left gripper",
    #        "Place the apple into the basket with the left gripper", "Static"]

    annotations: dict[str, str] = field(default_factory=lambda: {
        "subtask_annotation": "auto_generated",
        "scene_annotation": "auto_generated",
        "eef_direction": "auto_generated",
        "eef_velocity": "auto_generated",
        "eef_acc_mag": "auto_generated",
        "gripper_mode": "auto_generated",
        "gripper_activity": "auto_generated"
    })
    # 数据来源: 模板固定值，表示存在哪些标注
    # 用途: 声明数据集包含的标注类型
    # 字段说明:
    #   - subtask_annotation: 子任务标注
    #   - scene_annotation: 场景标注
    #   - eef_direction: 末端执行器方向
    #   - eef_velocity: 末端执行器速度
    #   - eef_acc_mag: 末端执行器加速度幅值
    #   - gripper_mode: 夹爪模式
    #   - gripper_activity: 夹爪活动状态
    # 注意: "auto_generated" 表示该标注会自动生成

    # ========== 来源：模板或自动生成 ==========

    cameras: list[dict[str, Any]] | str = "auto_generated"
    # 数据来源: 模板或自动生成
    # 用途: 相机数量信息
    # 计算方式: 从 features 中统计 observation.images.* 字段数量
    # 示例: "3" 或 "auto_generated"

    observation_space: dict[str, Any] = field(default_factory=lambda: {
        "images": "auto_generated",
        "state": "auto_generated"
    })
    # 数据来源: 从 features 自动提取
    # 用途: 观测空间定义
    # 字段说明:
    #   - images: 图像观测列表 (从 observation.images.* 提取)
    #   - state: 状态观测维度 (从 observation.state 提取)

    action_space: dict[str, Any] | str = "auto_generated"
    # 数据来源: 从 features 自动提取
    # 用途: 动作空间定义
    # 计算方式: 从 features["action"] 提取 shape 和 names

    eef_sim_pose: str = "auto_generated"
    # 数据来源: 模板固定值
    # 用途: 表示数据集包含末端执行器仿真位姿特征
    # 说明: "auto_generated" 表示该特征会自动计算

    gripper_open_scale: str = "auto_generated"
    # 数据来源: 模板固定值
    # 用途: 表示数据集包含夹爪开合比例特征
    # 说明: "auto_generated" 表示该特征会自动计算

    depth_enabled: bool = False
    # 数据来源: 从 features 计算生成
    # 用途: 标识数据集是否包含深度图
    # 计算方式: 检查 features 中所有 observation.images.* 字段
    #          如果任一 info["video.is_depth_map"] 为 True，则此字段为 True
    # 示例: True 或 False

    data_schema: str = "auto_generated"
    # 数据来源: 模板或手动添加
    # 用途: 数据模式的文本描述（可选）

    structure: str = "auto_generated"
    # 数据来源: 自动遍历数据集目录生成
    # 用途: 数据集目录结构的树状表示
    # 生成方式: generate_folder_structure(dataset_path, max_files_per_dir=5)
    # 说明: 显示叶子目录及其前5个文件

    # ========== 来源：模板固定值（可选字段） ==========

    contact_email: str | None = None
    # 数据来源: 模板固定值（可选）
    # 用途: 联系邮箱
    # 示例: "contact@robocoin.ai"

    contact_info: str = "For questions, issues, or feedback regarding this dataset, please contact us."
    # 数据来源: 模板固定值（可选）
    # 用途: 联系信息说明

    support_info: str = "For technical support, please open an issue on our GitHub repository."
    # 数据来源: 模板固定值（可选）
    # 用途: 技术支持信息

    license_details: str = "Please refer to the LICENSE file for full license terms and conditions."
    # 数据来源: 模板固定值（可选）
    # 用途: 许可证详细说明

    citation_bibtex: str | None = """@article{robocoin,
    title={RoboCOIN: An Open-Sourced Bimanual Robotic Data Collection for Integrated Manipulation},
    author={Shihan Wu, Xuecheng Liu, Shaoxuan Xie, Pengwei Wang, Xinghang Li, Bowen Yang, Zhe Li, Kai Zhu, Hongyu Wu, Yiheng Liu, Zhaoye Long, Yue Wang, Chong Liu, Dihan Wang, Ziqiang Ni, Xiang Yang, You Liu, Ruoxuan Feng, Runtian Xu, Lei Zhang, Denghang Huang, Chenghao Jin, Anlan Yin, Xinlong Wang, Zhenguo Sun, Junkai Zhao, Mengfei Du, Mingyu Cao, Xiansheng Chen, Hongyang Cheng, Xiaojie Zhang, Yankai Fu, Ning Chen, Cheng Chi, Sixiang Chen, Huaihai Lyu, Xiaoshuai Hao, Yequan Wang, Bo Lei, Dong Liu, Xi Yang, Yance Jiao, Tengfei Pan, Yunyan Zhang, Songjing Wang, Ziqian Zhang, Xu Liu, Ji Zhang, Caowei Meng, Zhizheng Zhang, Jiyang Gao, Song Wang, Xiaokun Leng, Zhiqiang Xie, Zhenzhen Zhou, Peng Huang, Wu Yang, Yandong Guo, Yichao Zhu, Suibing Zheng, Hao Cheng, Xinmin Ding, Yang Yue, Huanqian Wang, Chi Chen, Jingrui Pang, YuXi Qian, Haoran Geng, Lianli Gao, Haiyuan Li, Bin Fang, Gao Huang, Yaodong Yang, Hao Dong, He Wang, Hang Zhao, Yadong Mu, Di Hu, Hao Zhao, Tiejun Huang, Shanghang Zhang, Yonghua Lin, Zhongyuan Wang and Guocai Yao},
    journal={arXiv preprint arXiv:2511.17441},
    url = {https://arxiv.org/abs/2511.17441},
    year={2025}
    }"""
    # 数据来源: 模板固定值，可手动添加
    # 用途: BibTeX 格式的引用信息

    additional_citations: str = "If you use this dataset, please also consider citing:\n- LeRobot Framework: https://github.com/huggingface/lerobot"
    # 数据来源: 模板固定值（可选）
    # 用途: 额外的引用说明

    version_info: str = "## Version History\n- v1.0.0 (2025-11): Initial release"
    # 数据来源: 模板固定值（可选）
    # 用途: 版本历史信息

    # ========== 来源：模板固定值（数据集访问控制） ==========
    # 这些字段用于HuggingFace Hub的gated dataset功能

    extra_gated_prompt: str = "By accessing this dataset, you agree to cite the associated paper in your research/publications—see the ''Citation'' section for details. You agree to not use the dataset to conduct experiments that cause harm to human subjects."
    # 数据来源: 模板固定值
    # 用途: 数据集访问时的提示信息，用于告知用户数据集使用条款
    # 示例: "You agree to not use the dataset to conduct experiments that cause harm to human subjects."

    extra_gated_fields: dict[str, dict[str, str]] = field(default_factory=lambda: {
        "Country": {
            "type": "country",
            "description": "e.g., ''Germany'', ''China'', ''United States''"
        },
        "Company/Organization": {
            "type": "text",
            "description": "e.g., ''ETH Zurich'', ''Boston Dynamics'', ''Independent Researcher''"
        }
    })
    # 数据来源: 模板固定值
    # 用途: 数据集访问时需要用户填写的字段，用于收集用户信息和使用目的
    # 格式: {field_name: {"type": field_type, "description": field_description}}
    # 字段说明:
    #   - type: 字段类型，可以是 "text", "country", "checkbox" 等
    #   - description: 字段描述，用于向用户解释该字段的用途和示例
    # 注意: 这些字段会在HuggingFace Hub的数据集页面上展示为表单

    # ========== 原始数据保留 ==========

    raw: dict[str, Any] = field(default_factory=dict)
    # 数据来源: 原始YAML解析后的完整对象
    # 计算位置: data-manager.js line 120
    # 用途: 保留原始数据用于调试和扩展
    # 说明: 存储未经处理的原始元数据，便于回溯和调试

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_yaml(self, file_path: str = None) -> str:
        data = self.to_dict()
        yaml_str = yaml.dump(data, allow_unicode=True, sort_keys=False, default_flow_style=False)

        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(yaml_str)

        return yaml_str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'UnifiedMetadata':
        return cls(**data)

    @classmethod
    def from_yaml(cls, file_path: str) -> 'UnifiedMetadata':
        with open(file_path, encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def update(self, **kwargs: Any) -> None:  # noqa: ANN003, ANN401
        """
        批量更新字段值
        参数:
            **kwargs: 要更新的字段名和值
        示例: metadata.update(dataset_name="new_name", fps=30)
        说明: 只会更新已存在的字段，忽略不存在的字段
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:  # noqa: ANN401
        return getattr(self, key, default)


# ========== 字段分组索引（便于按来源访问） ==========

FIELD_GROUPS = {
    # 从YAML文件直接读取的字段（可能包含错误）
    "yaml_direct": [
        "dataset_name", "dataset_uuid", "task_descriptions", "scene_type",
        "atomic_actions", "device_model", "end_effector_type",
        "operation_platform_height", "objects"
    ],
    # 自动生成的字段（根据其他字段计算）
    "auto_generated": [
        "path", "video_url", "thumbnail_url", "dataset_size"
    ],
    # 模板固定值字段（从配置文件或模板获取）
    "template_fixed": [
        "license", "language", "task_categories", "tags", "frame_range",
        "configs", "authors", "dataset_description", "homepage", "paper",
        "repository", "project_page", "contact_email", "contact_info",
        "support_info", "license_details", "citation_bibtex",
        "additional_citations", "version_info", "extra_gated_prompt",
        "extra_gated_fields"
    ],
    # 从 meta/info.json 提取的字段
    "meta_info_json": [
        "robot_type", "codebase_version", "statistics", "splits",
        "data_path", "video_path", "features"
    ],
    # 从 meta/tasks.jsonl 提取的字段
    "meta_tasks_jsonl": [
        "tasks"
    ],
    # 从 annotations/ 目录提取的字段
    "annotations": [
        "sub_tasks", "annotations"
    ],
    # 计算生成的字段（需要额外处理）
    "computed": [
        "cameras", "observation_space", "action_space", "eef_sim_pose",
        "gripper_open_scale", "depth_enabled", "data_schema", "structure"
    ]
}

# NOTE:
# - FIELD_GROUPS 目前主要用于“按来源解释字段”的索引（写注释/排查问题更直观），并不参与运行时逻辑。
# - 其中有少量历史字段名（如 task_descriptions/device_model）在 dataclass 中可能不存在；
#   这是刻意保留的兼容痕迹，方便阅读旧配置或旧脚本时对照理解。不要在运行时代码里硬依赖它们。
