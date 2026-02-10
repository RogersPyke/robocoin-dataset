{
    # ========================================
    # 来源：YAML文件直接字段, 这里的YAML文件是数据库中给出了路径的原始文件里的YAML,包含有可能的大量错误
    # ========================================

    "dataset_name": "string",
    # 数据来源：YAML文件的 dataset_name 字段
    # 用途：数据集的基本名称标识
    # 示例：battery_storage_b

    "dataset_uuid": "string | null",
    # 数据来源：YAML文件的 dataset_uuid 字段
    # 用途：数据集的唯一标识符（当前多数为null）
    # 示例：null（暂未分配UUID）

    "task_descriptions": ["string"],
    # 数据来源：YAML文件的 task_descriptions 字段（数组）
    # 用途：描述机器人需要执行的具体任务
    # 示例：["place the batteries in the box on the table."]
    # 处理：在data-manager.js中直接映射到 description 字段

    "scene_type": ["string"],
    # 数据来源：YAML文件的 scene_type 字段（数组）
    # 用途：标识任务执行的环境场景类型
    # 示例：["home", "restaurant", "office"]
    # 处理：在data-manager.js中映射到 scenes 字段
    # 使用：filter-manager.js 的 'scene' 过滤器组

    "atomic_actions": ["string"],
    # 数据来源：YAML文件的 atomic_actions 字段（数组）
    # 用途：任务分解的基本动作单元
    # 示例：["grasp", "place", "pick"]
    # 处理：在data-manager.js中映射到 actions 字段
    # 使用：filter-manager.js 的 'action' 过滤器组

    "objects": [
        {
            "object_name": "string",
            # 数据来源：YAML文件 objects 数组中的 object_name 字段
            # 用途：操作对象的具体名称
            # 示例：table, box, battery
            # 处理：data-manager.js 映射到 objects[].name

            "level1": "string | null",
            # 数据来源：YAML文件 objects 数组中的 level1 字段
            # 用途：对象分类的第一级（最粗粒度）
            # 示例：furniture, container, electronic_products
            # 处理：data-manager.js 将 level1-5 合并为 hierarchy 数组
            # 使用：filter-manager.js 的 'object' 层次过滤器

            "level2": "string | null",
            # 数据来源：YAML文件 objects 数组中的 level2 字段
            # 用途：对象分类的第二级（中等粒度）
            # 示例：table, box, battery
            # 处理：data-manager.js 将 level1-5 合并为 hierarchy 数组

            "level3": "string | null",
            # 数据来源：YAML文件 objects 数组中的 level3 字段
            # 用途：对象分类的第三级（细粒度）
            # 示例：通常为null，某些对象可能有更细分类

            "level4": "string | null",
            # 数据来源：YAML文件 objects 数组中的 level4 字段
            # 用途：对象分类的第四级（更细粒度）
            # 示例：通常为null

            "level5": "string | null"
            # 数据来源：YAML文件 objects 数组中的 level5 字段
            # 用途：对象分类的第五级（最细粒度）
            # 示例：通常为null
        }
    ],
    # 转换处理（data-manager.js line 106-116）：
    # objects: (raw.objects || []).map(obj => ({
    #     name: obj.object_name,
    #     hierarchy: [obj.level1, obj.level2, obj.level3, obj.level4, obj.level5]
    #                .filter(level => level !== null && level !== undefined),
    #     raw: obj
    # }))

    "device_model": "string",
    # 数据来源：YAML文件的 device_model 字段
    # 用途：标识执行任务的机器人型号
    # 示例：AgiBot-g1, AIRBOT_MMK2, R1_Lite, Cobot_Magic
    # 处理：data-manager.js 映射到 robot 字段
    # 使用：filter-manager.js 的 'robot' 过滤器组
    # 注意：可能是字符串或字符串数组（支持多机器人）

    "end_effector_type": "string | string[]",
    # 数据来源：数据库或 YAML 的 end_effector_type 字段（数据库会返回 "/" 分隔值）
    # 用途：标识机器人末端执行器的类型，可以包含多个
    # 示例：two_finger_gripper, [three_finger_hand, suction_cup]
    # 处理：data-manager.js 映射到 endEffector 字段
    # 使用：filter-manager.js 的 'end' 过滤器组

    "operation_platform_height": "number | null",
    # 数据来源：YAML文件的 operation_platform_height 字段
    # 用途：操作平台的高度（单位：厘米）
    # 示例：77.2, null
    # 处理：data-manager.js 映射到 platformHeight 字段
    # 注意：此参数不用于过滤，仅作为描述性元数据


    # ========================================
    # 来源：自动生成/计算字段
    # ========================================

    "path": "string",
    # 数据来源：由文件名或数据集键名生成
    # 计算位置：data-manager.js createDatasetObject() 方法
    # 格式：{robot}_{task_name}
    # 示例：AgiBot-g1_battery_storage_b
    # 用途：作为数据集的唯一标识符和文件路径基础

    "name": "string",
    # 数据来源：优先使用 path，否则使用 dataset_name
    # 计算位置：data-manager.js line 98
    # 代码：name: path || raw.dataset_name
    # 用途：在UI中显示的数据集名称

    "video_url": "string",
    # 数据来源：根据 path 自动生成
    # 计算位置：data-manager.js line 99
    # 代码：video_url: `${this.config.paths.videos}/${path}.mp4`
    # 格式：./assets/videos/{path}.mp4
    # 示例：./assets/videos/AgiBot-g1_battery_storage_b.mp4
    # 用途：视频文件的URL路径

    "thumbnail_url": "string",
    # 数据来源：根据 path 自动生成
    # 计算位置：data-manager.js line 102
    # 代码：thumbnail_url: `${this.config.paths.assetsRoot}/thumbnails/${path}.jpg`
    # 格式：./assets/thumbnails/{path}.jpg
    # 示例：./assets/thumbnails/AgiBot-g1_battery_storage_b.jpg
    # 用途：缩略图的URL路径
    # 注意：缩略图必须预先存在，不会自动生成

    "raw": "Object",
    # 数据来源：原始YAML解析后的完整对象
    # 计算位置：data-manager.js line 120
    # 用途：保留原始数据用于调试和扩展


    # ========================================
    # 来源：配置文件
    # ========================================
    # 路径配置来自 config.js (modules/config.js)

    "config.paths.assetsRoot": "./assets",
    # 定义位置：config.js 或自动检测
    # 用途：资源文件的根目录

    "config.paths.info": "./assets/info",
    # 定义位置：config.js
    # 用途：JSON索引文件目录

    "config.paths.datasetInfo": "./assets/dataset_info",
    # 定义位置：config.js
    # 用途：YAML元数据文件目录

    "config.paths.videos": "./assets/videos",
    # 定义位置：config.js
    # 用途：视频文件目录


    # ========================================
    # 来源：data_index.json 和 consolidated_datasets.json
    # ========================================

    "data_index.json": {
        "datasets": ["string"],
        # 文件位置：./assets/info/data_index.json
        # 内容：所有数据集的path列表
        # 用途：快速获取所有数据集名称，fallback模式使用
        # 示例：["AIRBOT_MMK2_clean_the_desktop_a", "AgiBot-g1_battery_storage_b", ...]
    },

    "consolidated_datasets.json": {
        "{path}": {
            "dataset_name": "...",
            "scene_type": [...],
            # ... 所有YAML字段
        }
    }
    # 文件位置：./assets/info/consolidated_datasets.json
    # 内容：所有YAML文件合并后的JSON对象
    # 用途：优化加载速度，一次请求获取所有元数据
    # 生成方式：通过脚本（如opti_init.py）预处理生成
    # 优势：避免268次独立的YAML文件请求
}
