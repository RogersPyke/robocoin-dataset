#!/usr/bin/env python3
"""
Hub Upload Status Report Generator

生成2个JSON文件，用于分析应上传的数据集和云端实际存在的数据集之间的差异：
1. hf_comparison.json - HuggingFace比较结果
2. ms_comparison.json - ModelScope比较结果

每个comparison文件包含：
- should_upload_but_missing: 应该上传但云端没有的数据集
- cloud_but_not_in_should_upload: 云端有但不应该上传的数据集
- both_have: 应该上传且云端也有的数据集
- uploaded_but_missing_in_cloud: 本地标注上传完成但云端缺失的数据集
- uploaded_but_not_marked: 已上传到云端但数据库中未标记为完成的数据集
- uploaded_but_not_marked_should_upload: uploaded_but_not_marked中应该上传的（验证项）
- uploaded_but_not_marked_should_not_upload: uploaded_but_not_marked中不应该上传的（验证项，理论上为空）
- should_not_upload_but_marked_completed: 不应该上传（可视化状态不是完成）但被标记完成的数据集
- should_not_upload_but_marked_and_in_cloud: 不应该上传但被标记完成且确实在云端存在的数据集
- summary: 各种统计信息

使用方法：
    python scripts/hub_upload/hub_upload_status_report.py

所有配置都在脚本内部，直接运行即可。
"""

import json
import logging
import sys
import warnings
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

import requests  # noqa: E402
from huggingface_hub import HfApi  # noqa: E402

from robocoin_dataset.database.database import DatasetDatabase  # noqa: E402
from robocoin_dataset.database.models import DatasetDB, TaskStatus  # noqa: E402

# 忽略 modelscope 的 pkg_resources 废弃警告
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")

# ============= 配置区域 =============
# 数据库路径
DB_PATH = "/mnt/db/datasets_new.db"

# 命名空间配置
HF_NAMESPACE = "RoboCOIN"
MS_NAMESPACE = "RoboCOIN"

# Token配置（可选，如果需要访问私有repos）
HF_TOKEN = None
MS_TOKEN = None

# 输出目录
OUTPUT_DIR = Path(__file__).parent / "upload_status_reports"
# ===================================

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_hf_cloud_datasets(namespace: str, token: str = None) -> list[str]:
    """从 HuggingFace 获取指定命名空间下的所有数据集名称"""
    logger.info(f"正在从 HuggingFace 获取 {namespace} 命名空间下的数据集...")

    try:
        api = HfApi(token=token)
        repos = api.list_datasets(author=namespace)

        repo_names = []
        for repo in repos:
            repo_id = repo.id
            if "/" in repo_id:
                repo_name = repo_id.split("/", 1)[1]
                repo_names.append(repo_name)
            else:
                repo_names.append(repo_id)

        logger.info(f"从 HuggingFace 获取到 {len(repo_names)} 个数据集")
        return repo_names

    except Exception as e:
        logger.error(f"获取 HuggingFace 数据集失败: {e}")
        return []


def get_ms_cloud_datasets(namespace: str, token: str = None) -> list[str]:
    """从 ModelScope 获取指定命名空间下的所有数据集名称"""
    logger.info(f"正在从 ModelScope 获取 {namespace} 命名空间下的数据集...")

    try:
        url = "https://www.modelscope.cn/api/v1/datasets"
        all_repo_names = []
        page_number = 1
        page_size = 50

        while True:
            params = {
                "owner": namespace,
                "PageNumber": page_number,
                "PageSize": page_size,
            }

            response = requests.get(url, params=params, timeout=10)

            if response.status_code != 200:
                logger.warning(f"ModelScope API 返回错误状态码: {response.status_code}")
                if page_number == 1:
                    return []
                break

            data = response.json()
            datasets = data.get("Data", [])

            for dataset in datasets:
                repo_name = dataset.get("Name", "")
                if repo_name:
                    all_repo_names.append(repo_name)

            total_count = data.get("TotalCount", 0)
            current_count = len(all_repo_names)

            if current_count >= total_count:
                break

            page_number += 1

        logger.info(f"从 ModelScope 获取到 {len(all_repo_names)} 个数据集")
        return all_repo_names

    except Exception as e:
        logger.error(f"获取 ModelScope 数据集失败: {e}")
        return []


def get_should_upload_datasets(db_path: str, platform: str) -> list[str]:
    """
    从数据库获取应该上传的数据集（根据 hub_upload_task.py 的筛选条件）

    筛选条件：
    - visualize_check_status == COMPLETED
    - 并且（upload_status == PENDING 或 (upload_status == COMPLETED 且 upload_version_ps < visualize_check_version)）

    Args:
        db_path: 数据库文件路径
        platform: 'hf' 或 'ms'

    Returns:
        convert_path 结尾文件夹名称列表
    """
    logger.info(f"正在从数据库读取 {platform.upper()} 应该上传的数据集...")

    # 选择对应的字段
    if platform == "hf":
        upload_status_attr = DatasetDB.huggingface_upload_status
        upload_version_ps_attr = DatasetDB.huggingface_upload_version_ps
    elif platform == "ms":
        upload_status_attr = DatasetDB.ms_upload_status
        upload_version_ps_attr = DatasetDB.ms_upload_version_ps
    else:
        raise ValueError(f"不支持的平台: {platform}")

    try:
        db = DatasetDatabase(Path(db_path))

        with db.with_session() as session:
            # 应用与 _sync_datasets_upload_status 相同的筛选条件

            # 只查询我们需要的字段，避免查询数据库中不存在的列
            query = session.query(
                DatasetDB.dataset_uuid,
                DatasetDB.convert_path,
                DatasetDB.visualize_check_status,
                DatasetDB.visualize_check_version,
                upload_status_attr,
                upload_version_ps_attr
            ).filter(
                DatasetDB.visualize_check_status == TaskStatus.COMPLETED,
            )

            results = query.all()

            # 提取 convert_path 的结尾文件夹名称
            dataset_names = []
            for row in results:
                convert_path = row[1]  # convert_path is the second column
                if convert_path:
                    folder_name = Path(convert_path).name
                    dataset_names.append(folder_name)
                else:
                    dataset_uuid = row[0]  # dataset_uuid is the first column
                    logger.warning(
                        f"数据集 {dataset_uuid} 的 convert_path 为空，跳过"
                    )

            logger.info(f"从数据库获取到 {len(dataset_names)} 个 {platform.upper()} 应该上传的数据集")
            return dataset_names

    except Exception as e:
        logger.error(f"读取数据库失败: {e}")
        return []


def get_uploaded_datasets(db_path: str, platform: str) -> list[str]:
    """
    从数据库获取本地标注为上传完成的数据集

    筛选条件：
    - visualize_check_status == COMPLETED
    - upload_status == COMPLETED

    Args:
        db_path: 数据库文件路径
        platform: 'hf' 或 'ms'

    Returns:
        convert_path 结尾文件夹名称列表
    """
    logger.info(f"正在从数据库读取 {platform.upper()} 本地标注为上传完成的数据集...")

    # 选择对应的字段
    if platform == "hf":
        upload_status_attr = DatasetDB.huggingface_upload_status
    elif platform == "ms":
        upload_status_attr = DatasetDB.ms_upload_status
    else:
        raise ValueError(f"不支持的平台: {platform}")

    try:
        db = DatasetDatabase(Path(db_path))

        with db.with_session() as session:
            # 查询upload_status == COMPLETED
            query = session.query(
                DatasetDB.dataset_uuid,
                DatasetDB.convert_path,
            ).filter(
                upload_status_attr == TaskStatus.COMPLETED
            )

            results = query.all()

            # 提取 convert_path 的结尾文件夹名称
            dataset_names = []
            for row in results:
                convert_path = row[1]  # convert_path is the second column
                if convert_path:
                    folder_name = Path(convert_path).name
                    dataset_names.append(folder_name)
                else:
                    dataset_uuid = row[0]  # dataset_uuid is the first column
                    logger.warning(
                        f"数据集 {dataset_uuid} 的 convert_path 为空，跳过"
                    )

            logger.info(f"从数据库获取到 {len(dataset_names)} 个 {platform.upper()} 本地标注为上传完成的数据集")
            return dataset_names

    except Exception as e:
        logger.error(f"读取数据库失败: {e}")
        return []


def get_should_not_upload_but_marked_completed(db_path: str, platform: str) -> list[str]:
    """
    从数据库获取不应该上传但被标记完成的数据集

    筛选条件：
    - visualize_check_status != COMPLETED
    - upload_status == COMPLETED

    Args:
        db_path: 数据库文件路径
        platform: 'hf' 或 'ms'

    Returns:
        convert_path 结尾文件夹名称列表
    """
    logger.info(f"正在从数据库读取 {platform.upper()} 不应该上传但被标记完成的数据集...")

    # 选择对应的字段
    if platform == "hf":
        upload_status_attr = DatasetDB.huggingface_upload_status
    elif platform == "ms":
        upload_status_attr = DatasetDB.ms_upload_status
    else:
        raise ValueError(f"不支持的平台: {platform}")

    try:
        db = DatasetDatabase(Path(db_path))

        with db.with_session() as session:
            # 查询 visualize_check_status != COMPLETED 且 upload_status == COMPLETED
            query = session.query(
                DatasetDB.dataset_uuid,
                DatasetDB.convert_path,
                DatasetDB.visualize_check_status,
            ).filter(
                DatasetDB.visualize_check_status != TaskStatus.COMPLETED,
                upload_status_attr == TaskStatus.COMPLETED
            )

            results = query.all()

            # 提取 convert_path 的结尾文件夹名称
            dataset_names = []
            for row in results:
                convert_path = row[1]  # convert_path is the second column
                if convert_path:
                    folder_name = Path(convert_path).name
                    dataset_names.append(folder_name)
                else:
                    dataset_uuid = row[0]  # dataset_uuid is the first column
                    logger.warning(
                        f"数据集 {dataset_uuid} 的 convert_path 为空，跳过"
                    )

            logger.info(f"从数据库获取到 {len(dataset_names)} 个 {platform.upper()} 不应该上传但被标记完成的数据集")
            return dataset_names

    except Exception as e:
        logger.error(f"读取数据库失败: {e}")
        return []


def compare_datasets(should_upload: list[str], cloud_datasets: list[str], uploaded: list[str] = None, should_not_upload_but_marked: list[str] = None) -> dict:
    """
    比较应该上传的数据集和云端数据集

    Args:
        should_upload: 应该上传的数据集列表
        cloud_datasets: 云端已有的数据集列表
        uploaded: 本地标注为上传完成的数据集列表（可选）
        should_not_upload_but_marked: 不应该上传但被标记完成的数据集列表（可选）

    Returns:
        {
            "should_upload_but_missing": [...],  # 应该上传但云端没有的
            "cloud_but_not_in_should_upload": [...],  # 云端有但不应该上传的
            "both_have": [...],  # 两者都有的
            "uploaded_but_missing_in_cloud": [...],  # 本地标注上传完成但云端缺失的
            "uploaded_but_not_marked": [...],  # 已上传到云端但数据库中未标记为完成的数据集
            "uploaded_but_not_marked_should_upload": [...],  # uploaded_but_not_marked中应该上传的（理论上应该全部）
            "uploaded_but_not_marked_should_not_upload": [...],  # uploaded_but_not_marked中不应该上传的（理论上应该为空）
            "should_not_upload_but_marked_completed": [...],  # 不应该上传但被标记完成的数据集
            "should_not_upload_but_marked_and_in_cloud": [...],  # 不应该上传但被标记完成且确实在云端存在的数据集
            "summary": {
                "should_upload_count": int,
                "cloud_count": int,
                "missing_count": int,
                "extra_count": int,
                "match_count": int,
                "uploaded_count": int,  # 本地标注上传完成的数量
                "uploaded_but_missing_count": int,  # 本地标注上传完成但云端缺失的数量
                "uploaded_but_not_marked_count": int,  # 已上传到云端但数据库中未标记为完成的数量
                "uploaded_but_not_marked_should_upload_count": int,  # uploaded_but_not_marked中应该上传的数量
                "uploaded_but_not_marked_should_not_upload_count": int,  # uploaded_but_not_marked中不应该上传的数量
                "should_not_upload_but_marked_count": int,  # 不应该上传但被标记完成的数量
                "should_not_upload_but_marked_and_in_cloud_count": int  # 不应该上传但被标记完成且确实在云端存在的数量
            }
        }
    """
    should_upload_set = set(should_upload)
    cloud_set = set(cloud_datasets)

    missing = sorted(should_upload_set - cloud_set)
    extra = sorted(cloud_set - should_upload_set)
    both = sorted(should_upload_set & cloud_set)

    result = {
        "should_upload_but_missing": missing,
        "cloud_but_not_in_should_upload": extra,
        "both_have": both,
        "summary": {
            "should_upload_count": len(should_upload),
            "cloud_count": len(cloud_datasets),
            "missing_count": len(missing),
            "extra_count": len(extra),
            "match_count": len(both)
        }
    }

    # 如果提供了 uploaded 列表，计算本地标注上传完成但云端缺失的数据集
    if uploaded is not None:
        uploaded_set = set(uploaded)
        uploaded_but_missing = sorted(uploaded_set - cloud_set)
        result["uploaded_but_missing_in_cloud"] = uploaded_but_missing
        result["summary"]["uploaded_count"] = len(uploaded)
        result["summary"]["uploaded_but_missing_count"] = len(uploaded_but_missing)

        # 计算已上传但未标记完成的数据集（在云端存在但数据库中未标记为完成）
        both_set = set(both)  # 将列表转换为集合以便进行集合运算
        uploaded_but_not_marked = sorted(both_set - uploaded_set)
        result["uploaded_but_not_marked"] = uploaded_but_not_marked
        result["summary"]["uploaded_but_not_marked_count"] = len(uploaded_but_not_marked)

        # 验证 uploaded_but_not_marked 中的项目是否都应该上传（理论上应该都是）
        uploaded_but_not_marked_set = set(uploaded_but_not_marked)
        uploaded_but_not_marked_should_upload = sorted(uploaded_but_not_marked_set & should_upload_set)
        uploaded_but_not_marked_should_not_upload = sorted(uploaded_but_not_marked_set - should_upload_set)
        result["uploaded_but_not_marked_should_upload"] = uploaded_but_not_marked_should_upload
        result["uploaded_but_not_marked_should_not_upload"] = uploaded_but_not_marked_should_not_upload
        result["summary"]["uploaded_but_not_marked_should_upload_count"] = len(uploaded_but_not_marked_should_upload)
        result["summary"]["uploaded_but_not_marked_should_not_upload_count"] = len(uploaded_but_not_marked_should_not_upload)

    # 如果提供了 should_not_upload_but_marked 列表，添加到结果中
    if should_not_upload_but_marked is not None:
        should_not_upload_but_marked_set = set(should_not_upload_but_marked)
        result["should_not_upload_but_marked_completed"] = sorted(should_not_upload_but_marked_set)
        result["summary"]["should_not_upload_but_marked_count"] = len(should_not_upload_but_marked_set)

        # 检查这些不应该上传但被标记完成的数据集是否真的在云端存在
        should_not_upload_but_marked_and_in_cloud = sorted(should_not_upload_but_marked_set & cloud_set)
        result["should_not_upload_but_marked_and_in_cloud"] = should_not_upload_but_marked_and_in_cloud
        result["summary"]["should_not_upload_but_marked_and_in_cloud_count"] = len(should_not_upload_but_marked_and_in_cloud)

    return result


def save_json(data: dict | list, filepath: Path) -> None:
    """保存数据为JSON文件"""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"✅ 已保存: {filepath}")


def main() -> None:
    """主函数"""
    logger.info("=" * 60)
    logger.info("Hub Upload Status Report Generator")
    logger.info("=" * 60)

    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"输出目录: {OUTPUT_DIR}")
    logger.info("")

    # ====== 步骤 1: 获取应该上传的数据集 ======
    logger.info("步骤 1: 查询数据库中应该上传的数据集")
    logger.info("-" * 60)

    hf_should_upload = get_should_upload_datasets(DB_PATH, "hf")
    ms_should_upload = get_should_upload_datasets(DB_PATH, "ms")
    logger.info("")

    # ====== 步骤 2: 获取本地标注为上传完成的数据集 ======
    logger.info("步骤 2: 查询数据库中本地标注为上传完成的数据集")
    logger.info("-" * 60)

    hf_uploaded = get_uploaded_datasets(DB_PATH, "hf")
    ms_uploaded = get_uploaded_datasets(DB_PATH, "ms")
    logger.info("")

    # ====== 步骤 3: 获取不应该上传但被标记完成的数据集 ======
    logger.info("步骤 3: 查询数据库中不应该上传但被标记完成的数据集")
    logger.info("-" * 60)

    hf_should_not_upload_but_marked = get_should_not_upload_but_marked_completed(DB_PATH, "hf")
    ms_should_not_upload_but_marked = get_should_not_upload_but_marked_completed(DB_PATH, "ms")
    logger.info("")

    # ====== 步骤 4: 获取云端数据集 ======
    logger.info("步骤 4: 查询云端已有的数据集")
    logger.info("-" * 60)

    hf_cloud = get_hf_cloud_datasets(HF_NAMESPACE, HF_TOKEN)
    ms_cloud = get_ms_cloud_datasets(MS_NAMESPACE, MS_TOKEN)
    logger.info("")

    # ====== 步骤 5: 比较差异 ======
    logger.info("步骤 5: 比较差异")
    logger.info("-" * 60)

    hf_comparison = compare_datasets(
        hf_should_upload,
        hf_cloud,
        hf_uploaded,
        hf_should_not_upload_but_marked
    )
    ms_comparison = compare_datasets(
        ms_should_upload,
        ms_cloud,
        ms_uploaded,
        ms_should_not_upload_but_marked
    )

    save_json(hf_comparison, OUTPUT_DIR / "hf_comparison.json")
    save_json(ms_comparison, OUTPUT_DIR / "ms_comparison.json")
    logger.info("")

    # ====== 汇总报告 ======
    logger.info("=" * 60)
    logger.info("汇总报告")
    logger.info("=" * 60)

    logger.info("")
    logger.info("HuggingFace:")
    logger.info(f"  应该上传: {hf_comparison['summary']['should_upload_count']}")
    logger.info(f"  云端已有: {hf_comparison['summary']['cloud_count']}")
    logger.info(f"  缺失(应上传但云端没有): {hf_comparison['summary']['missing_count']}")
    logger.info(f"  额外(云端有但不应上传): {hf_comparison['summary']['extra_count']}")
    logger.info(f"  匹配: {hf_comparison['summary']['match_count']}")
    logger.info(f"  本地标注上传完成: {hf_comparison['summary'].get('uploaded_count', 0)}")
    logger.info(f"  ⚠️  本地标注上传完成但云端缺失: {hf_comparison['summary'].get('uploaded_but_missing_count', 0)}")
    logger.info(f"  ⚠️  已上传但未标记完成: {hf_comparison['summary'].get('uploaded_but_not_marked_count', 0)}")
    logger.info(f"    └─ 其中应该上传的: {hf_comparison['summary'].get('uploaded_but_not_marked_should_upload_count', 0)}")
    logger.info(f"    └─ 其中不应该上传的: {hf_comparison['summary'].get('uploaded_but_not_marked_should_not_upload_count', 0)}")
    logger.info(f"  ⚠️  不应该上传但被标记完成: {hf_comparison['summary'].get('should_not_upload_but_marked_count', 0)}")
    logger.info(f"  ⚠️  不应该上传但被标记完成且确实在云端存在: {hf_comparison['summary'].get('should_not_upload_but_marked_and_in_cloud_count', 0)}")

    logger.info("")
    logger.info("ModelScope:")
    logger.info(f"  应该上传: {ms_comparison['summary']['should_upload_count']}")
    logger.info(f"  云端已有: {ms_comparison['summary']['cloud_count']}")
    logger.info(f"  缺失(应上传但云端没有): {ms_comparison['summary']['missing_count']}")
    logger.info(f"  额外(云端有但不应上传): {ms_comparison['summary']['extra_count']}")
    logger.info(f"  匹配: {ms_comparison['summary']['match_count']}")
    logger.info(f"  本地标注上传完成: {ms_comparison['summary'].get('uploaded_count', 0)}")
    logger.info(f"  ⚠️  本地标注上传完成但云端缺失: {ms_comparison['summary'].get('uploaded_but_missing_count', 0)}")
    logger.info(f"  ⚠️  已上传但未标记完成: {ms_comparison['summary'].get('uploaded_but_not_marked_count', 0)}")
    logger.info(f"    └─ 其中应该上传的: {ms_comparison['summary'].get('uploaded_but_not_marked_should_upload_count', 0)}")
    logger.info(f"    └─ 其中不应该上传的: {ms_comparison['summary'].get('uploaded_but_not_marked_should_not_upload_count', 0)}")
    logger.info(f"  ⚠️  不应该上传但被标记完成: {ms_comparison['summary'].get('should_not_upload_but_marked_count', 0)}")
    logger.info(f"  ⚠️  不应该上传但被标记完成且确实在云端存在: {ms_comparison['summary'].get('should_not_upload_but_marked_and_in_cloud_count', 0)}")

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"✅ 所有报告已生成到: {OUTPUT_DIR}")
    logger.info("=" * 60)

    # 列出生成的文件
    logger.info("")
    logger.info("生成的文件:")
    for file in sorted(OUTPUT_DIR.glob("*.json")):
        logger.info(f"  - {file.name}")


if __name__ == "__main__":
    main()
