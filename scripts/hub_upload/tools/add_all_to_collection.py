"""
命令行入口：将指定命名空间下的全部数据集加入目标 Collection。

本脚本提供了一个命令行工具，用于批量将 Hugging Face Hub 上某个命名空间（namespace/organization）
下的所有数据集添加到指定的 Collection 中。

功能特点：
    - 支持通过命令行参数指定命名空间和目标 Collection
    - 支持从环境变量读取 Hugging Face 访问令牌（HF_TOKEN）
    - 可配置批量处理大小以优化 API 调用
    - 提供详细的日志输出，包括操作进度和结果统计

使用方法：
    python add_all_to_collection.py \\
        --namespace <namespace> \\
        --collection <collection-slug> \\
        [--token <hf-token>] \\
        [--batch-size <size>]

参数说明：
    --namespace: 必需参数，指定要处理的命名空间或组织名称（例如：RoboCOIN）
    --collection: 必需参数，指定目标 Collection 的 slug（例如：org/my-collection）
    --token: 可选参数，Hugging Face 访问令牌。如未提供，将从环境变量 HF_TOKEN 读取
    --batch-size: 可选参数，每次 API 调用处理的最大数据集数量（1-100，默认 100）

环境变量：
    HF_TOKEN: Hugging Face 访问令牌，当未通过 --token 参数提供时使用

退出码：
    0: 操作成功完成
    1: 操作失败（会输出详细的错误信息）

示例：
    # 使用环境变量中的 token
    python add_all_to_collection.py --namespace RoboCOIN --collection RoboCOIN/all-datasets

    # 显式指定 token
    python add_all_to_collection.py \\
        --namespace RoboCOIN \\
        --collection RoboCOIN/all-datasets \\
        --token hf_xxxxxxxxxxxxx

    # 自定义批量大小
    python add_all_to_collection.py \\
        --namespace RoboCOIN \\
        --collection RoboCOIN/all-datasets \\
        --batch-size 50
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from robocoin_dataset.hub_upload.collection.add_collection_utils import (
    add_all_datasets_to_collection,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Add all Hugging Face datasets under a namespace to a specified Collection. "
        "This tool automatically lists all datasets under the specified namespace and adds them "
        "in batches to the target Collection. Supports batch processing to avoid API limits and "
        "provides detailed progress logging.",
    )
    parser.add_argument(
        "--token",
        type=str,
        help="Hugging Face access token with write permissions. "
        "If not provided, the script will attempt to read from the HF_TOKEN environment variable. "
        "If neither is provided, the program will exit with an error.",
    )
    parser.add_argument(
        "--namespace",
        required=True,
        help="Target namespace or organization name. Example: RoboCOIN. "
        "The script will list all datasets under this namespace and add them to the Collection.",
    )
    parser.add_argument(
        "--collection",
        required=True,
        help="Target Collection slug in the format 'org/collection-name'. "
        "Example: RoboCOIN/all-datasets. All found datasets will be added to this Collection.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Maximum number of datasets processed per API call. "
        "Valid range: 1-100 (limited by Hugging Face API). "
        "Default is 100. Larger values can reduce API calls but must not exceed API limits.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    token = args.token or os.getenv("HF_TOKEN")
    if not token:
        parser.error("Must provide --token or set HF_TOKEN environment variable")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    try:
        added = add_all_datasets_to_collection(
            token=token,
            namespace=args.namespace,
            collection_slug=args.collection,
            batch_size=args.batch_size,
        )
        logging.info("Operation completed. Total datasets submitted: %d.", added)
    except Exception as exc:  # pragma: no cover - CLI error path
        logging.exception("Failed to add datasets to collection: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
