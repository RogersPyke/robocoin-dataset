#!/usr/bin/env python3
"""
直接修改 YAML 文件中的 dataset_name 和 device_model 为数据库中的值（不备份）。
用法：
    python update_yaml_from_db.py [--db db_config.yaml] [--uuids uuid1,uuid2,...]
依赖：
    pip install ruamel.yaml
"""

import sys
from pathlib import Path
from typing import List, Optional

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from robocoin_dataset.database.database import DatasetDatabase
from robocoin_dataset.database.models import DatasetHardLinkDB

try:
    from ruamel.yaml import YAML
except ImportError:
    print("请安装 ruamel.yaml: pip install ruamel.yaml")
    sys.exit(1)

# ---------- 默认路径 ----------
DEFAULT_DB_CONFIG = "db/postgresql_config_test.yaml"


def update_yaml_file(yaml_path: Path, db_dataset_name: str, db_device_model: str) -> bool:
    """
    直接修改 YAML 文件（不备份），成功返回 True，失败返回 False。
    如果 device_model 原来是列表，会被替换为字符串。
    """
    try:
        # 检查文件是否可写（尝试打开以写入的方式，但不真正写）
        if not yaml_path.exists():
            print(f"   错误：文件不存在")
            return False
        if not yaml_path.is_file():
            print(f"   错误：路径不是文件")
            return False
        # 尝试以追加方式打开检查可写性（不会改变内容）
        try:
            with yaml_path.open('a'):
                pass
        except Exception:
            print(f"   错误：文件不可写")
            return False

        yaml = YAML()
        yaml.preserve_quotes = True
        yaml.width = 4096
        yaml.indent(mapping=2, sequence=4, offset=2)

        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.load(f)

        # 更新字段
        data['dataset_name'] = db_dataset_name
        data['device_model'] = db_device_model

        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f)

        return True
    except Exception as e:
        print(f"   错误：{type(e).__name__} - {e}")
        return False


def main(db_config: str, target_uuids: Optional[List[str]] = None):
    db = DatasetDatabase(Path(db_config).expanduser().absolute())

    with db.with_session() as session:
        query = session.query(DatasetHardLinkDB)
        if target_uuids:
            query = query.filter(DatasetHardLinkDB.dataset_uuid.in_(target_uuids))
        records = query.all()

        if not records:
            print("✅ 没有找到匹配的记录。")
            return

        total = len(records)
        success = 0
        fail = 0

        for idx, rec in enumerate(records, 1):
            yaml_path_str = rec.yaml_file_path
            db_name = rec.dataset_name
            db_device = rec.device_model

            print(f"\r[{idx}/{total}] 处理: {yaml_path_str or '空路径'}", end="")
            if not yaml_path_str:
                print(f"\n⚠️  ID={rec.id} 缺少 yaml_file_path，跳过")
                fail += 1
                continue

            yaml_path = Path(yaml_path_str)
            if not yaml_path.exists():
                print(f"\n❌ ID={rec.id} 文件不存在: {yaml_path}")
                fail += 1
                continue

            if not db_name or not db_device:
                print(f"\n⚠️  ID={rec.id} dataset_name({db_name}) 或 device_model({db_device}) 为空，跳过")
                fail += 1
                continue

            print(f"\n📝 更新: {yaml_path}")
            print(f"   dataset_name: {db_name}")
            print(f"   device_model: {db_device}")

            if update_yaml_file(yaml_path, db_name, db_device):
                print(f"   ✅ 成功")
                success += 1
            else:
                print(f"   ❌ 失败")
                fail += 1

        print(f"\n{'='*50}")
        print(f"📊 统计：总记录 {total}，成功 {success}，失败 {fail}")
        print(f"{'='*50}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="用数据库值直接更新 YAML（不备份）")
    parser.add_argument("--db", default=DEFAULT_DB_CONFIG, help="数据库配置文件路径")
    parser.add_argument("--uuids", default=None, help="指定数据集 UUID，逗号分隔")
    args = parser.parse_args()

    uuids = None
    if args.uuids:
        uuids = [u.strip() for u in args.uuids.split(",") if u.strip()]

    main(args.db, uuids)