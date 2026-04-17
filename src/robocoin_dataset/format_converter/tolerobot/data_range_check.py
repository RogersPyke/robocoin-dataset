import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ==========================================
# 🌟 细粒度物理绝对极限定义 (Hard Bounds)
# ==========================================
FINE_GRAINED_BOUNDS = {
    "_rad": {"min": -3.1415926, "max": 3.1415926},
    "_m": {"min": -2.0, "max": 2.0},
    "gripper_open_scale": {"min": 0.0, "max": 1.0},
    "gripper_open": {"min": 0.0, "max": 1025.0},
}

GLOBAL_FALLBACK_BOUNDS = {"min": -10.0, "max": 10.0}


class LerobotDatasetValidator:
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path).expanduser().absolute()
        self.info_path = self.dataset_path / "meta" / "info.json"
        self.data_dir = self.dataset_path / "data"

        if not self.info_path.exists():
            self.info_path = self.dataset_path / "info.json"

        self._validate_paths()
        self.info_data = self._load_info_json()
        self.expected_features = self.info_data.get('features', {})
        self.expected_stats = self.info_data.get('stats', {})
        self.feature_names_map = self._extract_feature_names()

    def _validate_paths(self):
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"数据集路径不存在: {self.dataset_path}")
        if not self.info_path.exists():
            raise FileNotFoundError(f"找不到 info.json 文件: {self.info_path}")
        if not self.data_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")

    def _load_info_json(self) -> dict:
        with open(self.info_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _extract_feature_names(self) -> Dict[str, List[str]]:
        names_map = {}
        for feat_name, feat_info in self.expected_features.items():
            if "names" in feat_info and isinstance(feat_info["names"], list):
                names_map[feat_name] = feat_info["names"]
        return names_map

    def _validate_single_parquet(self, file_path: Path) -> List[str]:
        """
        单个文件校验：任何异常都捕获，只返回错误列表，绝不崩溃
        """
        errors = []
        try:
            df = pd.read_parquet(file_path)
        except Exception as e:
            errors.append(f"读取失败: {str(e)}")
            return errors

        for feature_name, feature_info in self.expected_features.items():
            if not (feature_name.startswith("action") or feature_name.startswith("observation.state")):
                continue

            if feature_name not in df.columns:
                errors.append(f"缺失列: {feature_name}")
                continue

            expected_shape = tuple(feature_info.get('shape', []))
            dtype = feature_info.get('dtype')

            if not expected_shape or dtype in ['video', 'image']:
                continue

            col_data = df[feature_name]

            try:
                np_data = np.vstack(col_data.values)
            except ValueError:
                errors.append(f"{feature_name} 数据格式异常，无法堆叠为数组")
                continue

            # 维度检查
            actual_shape = np_data.shape[1:]
            if actual_shape != expected_shape:
                errors.append(f"{feature_name} 维度错误：预期{expected_shape}，实际{actual_shape}")
                continue

            # NaN / Inf
            if np_data.dtype.kind in 'fc':
                if np.isnan(np_data).any():
                    errors.append(f"{feature_name} 存在 NaN 缺失值")
                if np.isinf(np_data).any():
                    errors.append(f"{feature_name} 存在 Inf 无限值")

            # 全零子维度
            if feature_name in self.feature_names_map:
                sub_names = self.feature_names_map[feature_name]
                for idx, sub in enumerate(sub_names):
                    if np.all(np_data[:, idx] == 0):
                        errors.append(f"{feature_name}.{sub} 全部为 0")

            # 整帧全零
            if np.all(np_data == 0):
                errors.append(f"{feature_name} 存在全零帧")

            # 物理硬限制
            if feature_name in self.feature_names_map:
                sub_names = self.feature_names_map[feature_name]
                for idx, sub in enumerate(sub_names):
                    col = np_data[:, idx]
                    matched = None
                    for kw, b in FINE_GRAINED_BOUNDS.items():
                        if kw in sub:
                            matched = b
                            break
                    if matched:
                        mn = matched["min"]
                        mx = matched["max"]
                        if np.any(col < mn - 1e-4):
                            errors.append(f"{feature_name}.{sub} 低于物理下限 {mn}")
                        if np.any(col > mx + 1e-4):
                            errors.append(f"{feature_name}.{sub} 高于物理上限 {mx}")
            else:
                amin = np.min(np_data)
                amax = np.max(np_data)
                if amin < -10 or amax > 10:
                    errors.append(f"{feature_name} 数值范围异常 [{amin:.2f}, {amax:.2f}]")

            # stats 软限制
            stats = self.expected_stats.get(feature_name)
            if stats and 'min' in stats and 'max' in stats:
                exp_min = np.array(stats['min'])
                exp_max = np.array(stats['max'])
                real_min = np.min(np_data, axis=0)
                real_max = np.max(np_data, axis=0)
                tol = 1e-5
                if np.any(real_min < exp_min - tol):
                    errors.append(f"{feature_name} 低于 info.json min")
                if np.any(real_max > exp_max + tol):
                    errors.append(f"{feature_name} 高于 info.json max")

        return errors

    def run(self) -> tuple[bool, str]:
        logger.info(f"🔍 开始校验数据集: {self.dataset_path.name}")

        parquet_files = list(self.data_dir.rglob("*.parquet"))
        if not parquet_files:
            return False, "无 parquet 文件"

        total_errors = 0
        all_errors = []

        for file in parquet_files:
            try:
                errs = self._validate_single_parquet(file)
            except Exception as e:
                # 最外层兜底：任何意外异常都捕获
                errs = [f"校验过程异常: {str(e)}"]

            if errs:
                total_errors += len(errs)
                msg = f"[{file.name}] {' | '.join(errs)}"
                all_errors.append(msg)
                logger.warning(f"⚠️ {msg}")

        if total_errors == 0:
            logger.info("✅ 校验完美通过！")
            return True, ""
        else:
            final = "\n".join(all_errors)
            logger.error(f"❌ 校验完成，共 {total_errors} 处异常\n{final}")
            return False, f"共 {total_errors} 处异常：\n{final}"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    DATASET_PATH = sys.argv[1] if len(sys.argv) > 1 else "/mnt/nas/synnas/成功区/五次成功区/AI2_Alphabot_2_organize_lab_equipment_0"

    # 最外层也保护，整个脚本永不崩
    try:
        validator = LerobotDatasetValidator(DATASET_PATH)
        validator.run()
    except Exception as e:
        logger.error(f"💥 校验程序异常: {e}")