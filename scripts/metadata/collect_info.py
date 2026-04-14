"""
Manual CLI entry for metadata collection.

Purpose:
    Trigger metadata collection explicitly and generate `info.yaml` at dataset root.
    This script is the standalone manual entrypoint for metadata stage.

Usage:
    python scripts/metadata/collect_info.py --dataset-path /data/my_dataset_qced_hardlink
    python scripts/metadata/collect_info.py --dataset-path /data/my_dataset_qced_hardlink --output-info-yaml-path /tmp/info.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from robocoin_dataset.metadata.collect import InfoCollector


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="collect_info.py",
        description="Manual metadata collection entrypoint. Generate info.yaml for one dataset.",
    )
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Dataset root path (hardlink directory).",
    )
    parser.add_argument(
        "--local-dataset-info-path",
        default=None,
        help="Optional explicit local_dataset_info.yaml path used by metadata collection.",
    )
    parser.add_argument(
        "--output-info-yaml-path",
        default=None,
        help="Optional output path for generated info.yaml. Default: <dataset-path>/info.yaml",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Optional log directory for metadata collection logs.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    collector = InfoCollector(
        dataset_path=dataset_path,
        local_dataset_info_path=args.local_dataset_info_path,
        output_info_yaml_path=args.output_info_yaml_path,
        log_dir=args.log_dir,
    )
    output_path = collector.collect()
    print(str(output_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
