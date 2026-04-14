#!/usr/bin/env python3
"""
Hub Upload Status Report Generator

Writes JSON reports:
1. hf_comparison.json — HuggingFace vs DB/should-upload comparison
2. ms_comparison.json — ModelScope vs DB/should-upload comparison
3. hf.json — full HuggingFace namespace repo name list (datasets)
4. ms.json — full ModelScope namespace repo name list (datasets)
5. hf_ms_repo_diff.json — HF vs MS repo set difference (only_on_hf, only_on_ms; summary includes on_both_count)

Each comparison file includes:
- should_upload_but_missing: should upload but missing on the hub
- cloud_but_not_in_should_upload: on the hub but not in the should-upload set
- cloud_but_not_in_db: on the hub but have NO corresponding record in the database (orphan cloud repos)
- both_have: in both should-upload and cloud
- uploaded_but_missing_in_cloud: marked uploaded locally but missing on the hub
- uploaded_but_not_marked: present on the hub but not marked completed in the DB
- uploaded_but_not_marked_should_upload: subset of uploaded_but_not_marked that should upload (sanity check)
- uploaded_but_not_marked_should_not_upload: subset that should not upload (expected empty)
- should_not_upload_but_marked_completed: data_loader_detection not complete but upload marked completed
- should_not_upload_but_marked_and_in_cloud: same as above and also present on the hub
- summary: aggregate counts

Usage:
    python scripts/hub_upload/tools/hub_upload_status_report.py
    python scripts/hub_upload/tools/hub_upload_status_report.py --db-cfg-path db/postgresql_config.yaml

Database connection uses PostgreSQL YAML; default is db/postgresql_config.yaml at the repository root.
"""

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

# Repo root: .../scripts/hub_upload/tools/this_file.py -> parents[3]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

import requests  # noqa: E402
from huggingface_hub import HfApi  # noqa: E402
from robocoin_dataset.database.database import DatasetDatabase  # noqa: E402
from robocoin_dataset.database.models import DatasetDB, TaskStatus  # noqa: E402

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")

# --- Configuration (override via CLI where applicable) ---
# Same relative path as scripts/hub_upload/config/*.yaml (pg_cfg_path: db/postgresql_config.yaml)
DEFAULT_PG_CONFIG_REL = Path("db/postgresql_config.yaml")
DEFAULT_PG_CONFIG = (REPO_ROOT / DEFAULT_PG_CONFIG_REL).resolve()

HF_NAMESPACE = "RoboCOIN"
MS_NAMESPACE = "RoboCOIN"

HF_TOKEN = None
MS_TOKEN = None

OUTPUT_DIR = Path(__file__).parent / "upload_status_reports"
# -----------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_hf_cloud_datasets(namespace: str, token: str | None = None) -> list[str]:
    """List dataset repo names under a HuggingFace namespace."""
    logger.info("Fetching HuggingFace datasets for namespace %s...", namespace)

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

        logger.info("HuggingFace: %d datasets", len(repo_names))
        return repo_names

    except Exception as e:
        logger.error("HuggingFace list failed: %s", e)
        return []


def get_ms_cloud_datasets(namespace: str, token: str | None = None) -> list[str]:
    """List dataset names under a ModelScope namespace via the public API."""
    logger.info("Fetching ModelScope datasets for namespace %s...", namespace)

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
                logger.warning("ModelScope API HTTP %s", response.status_code)
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

        logger.info("ModelScope: %d datasets", len(all_repo_names))
        return all_repo_names

    except Exception as e:
        logger.error("ModelScope list failed: %s", e)
        return []


def get_all_db_dataset_names(pg_config_path: Path | str) -> list[str]:
    """
    All dataset folder names that exist in the database (any record with convert_path).

    Used to detect orphan cloud repos: items on hub with no DB record at all.
    Returns folder names from convert_path (basename).
    """
    logger.info("DB: all dataset names (for orphan detection)...")

    try:
        db = DatasetDatabase(pg_config_path)

        with db.with_session() as session:
            query = session.query(DatasetDB.convert_path).filter(
                DatasetDB.convert_path.isnot(None),
                DatasetDB.convert_path != "",
            )
            results = query.all()

            seen: set[str] = set()
            dataset_names: list[str] = []
            for row in results:
                convert_path = row[0]
                if convert_path:
                    folder_name = Path(convert_path).name
                    if folder_name not in seen:
                        seen.add(folder_name)
                        dataset_names.append(folder_name)

            logger.info("DB: %d unique dataset names total", len(dataset_names))
            return dataset_names

    except Exception as e:
        logger.error("Database read failed: %s", e)
        return []


def get_should_upload_datasets(pg_config_path: Path | str, platform: str) -> list[str]:
    """
    Datasets that should be uploaded.

    Condition: visualize_check_status == COMPLETED (upload pre-stage).

    Returns folder names from convert_path (basename).
    """
    logger.info("DB: should-upload list for %s...", platform.upper())

    try:
        db = DatasetDatabase(pg_config_path)

        with db.with_session() as session:
            query = session.query(
                DatasetDB.dataset_uuid,
                DatasetDB.convert_path,
            ).filter(
                DatasetDB.visualize_check_status == TaskStatus.COMPLETED
            )

            results = query.all()

            dataset_names = []
            for row in results:
                convert_path = row[1]
                if convert_path:
                    folder_name = Path(convert_path).name
                    dataset_names.append(folder_name)
                else:
                    dataset_uuid = row[0]
                    logger.warning("dataset_uuid %s has empty convert_path, skipped", dataset_uuid)

            logger.info("DB: %d should-upload rows for %s", len(dataset_names), platform.upper())
            return dataset_names

    except Exception as e:
        logger.error("Database read failed: %s", e)
        return []


def get_uploaded_datasets(pg_config_path: Path | str, platform: str) -> list[str]:
    """
    Datasets marked uploaded locally.

    Condition: upload_status == COMPLETED.

    Returns folder names from convert_path (basename).
    """
    logger.info("DB: uploaded-completed list for %s...", platform.upper())

    if platform == "hf":
        upload_status_attr = DatasetDB.huggingface_upload_status
    elif platform == "ms":
        upload_status_attr = DatasetDB.ms_upload_status
    else:
        raise ValueError(f"Unsupported platform: {platform}")

    try:
        db = DatasetDatabase(pg_config_path)

        with db.with_session() as session:
            query = session.query(
                DatasetDB.dataset_uuid,
                DatasetDB.convert_path,
            ).filter(
                upload_status_attr == TaskStatus.COMPLETED
            )

            results = query.all()

            dataset_names = []
            for row in results:
                convert_path = row[1]
                if convert_path:
                    folder_name = Path(convert_path).name
                    dataset_names.append(folder_name)
                else:
                    dataset_uuid = row[0]
                    logger.warning("dataset_uuid %s has empty convert_path, skipped", dataset_uuid)

            logger.info("DB: %d uploaded-completed rows for %s", len(dataset_names), platform.upper())
            return dataset_names

    except Exception as e:
        logger.error("Database read failed: %s", e)
        return []


def get_should_not_upload_but_marked_completed(
    pg_config_path: Path | str, platform: str
) -> list[str]:
    """
    visualize_check_status != COMPLETED but upload_status == COMPLETED.

    Returns folder names from convert_path (basename).
    """
    logger.info("DB: should-not-upload but marked completed for %s...", platform.upper())

    if platform == "hf":
        upload_status_attr = DatasetDB.huggingface_upload_status
    elif platform == "ms":
        upload_status_attr = DatasetDB.ms_upload_status
    else:
        raise ValueError(f"Unsupported platform: {platform}")

    try:
        db = DatasetDatabase(pg_config_path)

        with db.with_session() as session:
            query = session.query(
                DatasetDB.dataset_uuid,
                DatasetDB.convert_path,
                DatasetDB.visualize_check_status,
            ).filter(
                DatasetDB.visualize_check_status != TaskStatus.COMPLETED,
                upload_status_attr == TaskStatus.COMPLETED
            )

            results = query.all()

            dataset_names = []
            for row in results:
                convert_path = row[1]
                if convert_path:
                    folder_name = Path(convert_path).name
                    dataset_names.append(folder_name)
                else:
                    dataset_uuid = row[0]
                    logger.warning("dataset_uuid %s has empty convert_path, skipped", dataset_uuid)

            logger.info(
                "DB: %d should-not-upload-but-marked rows for %s",
                len(dataset_names),
                platform.upper(),
            )
            return dataset_names

    except Exception as e:
        logger.error("Database read failed: %s", e)
        return []


def compare_datasets(
    should_upload: list[str],
    cloud_datasets: list[str],
    uploaded: list[str] | None = None,
    should_not_upload_but_marked: list[str] | None = None,
    all_db: list[str] | None = None,
) -> dict:
    """
    Compare should-upload vs cloud; optionally enrich with uploaded / inconsistent flags.

    See module docstring for result keys.
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
            "match_count": len(both),
        },
    }

    # Orphan cloud repos: on hub but no corresponding DB record at all
    if all_db is not None:
        all_db_set = {name.lower() for name in all_db}
        cloud_but_not_in_db = sorted(
            name for name in cloud_set
            if name.lower() not in all_db_set
        )
        result["cloud_but_not_in_db"] = cloud_but_not_in_db
        result["summary"]["cloud_but_not_in_db_count"] = len(cloud_but_not_in_db)

    if uploaded is not None:
        uploaded_set = set(uploaded)
        uploaded_but_missing = sorted(uploaded_set - cloud_set)
        result["uploaded_but_missing_in_cloud"] = uploaded_but_missing
        result["summary"]["uploaded_count"] = len(uploaded)
        result["summary"]["uploaded_but_missing_count"] = len(uploaded_but_missing)

        both_set = set(both)
        uploaded_but_not_marked = sorted(both_set - uploaded_set)
        result["uploaded_but_not_marked"] = uploaded_but_not_marked
        result["summary"]["uploaded_but_not_marked_count"] = len(uploaded_but_not_marked)

        uploaded_but_not_marked_set = set(uploaded_but_not_marked)
        uploaded_but_not_marked_should_upload = sorted(uploaded_but_not_marked_set & should_upload_set)
        uploaded_but_not_marked_should_not_upload = sorted(uploaded_but_not_marked_set - should_upload_set)
        result["uploaded_but_not_marked_should_upload"] = uploaded_but_not_marked_should_upload
        result["uploaded_but_not_marked_should_not_upload"] = uploaded_but_not_marked_should_not_upload
        result["summary"]["uploaded_but_not_marked_should_upload_count"] = len(
            uploaded_but_not_marked_should_upload
        )
        result["summary"]["uploaded_but_not_marked_should_not_upload_count"] = len(
            uploaded_but_not_marked_should_not_upload
        )

    if should_not_upload_but_marked is not None:
        should_not_upload_but_marked_set = set(should_not_upload_but_marked)
        result["should_not_upload_but_marked_completed"] = sorted(should_not_upload_but_marked_set)
        result["summary"]["should_not_upload_but_marked_count"] = len(should_not_upload_but_marked_set)

        should_not_upload_but_marked_and_in_cloud = sorted(should_not_upload_but_marked_set & cloud_set)
        result["should_not_upload_but_marked_and_in_cloud"] = should_not_upload_but_marked_and_in_cloud
        result["summary"]["should_not_upload_but_marked_and_in_cloud_count"] = len(
            should_not_upload_but_marked_and_in_cloud
        )

    return result


def compare_hf_ms_repos(hf_repos: list[str], ms_repos: list[str]) -> dict:
    """
    Compare dataset repo names between HuggingFace and ModelScope (same logical namespace).

    @input: hf_repos, ms_repos — list of repo names (folder names, no owner prefix).
    @output: dict with only_on_hf, only_on_ms, on_both, summary counts.
    @scenario: Find repos present on one hub but not the other.
    """
    hf_set = set(hf_repos)
    ms_set = set(ms_repos)

    only_hf = sorted(hf_set - ms_set)
    only_ms = sorted(ms_set - hf_set)
    on_both = sorted(hf_set & ms_set)

    return {
        "only_on_hf": only_hf,
        "only_on_ms": only_ms,
        "on_both": on_both,
        "summary": {
            "hf_count": len(hf_repos),
            "ms_count": len(ms_repos),
            "only_on_hf_count": len(only_hf),
            "only_on_ms_count": len(only_ms),
            "on_both_count": len(on_both),
        },
    }


def hf_ms_diff_for_export(full: dict) -> dict:
    """
    Slim HF-vs-MS diff for JSON export: omit large on_both list (see hf.json / ms.json).

    @input: full — return value of compare_hf_ms_repos.
    @output: dict with only_on_hf, only_on_ms, summary.
    @scenario: Avoid duplicating full intersection in hf_ms_repo_diff.json.
    """
    return {
        "only_on_hf": full["only_on_hf"],
        "only_on_ms": full["only_on_ms"],
        "summary": full["summary"],
    }


def hub_repo_list_payload(namespace: str, platform: str, repos: list[str]) -> dict:
    """
    Serializable payload for hf.json / ms.json (full cloud repo list).

    @input: namespace (hub org name), platform ("hf" or "ms"), repos (repo names).
    @output: dict suitable for json.dump.
    @scenario: Stable export of raw hub dataset lists for diffing or auditing.
    """
    sorted_repos = sorted(repos)
    return {
        "platform": platform,
        "namespace": namespace,
        "count": len(sorted_repos),
        "repos": sorted_repos,
    }


def save_json(data: dict | list, filepath: Path) -> None:
    """Write JSON with UTF-8 encoding."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info("Saved: %s", filepath)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hub upload status report (HF / ModelScope vs DB).")
    parser.add_argument(
        "--db-cfg-path",
        type=Path,
        default=DEFAULT_PG_CONFIG,
        help=(
            "PostgreSQL YAML config. "
            f"Default: {DEFAULT_PG_CONFIG_REL} under the repository root ({DEFAULT_PG_CONFIG})"
        ),
    )
    args = parser.parse_args()
    pg_config_path = args.db_cfg_path.resolve()

    logger.info("=" * 60)
    logger.info("Hub Upload Status Report Generator")
    logger.info("=" * 60)
    logger.info("PostgreSQL config: %s", pg_config_path)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", OUTPUT_DIR)
    logger.info("")

    logger.info("Step 1: Query datasets from DB")
    logger.info("-" * 60)

    all_db_names = get_all_db_dataset_names(pg_config_path)
    hf_should_upload = get_should_upload_datasets(pg_config_path, "hf")
    ms_should_upload = get_should_upload_datasets(pg_config_path, "ms")
    logger.info("")

    logger.info("Step 2: Query locally marked upload-completed datasets")
    logger.info("-" * 60)

    hf_uploaded = get_uploaded_datasets(pg_config_path, "hf")
    ms_uploaded = get_uploaded_datasets(pg_config_path, "ms")
    logger.info("")

    logger.info(
        "Step 3: Query inconsistent: data_loader_detection not complete but upload marked complete"
    )
    logger.info("-" * 60)

    hf_should_not_upload_but_marked = get_should_not_upload_but_marked_completed(pg_config_path, "hf")
    ms_should_not_upload_but_marked = get_should_not_upload_but_marked_completed(pg_config_path, "ms")
    logger.info("")

    logger.info("Step 4: Fetch cloud dataset lists")
    logger.info("-" * 60)

    hf_cloud = get_hf_cloud_datasets(HF_NAMESPACE, HF_TOKEN)
    ms_cloud = get_ms_cloud_datasets(MS_NAMESPACE, MS_TOKEN)
    logger.info("")

    logger.info("Step 5: Compare and write JSON")
    logger.info("-" * 60)

    hf_comparison = compare_datasets(
        hf_should_upload,
        hf_cloud,
        hf_uploaded,
        hf_should_not_upload_but_marked,
        all_db_names,
    )
    ms_comparison = compare_datasets(
        ms_should_upload,
        ms_cloud,
        ms_uploaded,
        ms_should_not_upload_but_marked,
        all_db_names,
    )

    save_json(hf_comparison, OUTPUT_DIR / "hf_comparison.json")
    save_json(ms_comparison, OUTPUT_DIR / "ms_comparison.json")

    hf_repo_payload = hub_repo_list_payload(HF_NAMESPACE, "hf", hf_cloud)
    ms_repo_payload = hub_repo_list_payload(MS_NAMESPACE, "ms", ms_cloud)
    save_json(hf_repo_payload, OUTPUT_DIR / "hf.json")
    save_json(ms_repo_payload, OUTPUT_DIR / "ms.json")

    hf_ms_diff = compare_hf_ms_repos(hf_cloud, ms_cloud)
    save_json(hf_ms_diff_for_export(hf_ms_diff), OUTPUT_DIR / "hf_ms_repo_diff.json")
    logger.info("")

    logger.info("HF vs MS repo diff: only_on_hf=%s only_on_ms=%s on_both=%s",
                hf_ms_diff["summary"]["only_on_hf_count"],
                hf_ms_diff["summary"]["only_on_ms_count"],
                hf_ms_diff["summary"]["on_both_count"])
    logger.info("")

    logger.info("=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)

    logger.info("")
    logger.info("HuggingFace:")
    logger.info("  should_upload: %s", hf_comparison["summary"]["should_upload_count"])
    logger.info("  cloud_count: %s", hf_comparison["summary"]["cloud_count"])
    logger.info("  missing (should upload, not on hub): %s", hf_comparison["summary"]["missing_count"])
    logger.info("  extra (on hub, not in should-upload): %s", hf_comparison["summary"]["extra_count"])
    logger.info(
        "  WARN cloud repos with NO DB record (orphans): %s",
        hf_comparison["summary"].get("cloud_but_not_in_db_count", 0),
    )
    logger.info("  match: %s", hf_comparison["summary"]["match_count"])
    logger.info("  local marked uploaded: %s", hf_comparison["summary"].get("uploaded_count", 0))
    logger.info(
        "  WARN local uploaded but missing on hub: %s",
        hf_comparison["summary"].get("uploaded_but_missing_count", 0),
    )
    logger.info(
        "  WARN on hub but not marked uploaded in DB: %s",
        hf_comparison["summary"].get("uploaded_but_not_marked_count", 0),
    )
    logger.info(
        "    - of which should_upload: %s",
        hf_comparison["summary"].get("uploaded_but_not_marked_should_upload_count", 0),
    )
    logger.info(
        "    - of which should_not_upload: %s",
        hf_comparison["summary"].get("uploaded_but_not_marked_should_not_upload_count", 0),
    )
    logger.info(
        "  WARN data_loader_detection not complete but upload marked complete: %s",
        hf_comparison["summary"].get("should_not_upload_but_marked_count", 0),
    )
    logger.info(
        "  WARN same and present on hub: %s",
        hf_comparison["summary"].get("should_not_upload_but_marked_and_in_cloud_count", 0),
    )

    logger.info("")
    logger.info("ModelScope:")
    logger.info("  should_upload: %s", ms_comparison["summary"]["should_upload_count"])
    logger.info("  cloud_count: %s", ms_comparison["summary"]["cloud_count"])
    logger.info("  missing (should upload, not on hub): %s", ms_comparison["summary"]["missing_count"])
    logger.info("  extra (on hub, not in should-upload): %s", ms_comparison["summary"]["extra_count"])
    logger.info(
        "  WARN cloud repos with NO DB record (orphans): %s",
        ms_comparison["summary"].get("cloud_but_not_in_db_count", 0),
    )
    logger.info("  match: %s", ms_comparison["summary"]["match_count"])
    logger.info("  local marked uploaded: %s", ms_comparison["summary"].get("uploaded_count", 0))
    logger.info(
        "  WARN local uploaded but missing on hub: %s",
        ms_comparison["summary"].get("uploaded_but_missing_count", 0),
    )
    logger.info(
        "  WARN on hub but not marked uploaded in DB: %s",
        ms_comparison["summary"].get("uploaded_but_not_marked_count", 0),
    )
    logger.info(
        "    - of which should_upload: %s",
        ms_comparison["summary"].get("uploaded_but_not_marked_should_upload_count", 0),
    )
    logger.info(
        "    - of which should_not_upload: %s",
        ms_comparison["summary"].get("uploaded_but_not_marked_should_not_upload_count", 0),
    )
    logger.info(
        "  WARN data_loader_detection not complete but upload marked complete: %s",
        ms_comparison["summary"].get("should_not_upload_but_marked_count", 0),
    )
    logger.info(
        "  WARN same and present on hub: %s",
        ms_comparison["summary"].get("should_not_upload_but_marked_and_in_cloud_count", 0),
    )

    logger.info("")
    logger.info("HF vs ModelScope (cloud repo names):")
    logger.info("  HF repos: %s", hf_ms_diff["summary"]["hf_count"])
    logger.info("  MS repos: %s", hf_ms_diff["summary"]["ms_count"])
    logger.info("  on both hubs: %s", hf_ms_diff["summary"]["on_both_count"])
    logger.info("  only on HuggingFace: %s", hf_ms_diff["summary"]["only_on_hf_count"])
    logger.info("  only on ModelScope: %s", hf_ms_diff["summary"]["only_on_ms_count"])

    logger.info("")
    logger.info("=" * 60)
    logger.info("Reports written under: %s", OUTPUT_DIR)
    logger.info("=" * 60)

    logger.info("")
    logger.info("Generated files:")
    for file in sorted(OUTPUT_DIR.glob("*.json")):
        logger.info("  - %s", file.name)


if __name__ == "__main__":
    main()
