#!/usr/bin/env python3
"""
HF-DB Dataset Name Mapping Analyzer

Compares HuggingFace dataset names against local PostgreSQL database names,
producing a YAML report with matched and unmatched entries.

Matching strategy:
  1. Exact match: HF repo name (after stripping namespace) == DB dataset name.
  2. Wide match (prefix normalization): replace non-standard robot prefix with
     the canonical common_name from robot_aliases.json, then re-attempt exact match.
  3. Suffix-only wide match: for entries still unmatched after step 2, strip the
     robot prefix entirely and match only the task suffix against DB suffixes
     (i.e. match by task description, ignoring robot name discrepancies).

@input:
  - PostgreSQL config YAML (default: db/postgresql_config.yaml)
  - robot_aliases.json path (default: src/robocoin_dataset/metadata/assets/robot_aliases.json)
  - HF namespace (default: RoboCOIN)
  - HF token (optional, from env HF_TOKEN)
@output:
  - YAML file: docs/HfDbMapAnaVer2.yaml
@scenario:
  Detect naming mismatches between HuggingFace repos and the local database,
  normalize robot names via alias mapping, and surface unmatched entries for
  manual review.
"""

import argparse
import json
import logging
import os
import sys
import warnings
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Repo root: .../scripts/hub_upload/tools/this_file.py -> parents[3]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

import yaml  # noqa: E402
from huggingface_hub import HfApi  # noqa: E402
from robocoin_dataset.database.database import DatasetDatabase  # noqa: E402
from robocoin_dataset.database.models import DatasetDB  # noqa: E402

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")

# --- Configuration ---
DEFAULT_PG_CONFIG = (REPO_ROOT / "db" / "postgresql_config.yaml").resolve()
DEFAULT_ALIASES_PATH = (
    REPO_ROOT / "src" / "robocoin_dataset" / "metadata" / "assets" / "robot_aliases.json"
).resolve()
DEFAULT_OUTPUT = (REPO_ROOT / "docs" / "HfDbMapAnaVer2.yaml").resolve()
HF_NAMESPACE = "RoboCOIN"
HF_TOKEN = None
# ----------------------

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(name)s] [%(levelname)s] %(message)s",
)
logger = logging.getLogger("hf_db_mapping")


# ---------------------------------------------------------------------------
# Data retrieval
# ---------------------------------------------------------------------------

def get_hf_datasets(namespace: str, token: str | None = None) -> list[str]:
    """
    List dataset repo names under a HuggingFace namespace (strip namespace prefix).

    @input:  namespace (str), token (str|None)
    @output: list[str] of repo names without namespace prefix
    """
    logger.info("Fetching HuggingFace datasets for namespace %s ...", namespace)
    try:
        api = HfApi(token=token)
        repos = api.list_datasets(author=namespace)
        repo_names = []
        for repo in repos:
            repo_id = repo.id
            name = repo_id.split("/", 1)[1] if "/" in repo_id else repo_id
            repo_names.append(name)
        logger.info("HuggingFace: %d datasets", len(repo_names))
        return repo_names
    except Exception as e:
        logger.error("HuggingFace list failed: %s", e)
        return []


def get_db_dataset_names(pg_config_path: Path | str) -> list[str]:
    """
    All unique dataset names from DB (dataset_name column).

    @input:  pg_config_path (Path|str)
    @output: list[str] of unique dataset names
    """
    logger.info("DB: fetching dataset names ...")
    try:
        db = DatasetDatabase(str(pg_config_path))
        with db.with_session() as session:
            results = session.query(DatasetDB.dataset_name).filter(
                DatasetDB.dataset_name.isnot(None),
                DatasetDB.dataset_name != "",
            ).all()
            seen: set[str] = set()
            names: list[str] = []
            for (dataset_name,) in results:
                if dataset_name not in seen:
                    seen.add(dataset_name)
                    names.append(dataset_name)
        logger.info("DB: %d unique dataset names", len(names))
        return names
    except Exception as e:
        logger.error("Database read failed: %s", e)
        return []


# ---------------------------------------------------------------------------
# Alias / prefix helpers
# ---------------------------------------------------------------------------

def load_aliases(path: Path) -> dict:
    """
    Load robot_aliases.json.

    @input:  path (Path) to robot_aliases.json
    @output: dict  original key -> {"common_name": str, "aliases": list[str]}
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_prefix_map(aliases: dict) -> dict[str, str]:
    """
    Build a lookup: every possible alias (lowercased) -> canonical common_name.

    Includes the original dict key and the common_name itself as lookup keys.

    @input:  aliases dict from load_aliases
    @output: dict[str, str]  lowercase_alias -> common_name
    """
    mapping: dict[str, str] = {}
    for _key, info in aliases.items():
        common = info["common_name"]
        mapping[_key.lower()] = common
        mapping[common.lower()] = common
        for a in info.get("aliases", []):
            mapping[a.lower()] = common
    return mapping


def normalize_prefix(name: str, prefix_map: dict[str, str]) -> tuple[str, str | None]:
    """
    Try to normalize the robot prefix of a dataset name.

    Strategy: try the longest matching prefix first. Split the name at '_'
    boundaries and try progressively shorter prefixes.

    @input:  name (str) dataset name, prefix_map (dict)
    @output: (normalized_name, matched_prefix_lower) -- if no alias matched,
             returns (name, None).
    """
    parts = name.split("_")
    for end in range(len(parts), 0, -1):
        candidate = "_".join(parts[:end]).lower()
        if candidate in prefix_map:
            canonical = prefix_map[candidate]
            suffix = "_".join(parts[end:])
            normalized = f"{canonical}_{suffix}" if suffix else canonical
            return normalized, candidate
    return name, None


def extract_suffix(name: str, prefix_map: dict[str, str]) -> str:
    """
    Extract the task suffix after the robot prefix.

    @input:  name (str) dataset name, prefix_map (dict)
    @output: str suffix after the matched prefix, or full name if no prefix matched
    """
    parts = name.split("_")
    for end in range(len(parts), 0, -1):
        candidate = "_".join(parts[:end]).lower()
        if candidate in prefix_map:
            return "_".join(parts[end:])
    return name


# ---------------------------------------------------------------------------
# Matching engine
# ---------------------------------------------------------------------------

def match_datasets(
    hf_names: list[str],
    db_names: list[str],
    prefix_map: dict[str, str],
) -> dict:
    """
    Three-pass matching:

    Pass 1 -- exact match: hf_name == db_name
    Pass 2 -- wide match: normalize hf_name prefix via aliases, then exact match
    Pass 3 -- suffix-only match: match hf suffix against db suffixes
              (ignoring prefix entirely, but only for entries that failed pass 2)

    @input:  hf_names, db_names, prefix_map
    @output: dict with keys: matched, wide_matched, suffix_matched, unmatched, metadata
    """
    db_set = set(db_names)

    # Build suffix index: suffix (lower) -> list of db_names having that suffix
    db_suffix_index: dict[str, list[str]] = {}
    for dbn in db_names:
        suf = extract_suffix(dbn, prefix_map)
        db_suffix_index.setdefault(suf.lower(), []).append(dbn)

    matched: dict[str, list[dict]] = {}
    wide_matched: dict[str, list[dict]] = {}
    suffix_matched: dict[str, list[dict]] = {}
    unmatched: list[dict] = []

    claimed_db: set[str] = set()

    # --- Pass 1: exact match ---
    still_unmatched: list[str] = []
    for hf in hf_names:
        if hf in db_set:
            prefix = hf.split("_")[0]
            matched.setdefault(prefix, []).append({
                "hf_name": hf,
                "db_name": hf,
            })
            claimed_db.add(hf)
        else:
            still_unmatched.append(hf)

    logger.info(
        "Pass 1 (exact): %d matched, %d remaining",
        len(hf_names) - len(still_unmatched),
        len(still_unmatched),
    )

    # --- Pass 2: wide match (prefix normalization) ---
    pass2_remaining: list[str] = []
    for hf in still_unmatched:
        normalized, _ = normalize_prefix(hf, prefix_map)
        if normalized in db_set and normalized not in claimed_db:
            canon_prefix = normalized.split("_")[0]
            wide_matched.setdefault(canon_prefix, []).append({
                "hf_name": hf,
                "wide_name": normalized,
                "db_name": normalized,
            })
            claimed_db.add(normalized)
        else:
            pass2_remaining.append(hf)

    logger.info(
        "Pass 2 (wide): %d matched, %d remaining",
        len(still_unmatched) - len(pass2_remaining),
        len(pass2_remaining),
    )

    # --- Pass 3: suffix-only wide match ---
    pass3_remaining: list[str] = []
    for hf in pass2_remaining:
        hf_suffix = extract_suffix(hf, prefix_map)
        candidates = db_suffix_index.get(hf_suffix.lower(), [])
        found = None
        for cand in candidates:
            if cand not in claimed_db:
                found = cand
                break
        if found:
            canon_prefix = found.split("_")[0]
            db_suffix = extract_suffix(found, prefix_map)
            suffix_matched.setdefault(canon_prefix, []).append({
                "hf_name": hf,
                "db_name": found,
                "same_suffix": hf_suffix if hf_suffix == db_suffix else f"{hf_suffix}|{db_suffix}",
            })
            claimed_db.add(found)
        else:
            pass3_remaining.append(hf)

    logger.info(
        "Pass 3 (suffix-only): %d matched, %d remaining",
        len(pass2_remaining) - len(pass3_remaining),
        len(pass3_remaining),
    )

    # --- Build unmatched list ---
    for hf in pass3_remaining:
        normalized, _ = normalize_prefix(hf, prefix_map)
        hf_suffix = extract_suffix(hf, prefix_map)
        unmatched.append({
            "hf_name": hf,
            "wide_name": normalized,
            "db_name": None,
            "same_suffix": hf_suffix,
        })

    total_matched = (
        sum(len(v) for v in matched.values())
        + sum(len(v) for v in wide_matched.values())
        + sum(len(v) for v in suffix_matched.values())
    )
    total_unmatched = len(unmatched)
    total = len(hf_names)

    tz_utc8 = timezone(timedelta(hours=8))
    metadata = {
        "generated_at": datetime.now(tz_utc8).strftime("%Y-%m-%d %H:%M:%S"),
        "total_hf_datasets": len(hf_names),
        "total_db_datasets": len(db_names),
        "exact_matched_count": sum(len(v) for v in matched.values()),
        "wide_matched_count": sum(len(v) for v in wide_matched.values()),
        "suffix_matched_count": sum(len(v) for v in suffix_matched.values()),
        "total_matched_count": total_matched,
        "unmatched_count": total_unmatched,
        "match_rate": f"{total_matched / total * 100:.1f}%" if total else "0%",
    }

    return {
        "metadata": metadata,
        "matched": matched,
        "wide_matched": wide_matched,
        "suffix_matched": suffix_matched,
        "unmatched": unmatched,
    }


# ---------------------------------------------------------------------------
# YAML serialization
# ---------------------------------------------------------------------------

def build_yaml_output(result: dict) -> dict:
    """Convert result dict into the target YAML structure."""
    out: dict = {}

    out["metadata"] = result["metadata"]

    out["matched"] = {}
    for prefix, entries in sorted(result["matched"].items()):
        out["matched"][prefix] = {
            "count": len(entries),
            "datasets": entries,
        }

    out["wide_matched"] = {}
    for prefix, entries in sorted(result["wide_matched"].items()):
        out["wide_matched"][prefix] = {
            "count": len(entries),
            "datasets": entries,
        }

    out["suffix_matched"] = {}
    for prefix, entries in sorted(result["suffix_matched"].items()):
        out["suffix_matched"][prefix] = {
            "count": len(entries),
            "datasets": entries,
        }

    out["unmatched"] = {
        "count": len(result["unmatched"]),
        "datasets": result["unmatched"],
    }

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="HF-DB dataset name mapping analyzer")
    parser.add_argument(
        "--db-cfg-path",
        type=Path,
        default=DEFAULT_PG_CONFIG,
        help="Path to PostgreSQL config YAML",
    )
    parser.add_argument(
        "--aliases-path",
        type=Path,
        default=DEFAULT_ALIASES_PATH,
        help="Path to robot_aliases.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output YAML path",
    )
    parser.add_argument(
        "--hf-namespace",
        type=str,
        default=HF_NAMESPACE,
        help="HuggingFace namespace",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=HF_TOKEN,
        help="HuggingFace API token (or set HF_TOKEN env)",
    )
    args = parser.parse_args()

    token = args.hf_token or os.environ.get("HF_TOKEN")

    hf_names = get_hf_datasets(args.hf_namespace, token)
    if not hf_names:
        logger.error("No HuggingFace datasets found. Aborting.")
        sys.exit(1)

    db_names = get_db_dataset_names(args.db_cfg_path)
    if not db_names:
        logger.error("No DB dataset names found. Aborting.")
        sys.exit(1)

    aliases = load_aliases(args.aliases_path)
    prefix_map = build_prefix_map(aliases)
    logger.info("Alias prefix map: %d entries", len(prefix_map))

    result = match_datasets(hf_names, db_names, prefix_map)

    yaml_data = build_yaml_output(result)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    meta = result["metadata"]
    logger.info(
        "Report written to %s  |  exact=%s  wide=%s  suffix=%s  unmatched=%s  rate=%s",
        args.output,
        meta["exact_matched_count"],
        meta["wide_matched_count"],
        meta["suffix_matched_count"],
        meta["unmatched_count"],
        meta["match_rate"],
    )


if __name__ == "__main__":
    main()
