This stage syncs dataset information to the page project by generating assets (YAML files and videos).
Run this after upload.

**Scripts**:

1. **sync.py** – Generate assets (dataset YAML + videos) from the database
2. **upload_assets.py** – Upload the local `assets/` folder to HuggingFace
3. **auto_sync_workflow.py** – Automated loop: generate assets → upload to HuggingFace
4. **reset_db_for_page_tst.py** – Reset hub/sync statuses for testing

---

**sync.py** (one-shot asset generation):

```bash
# Basic usage
python scripts/page_sync/sync.py \
  --db-cfg-path db/postgresql_config.yaml \
  --target-dir ~/projects/DataManager/robocoin_datamanager_assets

# Force regenerate videos and thumbnails
python scripts/page_sync/sync.py \
  --db-cfg-path db/postgresql_config.yaml \
  --target-dir /path/to/page-project \
  --update-videos \
  --crf 30

# With HuggingFace upload
python scripts/page_sync/sync.py \
  --db-cfg-path db/postgresql_config.yaml \
  --target-dir /path/to/page-project \
  --hf-token your_hf_token \
  --hf-repo-id RogersPyke/robocoin_datamanager_assets \
  --force-regenerate
```

`--crf` controls video compression (0–51; lower = better quality). Default 23; ~30 gives ~500KB per video.

---

**upload_assets.py** (upload assets only):

```bash
python scripts/page_sync/upload_assets.py \
  --assets-dir /path/to/page-project/assets \
  --repo-id RogersPyke/robocoin_datamanager_assets \
  --hf-token hf_xxx
```

If `--assets-dir` is omitted, it defaults to `./assets`.
Use `--only-missing` if you want explicit incremental upload behavior.

---

**auto_sync_workflow.py** (periodic sync + HF upload):

```bash
export HF_TOKEN=

# Single run (for debugging)
python scripts/page_sync/auto_sync_workflow.py \
  --db-cfg-path /mnt/db/postgresql_config.yaml \
  --target-dir /path/to/projects \
  --token <your_hf_token> \
  --run-once

# Loop every 2 hours (default)
python scripts/page_sync/auto_sync_workflow.py \
  --db-cfg-path db/postgresql_config.yaml \
  --target-dir ~/projects/robocoin_datamanager_assets \
  --interval-hours 2

# Background run
nohup python scripts/page_sync/auto_sync_workflow.py \
  --db-cfg-path /mnt/db/postgresql_config.yaml \
  --target-dir /path/to/projects \
  --token <your_hf_token> \
  > auto_sync.log 2>&1 &
```

Workflow steps:
(1) pull existing `dataset_info/*.yaml` from HuggingFace into local `target-dir/assets`,
(2) run page sync (generate/update local `target-dir/assets`),
(3) upload assets to HuggingFace to keep remote files updated.

---

**reset_db_for_page_tst.py** (testing):

```bash
python scripts/page_sync/reset_db_for_page_tst.py --db path/to/database.db
```

Resets hub upload statuses and `dataset_info_sync_status` to PENDING for all datasets.

AFTER assets generation, to take efffect on the website, you must upload them to https://huggingface.co/datasets/RogersPyke/robocoin_datamanager_assets.