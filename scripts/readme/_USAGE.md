This script supports two modes:
1) dataset mode (single dataset)
2) batch mode (query tasks from database and process one-by-one)

In dataset mode:
- It searches existing `info.yaml` in the hardlink folder.
- If not found (or `--force`), it calls metadata collection to regenerate `info.yaml`.
- Then it renders `README.md`.

To run dataset mode:
```
python scripts/readme/gen_readme.py \
    --dataset-path ~/projects/TestDatasets_0/Agilex_Cobot_Magic_pour_water_into_cup_0_qced_hardlink \
    --local-dataset-info-path ~/projects/TestDatasets_0/local_dataset_info.yaml
```
If `--local-dataset-info-path` is not provided, the script tries to find `local_dataset_info.yaml` in the hardlink folder.

To generate for many datasets from DB:
```
python scripts/readme/gen_readme.py \
--db-cfg-path db/postgresql_config.yaml  \
--force
```
In batch mode, script selects datasets that:

    i. passed dataloader_check.
    ii. should be uploaded but not fully uploaded.

If you want to generate for ALL datasets and don't care if uploaded, pass `--ignore-uploaded`.