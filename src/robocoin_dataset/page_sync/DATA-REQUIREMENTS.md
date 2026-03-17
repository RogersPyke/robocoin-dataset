# Data Requirements: Single to Multiple Datasets

What the page needs to run at full function, from one dataset to many.

---

## 0. Alignment with current page_sync output

**File layout:** Current output matches the required layout: `info/consolidated_datasets.json`, `info/data_index.json`, `dataset_info/<path>.yaml`, `info/exclude.json`, `thumbnails/<path>.jpg`, `videos/<path>.mp4` (path = dataset name, no extension).

**Info.yaml is a superset of page requirements.** The metadata schema (and thus every generated info.yaml) defines a field set that **fully contains** the information the page needs: every field the page uses (required or optional) either exists in info.yaml under the same name or is derivable from one or more info.yaml fields. Section **6** gives the **strict per-field mapping** so that each page-used field has exactly one correspondence (same name or derivation rule).

---

## 1. Assets root

All data lives under an **assets root** URL (e.g. Hugging Face `resolve/main`).  
Paths below are relative to that root.

---

## 2. Files required

| File | Required | Purpose |
|------|----------|--------|
| **info/consolidated_datasets.json** | Preferred | One JSON with all datasets; fast load, full function. |
| **info/data_index.json** | If no consolidated | List of dataset paths; used to load each YAML. |
| **dataset_info/\<path\>.yml** (or .yaml) | If using data_index | One file per dataset; same content as one entry in consolidated. |
| **info/exclude.json** | Optional | List of dataset names/paths to hide from the UI. |
| **thumbnails/\<path\>.jpg** | Optional | Thumbnail per dataset; path = dataset path without extension. |
| **videos/\<path\>.mp4** | Optional | Video per dataset; path = dataset path without extension. |

- **Single dataset:** Use either one entry in `consolidated_datasets.json` or one entry in `data_index.json` plus one YAML in `dataset_info/`.
- **Multiple datasets / full function:** Either `consolidated_datasets.json` with many entries (best), or `data_index.json` with many paths and one YAML per path in `dataset_info/`.

---

## 3. File structures

### 3.1 `info/consolidated_datasets.json`

Object: keys = dataset path (no extension), values = one dataset object (same shape as one YAML).

```json
{
  "path/to/dataset1": { ...dataset object... },
  "path/to/dataset2": { ...dataset object... }
}
```

### 3.2 `info/data_index.json`

Either an array of YAML filenames, or an object whose keys are used as the list:

```json
["path/to/dataset1.yml", "path/to/dataset2.yaml"]
```
or
```json
{ "path/to/dataset1.yml": null, "path/to/dataset2.yaml": null }
```

Each path is stripped of `.yml`/`.yaml` to get the dataset path; the app then loads `dataset_info/<file>`.

### 3.3 `info/exclude.json`

Array of dataset names or paths to exclude from the grid:

```json
["dataset_to_hide", "another/path"]
```

### 3.4 `dataset_info/<path>.yml` (one per dataset)

One YAML file per dataset. Structure must match the per-dataset object used in `consolidated_datasets.json` (see **4. Dataset object (fields)** below).

---

## 4. Dataset object (fields)

Each dataset is one object (in consolidated JSON or in a YAML). The page uses these fields; others are optional and can be stored in `raw` or in the object for detail views.

| Field | Type | Used for |
|-------|------|----------|
| **path** | string | From the key in consolidated JSON, or from the YAML filename in YAML mode. |
| **dataset_name** | string | Name/hub id; fallback if path is empty. |
| **robot_type** | string | Robot model; filters + display. |
| **end_effector_type** | string or string[] | End effector(s); filters. |
| **scene_type** | string[] | Scene filter + display. |
| **atomic_actions** | string[] | Action filter + display. |
| **tasks** | string (or from task_descriptions[0]) | Task description on card. |
| **objects** | array of object items | Operation object filter (hierarchical). |
| **operation_platform_height** | number | Display. |
| **frame_range** | string | Filter + display. |
| **dataset_size** | string/number | Display. |
| **statistics** | object | Display. |

**Object item** (each element of `objects`):

| Field | Type |
|-------|------|
| **object_name** | string |
| **level1** … **level5** | string (optional) |

Levels form the hierarchy (e.g. level1 = category, level2 = subcategory). Missing levels can be null/omitted.

Optional metadata (used in detail/export): `dataset_uuid`, `language`, `task_categories`, `sub_tasks`, `annotations`, `authors`, `homepage`, `paper`, `repository`, `license`, `tags`, `cameras`, `citation_bibtex`, `depth_enabled`, `data_schema`, `structure`, etc.

---

## 5. Summary

- **Minimum for one dataset:**  
  Either `info/consolidated_datasets.json` with one key, or `info/data_index.json` with one entry + `dataset_info/<path>.yml` with the fields above.
- **Full function, multiple datasets:**  
  Prefer **info/consolidated_datasets.json** with all datasets (same field names and structure as above).  
  Alternatively: **info/data_index.json** (list of YAML filenames) + one **dataset_info/\<path\>.yml** per dataset.
- **Optional:** **info/exclude.json**, **thumbnails/\<path\>.jpg**, **videos/\<path\>.mp4** (path = dataset path without extension).

---

## 6. Mapping: page fields vs info.yaml (metadata output)

Generated assets are **info.yaml** (per dataset) under `dataset_info/` and inside `consolidated_datasets.json`. Their field names follow the **metadata schema**; the page uses the names in section 4. **Every field the page uses has exactly one mapping below** (either same name in info.yaml or a derivation rule). No page-used field is missing a correspondence.

**Mapping method:** When reading a dataset object (from consolidated JSON or from `dataset_info/<path>.yaml`), derive page fields as follows. Prefer doing this in the **frontend** when rendering; alternatively, a backend can add these keys to each object before serving.

### 6.1 Required / primary fields (section 4 table)

| Page field | Type (page) | info.yaml / mapping |
|------------|-------------|----------------------|
| **path** | string | Not in object. Use the **key** in `consolidated_datasets.json`, or the YAML **filename stem** in dataset_info. Equals `dataset_name` in our output. |
| **dataset_name** | string | **Same name.** `dataset_name`. |
| **robot_type** | string | **Map:** `robot_name` if present, else `device_model`, else `""`. |
| **end_effector_type** | string or string[] | **Same name.** `end_effector_type`. |
| **scene_type** | string[] | **Map:** info.yaml `scene_type` is object `{ level1, level2, level3, level4, level5 }`. Build array from non-null values: `[level1, level2, level3, level4, level5]` then drop null/empty. |
| **atomic_actions** | string[] | **Same name.** `atomic_actions`. |
| **tasks** | string | **Map:** If `task_instruction` is string, use it; if list, join with newline or use first element; else first element of `sub_tasks`, or `""`. |
| **objects** | array of { object_name, level1…level5 } | **Same name and shape.** `objects`. |
| **operation_platform_height** | number | **Same name.** `operation_platform_height`. |
| **frame_range** | string | **Map:** From `frame_num` and/or `statistics.total_frames`. Build display string (e.g. `"0–" + total_frames` or `frame_num` as-is). |
| **dataset_size** | string/number | **Same name** (top-level or inside statistics). Use top-level `dataset_size`, else `statistics.dataset_size`, else `""`. |
| **statistics** | object | **Same name.** `statistics`. |

### 6.2 Optional fields (detail/export, section 4 last paragraph)

| Page field | info.yaml / mapping |
|------------|----------------------|
| **dataset_uuid** | **Same name.** `dataset_uuid`. |
| **language** | **Same name.** `language`. |
| **task_categories** | **Same name.** `task_categories`. |
| **sub_tasks** | **Same name.** `sub_tasks`. |
| **annotations** | **Same name.** `annotations`. |
| **authors** | **Same name.** `authors`. |
| **homepage** | **Same name.** `homepage`. |
| **paper** | **Same name.** `paper`. |
| **repository** | **Same name.** `repository`. |
| **license** | **Same name.** `license`. |
| **tags** | **Same name.** `tags`. |
| **cameras** | **Map:** info.yaml has `came_info` (canonical) or alias `camera_info`. Use `came_info` or `camera_info` as page `cameras`. |
| **citation_bibtex** | **Same name.** `citation_bibtex`. |
| **depth_enabled** | **Same name.** `depth_enabled`. |
| **data_schema** | **Map:** Use `data_structure` (canonical) or alias `structure`. |
| **structure** | **Same name.** info.yaml exposes `structure` (alias of `data_structure`). Use `structure` or `data_structure`. |

### 6.3 data_index.json format

Current output is `{ "datasets": [ "<name1>", "<name2>", ... ], "count": N }`. Each entry is the dataset path **without** extension (same as the key in consolidated and the YAML filename stem). The page should load `dataset_info/<name>.yaml` (or `.yml`) for each; no need to append `.yml` in the request if the app resolves by stem.
