---
task_categories:
  - task_categories

language:
  - language


extra_gated_prompt: 'extra_gated_prompt'



extra_gated_fields:

  extra_gated_fields:
    type: 'text'
    description: 'extra_gated_fields'



tags:
  - tags

license: license

configs:
- config_name: configs
  data_files: configs
---

# dataset_name

### Overview

- **Total Episodes:** None
- **Total Frames:** None
- **FPS:** None
- **Dataset Size:** dataset_size
- **Robot Name:** `robot_name`
- **End-Effector Type:** `end_effector_type`
- **Teleoperation Type:** `tele_type`
- **Sensors:** `sensor_list`

- **Camera Information:** came_info

- **Scene:** `scene_type-scene_type`
- **Objects:** `objects-objects`

- **Task Description:** task_description



### Robot Configuration

- **Robot Name:** `robot_name`

- **Codebase Version:** `codebase_version`

- **End-Effector Type:** `end_effector_type`
- **Teleoperation Type:** `tele_type`

## Scene and Objects

### Scene Type
`scene_type-scene_type`

### Objects
- `objects-objects`


## Task Descriptions

- **Standardized Task Name:** `task_name`
- **Standardized Task Description:** `task_description`
- **Operation Type:** `task_operation_type`
- **Task Result:** `task_result`
- **Environment Type:** `env_type`

### Sub-Tasks
This dataset includes 1 distinct subtasks:

1. **sub_tasks** 


### Atomic Actions
- `atomic_actions`


## Hardware and Sensors

### Sensors
- `sensor_list`


### Camera Information- came_info


### Coordinate System
- **Definition:** `coordinate_definition`
- **Origin (XYZ):** `origin_xyz`

### Dimensions & Units
- **Joint Rotation:** `joint_rotation_dim`
- **End-Effector Rotation:** `end_rotation_dim`
- **End-Effector Translation:** `end_translation_dim`
- **Base Rotation:** `base_robtation_dim`
- **Base Translation:** `base_translation_dim`
- **Operation Platform Height:** `operation_platform_height cm`

## Dataset Statistics

| Metric | Value |
|--------|-------|
| **Dataset Size** | dataset_size |




## Data Splits

The dataset is organized into the following splits:

- **Training**: Episodes splits
- **Validation**: Episodes splits
- **Test**: Episodes splits


## Dataset Structure

This dataset follows the LeRobot format and contains the following components:

### Data Files
- **Videos**: Compressed video files containing RGB camera observations
- **Depth Maps**: Depth information from depth cameras
- **State Data**: Robot joint positions, velocities, and other state information
- **Action Data**: Robot action commands and trajectories
- **Metadata**: Episode metadata, timestamps, and annotations

### File Organization
- **Data Path Pattern**: `data_path`
- **Video Path Pattern**: `video_path`

### Data Structure (Tree)

```
data_structure
```

## Camera Views



## Features (Full YAML)



## Meta Information

The complete dataset metadata is available in [meta/info.json](meta/info.json).



## Available Annotations

This dataset includes rich annotations to support diverse learning approaches:

- annotations

## Dataset Tags

- `tags`


## Authors

### Contributors
This dataset is contributed by:-authors

### Annotators
This dataset is annotated by:- authors

## Links

- **Homepage:** [homepage](homepage)
- **Paper:** [paper](paper)
- **Repository:** [repository](repository)
- **License:** license

## Contact and Support

contact_info
### Support
support_info

## License

This dataset is released under the **license** license.

license_details

## Citation

If you use this dataset in your research, please cite:

```bibtex
citation_bibtex
```

### Additional References

additional_citations

## Version Information

version_info

## Dataset Description

dataset_description