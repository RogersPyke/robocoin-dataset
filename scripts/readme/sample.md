---
task_categories:
  - robotics

language:
  - en
  - zh


extra_gated_prompt: 'By accessing this dataset, you agree to cite the associated paper in your research/publications—see the ''Citation'' section for details. You agree to not use the dataset to conduct experiments that cause harm to human subjects.'



extra_gated_fields:

  Country:
    type: 'country'
    description: 'e.g., ''Germany'', ''China'', ''United States'''

  Company/Organization:
    type: 'text'
    description: 'e.g., ''ETH Zurich'', ''Boston Dynamics'', ''Independent Researcher'''



tags:
  - RoboCOIN
  - LeRobot

license: apache-2.0

configs:
- config_name: default
  data_files: data/*/*.parquet
---

# Agibot_g1_Pick_apple

### Overview

- **Total Episodes:** 100
- **Total Frames:** 50000
- **FPS:** 30
- **Dataset Size:** 2.7GB
- **Robot Name:** `Agibot+G1edu-u3`
- **End-Effector Name:** `Agibot+two_finger_gripper, Agibot+three_finger_gripper`
- **Scene:** `home-kitchen`
- **Objects:** `table-furniture-table`, 
`basket-container-basket`

- **Task Description:** Left_arm+pick+apple### Primary Task Instruction
> pick up the apple from the table and place it into the basket.

### Robot Configuration

- **Robot Name:** `Agibot+G1edu-u3`
- **Robot Type:** `G1edu-u3`
- **Codebase Version:** `v2.1`
- **End-Effector Name:** `Agibot+two_finger_gripper, Agibot+three_finger_gripper`
- **End-Effector Type:** `two_finger_gripper, three_finger_gripper`
- **Teleoperation Type:** `cable, wireless`

## Scene and Objects

### Scene Type
`home-kitchen`

### Objects
- `table-furniture-table`
- `basket-container-basket`


## Task Descriptions

- **Standardized Task Name:** `Agibot_g1_Pick_apple`
- **Standardized Task Description:** `Left_arm+pick+apple`
- **Operation Type:** `simgle_arm`
- **Task Result:** `success`
- **Environment Type:** `simulation`

### Sub-Tasks
This dataset includes 3 distinct subtasks:

1. **Grasp the apple with the left gripper** 
2. **Place the apple into the basket with the left gripper** 
3. **End** 


### Atomic Actions
- `pick`
- `place`
- `grasp`


## Hardware and Sensors

### Sensors
- `Depth_camera`
- `RGB_camera`
- `IMU`
- `Force_sensor`


### Camera Information- Camera 1: RGB, 1280x720, 30fps
- Camera 2: RGB, 1280x720, 30fps
- Camera 3: Depth, 640x480, 30fps


### Coordinate System
- **Definition:** `right_hand_frame`
- **Origin (XYZ):** `[0, 0, 0]`

### Dimensions & Units
- **Joint Rotation:** `radian`
- **End-Effector Rotation:** `radian`
- **End-Effector Translation:** `meter`
- **Base Rotation:** `radian`
- **Base Translation:** `meter`
- **Operation Platform Height:** `77.2 cm`

## Dataset Statistics

| Metric | Value |
|--------|-------|
| **Total Episodes** | 100 |
| **Total Frames** | 50000 |
| **Total Tasks** | 100 |
| **Total Videos** | 100 |
| **Total Chunks** | 10 |
| **Chunk Size** | 10 |
| **FPS** | 30 |
| **Total Duration** | 27:46:40 |
| **Video Resolution** | 1280x720 |
| **State Dimensions** | 14 |
| **Action Dimensions** | 7 |
| **Camera Views** | 3 |
| **Dataset Size** | 2.7GB |

- **Frame Range Label:** `10K-100K`
- **Data Schema:** LeRobot-compatible format with parquet data files and MP4 video files

## Data Splits

The dataset is organized into the following splits:

- **Training**: Episodes 0:89
- **Validation**: Episodes 89:99
- **Test**: Episodes 99:109


## Dataset Structure

This dataset follows the LeRobot format and contains the following components:

### Data Files
- **Videos**: Compressed video files containing RGB camera observations
- **State Data**: Robot joint positions, velocities, and other state information
- **Action Data**: Robot action commands and trajectories
- **Metadata**: Episode metadata, timestamps, and annotations

### File Organization
- **Data Path Pattern**: `data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet`
- **Video Path Pattern**: `videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4`
- **Chunking**: Data is organized into 10 chunk(s)
of size 10

### Data Structure (Tree)

```
dataset/
├── data/
│   ├── chunk-000/
│   │   ├── episode_000000.parquet
│   │   └── episode_000001.parquet
│   └── chunk-001/
│       ├── episode_000010.parquet
│       └── episode_000011.parquet
├── videos/
│   ├── chunk-000/
│   │   ├── camera_1/
│   │   │   ├── episode_000000.mp4
│   │   │   └── episode_000001.mp4
│   │   └── camera_2/
│   │       ├── episode_000000.mp4
│   │       └── episode_000001.mp4
│   └── chunk-001/
│       └── ...
├── meta/
│   └── info.json
└── README.md

```

## Camera Views

This dataset includes 3 camera views.

## Features (Full YAML)

```yaml
observation.images.camera_1:
  dtype: video
  shape:
  - 720
  - 1280
  - 3
  names:
  - height
  - width
  - channels
  info:
    video.fps: 30
    video.codec: h264
observation.state:
  dtype: float32
  shape:
  - 14
  names:
  - joint_1
  - joint_2
  - '...'
action:
  dtype: float32
  shape:
  - 7
  names:
  - action_1
  - action_2
  - '...'
timestamp:
  dtype: float64
frame_index:
  dtype: int64
episode_index:
  dtype: int64
subtask_annotation:
  dtype: string
scene_annotation:
  dtype: string
eef_sim_pose_state:
  dtype: float32
  shape:
  - 7
  names:
  - x
  - y
  - z
  - qx
  - qy
  - qz
  - qw

```

## Meta Information

The complete dataset metadata is available in [meta/info.json](meta/info.json).

### Directory Structure

The dataset is organized as follows (showing leaf directories with first 5 files only):

```
dataset/
├── data/
│   └── chunk-*/episode_*.parquet
├── videos/
│   └── chunk-*/camera_*/episode_*.mp4
├── meta/
│   └── info.json
└── README.md

```

## Available Annotations

This dataset includes rich annotations to support diverse learning approaches:

- `subtask_annotation`
- `scene_annotation`
- `eef_direction`
- `eef_velocity`
- `eef_acc_mag`
- `gripper_mode`
- `gripper_activity`


## Dataset Tags

- `RoboCOIN`
- `LeRobot`


## Authors

### Contributors
This dataset is contributed by:-[RoboCOIN](https://flagopen.github.io/RoboCOIN/) - RoboCOIN Team

### Annotators
This dataset is annotated by:- RoboCOIN- [https://flagopen.github.io/RoboCOIN/](https://flagopen.github.io/RoboCOIN/)-  - RoboCOIN Team

## Links

- **Homepage:** [https://flagopen.github.io/RoboCOIN/](https://flagopen.github.io/RoboCOIN/)
- **Paper:** [https://arxiv.org/abs/2511.17441](https://arxiv.org/abs/2511.17441)
- **Repository:** [https://github.com/FlagOpen/RoboCOIN](https://github.com/FlagOpen/RoboCOIN)
- **License:** apache-2.0

## Contact and Support

For questions, issues, or feedback regarding this dataset, please contact:
- **Email:** robocoin@baai.ac.cn
For questions, issues, or feedback regarding this dataset, please contact us.
### Support
For technical support, please open an issue on our GitHub repository.

## License

This dataset is released under the **apache-2.0** license.

Please refer to the LICENSE file for full license terms and conditions.

## Citation

If you use this dataset in your research, please cite:

```bibtex
@article{robocoin,
  title={RoboCOIN: An Open-Sourced Bimanual Robotic Data Collection for Integrated Manipulation},
  author={...},
  journal={arXiv preprint arXiv:2511.17441},
  url={https://arxiv.org/abs/2511.17441},
  year={2025}
}

```

### Additional References

If you use this dataset, please also consider citing:
- LeRobot Framework: https://github.com/huggingface/lerobot

## Version Information

## Version History
- v1.0.0 (2025-11): Initial release


## Dataset Description

This dataset uses an extended format based on LeRobot and is fully compatible with LeRobot.